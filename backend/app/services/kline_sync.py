"""日 K 同步服务(§7.7 Step 1)。

调度器在 capability 允许下,把符号集合的日 K 批量同步到本地 Parquet。
策略:
  - 日 K 仅使用 `kline.daily.batch`
  - 除权因子仅使用 `adj_factor`
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import NamedTuple

import polars as pl

from app.data_providers.base import AssetType, ProviderRequestError
from app.data_providers.minute_normalizer import (
    CANONICAL_MINUTE_COLS as CANONICAL_MINUTE_COLS,
)
from app.data_providers.minute_normalizer import (
    _compact_klines_to_df,
    _datetime_to_ms,
    _enforce_minute_beijing_wallclock,
    _normalize_minute,
    _timestamp_to_beijing_datetime,
)
from app.data_providers.minute_stream import iter_minute_frames
from app.indicators.pipeline import filter_halt_days
from app.market_time import CN_TZ, cn_now, cn_today
from app.parquet import read_parquet_safe, scan_daily_parquet
from app.services import minute_adjust, preferences
from app.services.shared_work import SharedWork, WorkBusyError
from app.tickflow.capabilities import Cap, CapabilitySet
from app.tickflow.client import get_client
from app.tickflow.rate_limits import chunked, resolve_limit, sleep_between_batches
from app.tickflow.repository import KlineRepository, replace_with_retry
from app.tickflow.request_guard import (
    SDKCallTimeoutError as SDKCallTimeoutError,
)
from app.tickflow.request_guard import (
    _sdk_call_with_deadline,
)

logger = logging.getLogger(__name__)

_single_minute_work = SharedWork("minute.single", max_pending=128, max_entries=512,
                               max_bytes=16 * 1024 * 1024, size_of=lambda frame: frame.estimated_size())


def _log_cap_unavailable(cap_id: str, action: str) -> None:
    """门禁未通过时记一条带生效源的日志。

    原实现是静默 return —— 切换数据源后「什么都没发生」且无迹可循。
    带源标签后可一眼看出是路由到不可用源, 还是 TickFlow 档位不足。
    """
    try:
        from app.data_providers.capabilities import capability_gate
        _, label = capability_gate(cap_id)
    except Exception:  # noqa: BLE001 — 仅日志, 失败不应影响主流程
        label = cap_id
    logger.info("%s 跳过: %s 不可用", action, label)

def _atomic_write_parquet(df: pl.DataFrame, out) -> None:
    """先写临时文件再原子替换, 避免进程中断留下损坏的 parquet。

    与 repository._atomic_write_parquet 同语义。adj_factor 的 all.parquet 是全市场
    单文件、每次「读→concat→原地写」, 直接 write_parquet(out) 在进程被 kill
    (dev.sh 清端口用 kill -9)、reap 超时或断电时会留下半截文件, 之后复权视图
    scan_parquet 整条链路报错、enriched 全市场重算不出。临时文件后缀 .tmp 不匹配
    *.parquet glob, 不会被扫描误读。Windows 下目标正被并发读取时由
    replace_with_retry 短退避穿过。
    """
    # 唯一后缀: 两个写入者共用固定名 tmp 并发写同一文件会交错损坏 (被 rename
    # 定版后分区报废)。崩溃残留的孤儿 tmp 不匹配 *.parquet glob, 读侧无影响。
    tmp = out.with_name(f"{out.name}.{uuid.uuid4().hex[:8]}.tmp")
    df.write_parquet(tmp)
    replace_with_retry(tmp, out)


def _daily_dates(df: pl.DataFrame) -> list[date]:
    if df.is_empty() or "date" not in df.columns:
        return []
    out: list[date] = []
    for value in df["date"].unique().to_list():
        if isinstance(value, datetime):
            out.append(value.date())
        elif isinstance(value, date):
            out.append(value)
        else:
            try:
                out.append(date.fromisoformat(str(value)[:10]))
            except ValueError:
                continue
    return sorted(set(out))


def _replace_existing_daily_partitions(repo: KlineRepository, df: pl.DataFrame, table: str = "kline_daily") -> None:
    days = _daily_dates(df)
    if not days:
        return
    from app.services.data_integrity import prune_table_partition_dates

    removed = prune_table_partition_dates(repo.store.data_dir, table, days)
    if removed:
        logger.info("%s 覆盖修复: 已删除 %d 个旧日期分区", table, removed)


# 标准列(无论 SDK 返回什么形状,我们把它规范成这套)
CANONICAL_DAILY_COLS = [
    "symbol", "date", "open", "high", "low", "close", "volume", "amount",
]


def _normalize_sync_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    if not symbol:
        return ""
    explicit_exchange = None
    if "." in symbol and symbol.rsplit(".", 1)[-1] in {"SH", "SZ", "BJ"}:
        explicit_exchange = symbol.rsplit(".", 1)[-1]
    for prefix in ("SH", "SZ", "BJ"):
        if symbol.startswith(prefix) and len(symbol) >= 8:
            explicit_exchange = prefix
            symbol = symbol[len(prefix):]
            break
    if "." in symbol:
        symbol = symbol.split(".", 1)[0]
    code = symbol.zfill(6) if symbol.isdigit() else symbol
    return f"{code}.{explicit_exchange}" if explicit_exchange else code


def _dedupe_symbols(symbols: list[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in symbols:
        symbol = _normalize_sync_symbol(value)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _stockformer_top_symbols(repo: KlineRepository, top_n: int) -> list[str]:
    try:
        from app.services.stockformer import multifactor_select, predict

        result = multifactor_select(repo.store.data_dir, limit=top_n)
        rows = result.get("rows") or []
        if not rows:
            result = predict(repo.store.data_dir, limit=top_n)
            rows = result.get("rows") or []
        return _dedupe_symbols([row.get("symbol") for row in rows[:top_n] if isinstance(row, dict)])
    except Exception as exc:  # noqa: BLE001
        logger.debug("resolve stockformer minute universe failed: %s", exc)
        return []


def _holding_symbols() -> list[str]:
    try:
        from app.services.aiagents_bridge import legacy_holdings

        data = legacy_holdings()
        rows = data.get("positions") or []
        return _dedupe_symbols([row.get("symbol") for row in rows if isinstance(row, dict)])
    except Exception as exc:  # noqa: BLE001
        logger.debug("resolve holdings minute universe failed: %s", exc)
        return []


def resolve_minute_sync_universe(
    repo: KlineRepository,
    capset: CapabilitySet | None = None,  # noqa: ARG001
    *,
    scope: str | None = None,
    symbols: list[str] | str | None = None,
    top_n: int | None = None,
) -> tuple[list[str], str]:
    """解析分钟K同步范围，避免默认全市场拉取导致超时。"""
    scope_value = (scope or preferences.get_minute_sync_scope()).lower()
    top_n_value = max(1, min(1000, int(top_n or preferences.get_minute_sync_top_n())))
    if isinstance(symbols, str):
        manual_symbols = _dedupe_symbols(symbols.replace("，", ",").replace("\n", ",").split(","))
    elif symbols is not None:
        manual_symbols = _dedupe_symbols(symbols)
    else:
        manual_symbols = preferences.get_minute_sync_symbols()

    def _watchlist() -> list[str]:
        try:
            from app.tickflow.pools import get_pool

            return _dedupe_symbols(get_pool("watchlist"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("resolve watchlist minute universe failed: %s", exc)
            return []

    def _all_a() -> list[str]:
        all_symbols: list[str] = []
        try:
            from app.tickflow.pools import get_pool

            all_symbols.extend(get_pool("CN_Equity_A"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("resolve CN_Equity_A minute universe failed: %s", exc)
        try:
            inst_path = repo.store.data_dir / "instruments" / "instruments.parquet"
            if inst_path.exists():
                inst = read_parquet_safe(inst_path, columns=["symbol"])
                if inst is not None and "symbol" in inst.columns:
                    all_symbols.extend(inst["symbol"].to_list())
        except Exception as exc:  # noqa: BLE001
            logger.debug("resolve instruments minute universe failed: %s", exc)
        try:
            index_set = repo.get_index_symbol_set()
            return [s for s in _dedupe_symbols(all_symbols) if s not in index_set]
        except Exception:
            return _dedupe_symbols(all_symbols)

    if scope_value == "manual":
        return manual_symbols, "手动标的"
    if scope_value == "watchlist":
        return _watchlist(), "自选股"
    if scope_value == "realtime_watchlist":
        return _dedupe_symbols(preferences.get_realtime_watchlist_symbols()), "实时盯盘列表"
    if scope_value == "holdings":
        return _holding_symbols(), "持仓股"
    if scope_value == "stockformer_top":
        return _stockformer_top_symbols(repo, top_n_value), f"StockFormer Top{top_n_value}"
    if scope_value == "all":
        return _all_a(), "全市场A股"

    combined = _dedupe_symbols([
        *_holding_symbols(),
        *_watchlist(),
        *_stockformer_top_symbols(repo, top_n_value),
        *manual_symbols,
    ])
    return combined, f"持仓+自选+推荐Top{top_n_value}"


def _normalize_daily(df_in, default_symbol: str | None = None) -> pl.DataFrame:
    """把 SDK 返回的 pandas/任意 DataFrame 规范成 canonical 列。"""
    if df_in is None or len(df_in) == 0:
        return pl.DataFrame()

    if not isinstance(df_in, pl.DataFrame):
        df = pl.from_pandas(df_in.reset_index() if hasattr(df_in, "reset_index") else df_in)
    else:
        df = df_in

    # 兼容字段名差异
    rename_map = {
        "ts_code": "symbol",
        "trade_date": "date",
        "vol": "volume",
        "amt": "amount",
        "datetime": "date",
    }
    df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})

    if "symbol" not in df.columns and default_symbol is not None:
        df = df.with_columns(pl.lit(default_symbol).alias("symbol"))

    # 类型规范
    if "date" in df.columns and df.schema["date"] != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date, strict=False))

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
    for col in ("volume", "amount"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # 过滤停牌日 (open/high 为 0; close 可能被填充为前收盘价, 不能用全零判断)
    df = filter_halt_days(df)

    # 只保留 canonical 列
    keep = [c for c in CANONICAL_DAILY_COLS if c in df.columns]
    return df.select(keep)


def _instruments_symbol_index(data_dir) -> dict[str, list[str]] | None:
    """裸代码 → instruments 维表完整符号列表; 维表缺失/不可读返回 None。"""
    path = data_dir / "instruments" / "instruments.parquet"
    if not path.exists():
        return None
    try:
        symbols = pl.read_parquet(path, columns=["symbol"])["symbol"].drop_nulls().to_list()
    except Exception as e:
        logger.warning("instruments 维表读取失败, 裸符号无法补全后缀: %s", e)
        return None
    index: dict[str, list[str]] = {}
    for s in symbols:
        code, _, _suffix = str(s).partition(".")
        if code:
            index.setdefault(code, []).append(str(s))
    return index


def _normalize_bare_symbols(
    symbols: list[str], data_dir,
) -> tuple[list[str], dict[str, str], list[str]]:
    """裸符号(无交易所后缀, 如 600000)按 instruments 维表补全为 600000.SH (#302)。

    返回 (规范化后的去重请求列表, {裸符号: 补全后符号}, 被跳过的裸符号)。
    上游对裸符号返回 200 空 payload, 本地静默写 0 行 — 补全失败时显式跳过并告警,
    不再带着已知无效的符号发请求。全部带后缀时不读维表, 行为与开销同旧版;
    带后缀符号原样透传, 由上游照常处理。
    """
    bare = [s for s in symbols if "." not in s]
    if not bare:
        return symbols, {}, []

    index = _instruments_symbol_index(data_dir)
    resolved: dict[str, str] = {}
    skipped: list[str] = []
    for s in bare:
        matches = (index or {}).get(s, [])
        if len(matches) == 1:
            resolved[s] = matches[0]
        else:
            skipped.append(s)

    if index is None:
        logger.warning(
            "日K同步: instruments 维表不可用, %d 个裸符号(无交易所后缀)将被跳过 — "
            "请先同步标的维表, 或改用带后缀符号 (如 600000.SH)", len(skipped))
    elif skipped:
        logger.warning(
            "日K同步: %d 个裸符号在维表中无唯一匹配(不支持或非股票), 已跳过: %s",
            len(skipped), skipped[:20])
    if resolved:
        logger.info("日K同步: 已按维表补全 %d 个裸符号后缀 (样例: %s)",
                    len(resolved), list(resolved.items())[:5])

    out: list[str] = []
    seen: set[str] = set()
    skipped_set = set(skipped)
    for s in symbols:
        if s in skipped_set:
            continue  # 无法补全的裸符号不发请求 (上游必然 200 空 payload)
        full = resolved.get(s, s)
        if full not in seen:
            seen.add(full)
            out.append(full)
    return out, resolved, skipped


def sync_daily_batch(symbols: list[str],
                     count: int | None = None,
                     batch_size: int | None = None,
                     rpm: int | None = None,
                     start_time: datetime | None = None,
                     end_time: datetime | None = None,
                     on_chunk_done: Callable[[int, int], None] | None = None,
                     failed_out: list[str] | None = None) -> pl.DataFrame:
    """批量拉取多股日 K。

    优先使用 start_time / end_time 区间 + count=10000,确保覆盖完整时间段。
    仅传 count 时按条数回溯。

    failed_out: 可选出参。拉取失败的分块标的会追加进该 list, 供上层判定「部分失败」
                而非静默当成功(某分块断网 → 这些标的本轮未更新, 保持旧数据)。
    """
    tf = get_client()
    out: list[pl.DataFrame] = []
    chunks = chunked(symbols, batch_size)
    failed_syms: list[str] = []

    for i, chunk in enumerate(chunks):
        sleep_between_batches(i, rpm)
        try:
            if start_time and end_time:
                raw = _sdk_call_with_deadline(lambda chunk=chunk: tf.klines.batch(
                    chunk, period="1d", adjust="none",
                    start_time=_datetime_to_ms(start_time),
                    end_time=_datetime_to_ms(end_time),
                    count=10000,
                    as_dataframe=False, show_progress=False,
                ))
            else:
                raw = _sdk_call_with_deadline(lambda chunk=chunk: tf.klines.batch(
                    chunk, period="1d", count=count or 250, adjust="none",
                    as_dataframe=False, show_progress=False))
        except SDKCallTimeoutError as e:
            logger.warning("batch fetch timed out for %d symbols (chunk %d/%d): %s",
                           len(chunk), i + 1, len(chunks), e)
            failed_syms.extend(chunk)
            if on_chunk_done:
                # 心跳: 分块失败也推进进度/取消检查点, 防持续超时的同步
                # 长时间无上报(reap 误判)且协作取消无边界可退出。
                on_chunk_done(i + 1, len(chunks))
            continue
        except Exception as e:  # noqa: BLE001
            logger.warning("batch fetch failed for %d symbols (chunk %d/%d): %s",
                           len(chunk), i + 1, len(chunks), e)
            failed_syms.extend(chunk)
            continue

        # False 直转: timestamp(UTC 毫秒) → 北京墙钟 datetime 列,
        # _normalize_daily 将 datetime 映射为 date — 与 SDK True 路径的
        # trade_date 字符串列同口径 (fromtimestamp(ts/1000, Asia/Shanghai))。
        seg = _compact_klines_to_df(raw)
        if not seg.is_empty():
            seg = seg.with_columns(_timestamp_to_beijing_datetime(pl.col("timestamp")).alias("datetime")).drop("timestamp")
            out.append(_normalize_daily(seg))

        if on_chunk_done:
            on_chunk_done(i + 1, len(chunks))

    # 部分失败可见化: 聚合一条 WARNING(而非只有逐块 debug/warning), 并回传出参。
    if failed_syms:
        logger.warning("日K批量同步部分失败: %d/%d 标的未获取, 本轮保持旧数据 (样例: %s)",
                       len(failed_syms), len(symbols), failed_syms[:10])
        if failed_out is not None:
            failed_out.extend(failed_syms)

    if not out:
        return pl.DataFrame()
    return pl.concat(out, how="diagonal_relaxed")


def iter_custom_daily_frames(provider, symbols, *, asset_type="stock", **kwargs):
    """复用流式/单帧插件契约; 旧股票插件可继续省略 asset_type 参数。"""
    if asset_type != "stock":
        kwargs["asset_type"] = asset_type
    stream = getattr(provider, "iter_daily", None)
    if callable(stream):
        yield from stream(symbols, **kwargs)
    else:
        frame = provider.get_daily(symbols, **kwargs)
        if not frame.is_empty():
            yield frame


class DailySourceUnavailableError(RuntimeError):
    """有效日K路由没有可用数据能力。"""


def iter_daily_frames(
    symbols: list[str], capset: CapabilitySet, *,
    start_time: datetime, end_time: datetime, asset_type: str = "stock", on_chunk_done=None,
):
    """只取标准日K帧, 不写正式数据、不计算指标; 调用方决定校验和发布边界。"""
    if asset_type not in {"stock", "etf", "index"}:
        raise ValueError("不支持的日K资产类型")
    from app.data_providers import custom

    provider_name = preferences.get_daily_data_provider()
    logger.info("日K修补取数: provider=%s asset=%s symbols=%d start=%s end=%s",
                provider_name, asset_type, len(symbols), start_time.isoformat(), end_time.isoformat())
    if provider_name != "tickflow" and custom.provider_has_dataset(provider_name, "daily"):
        found = False
        for frame in iter_custom_daily_frames(
            custom.get_provider(provider_name), symbols, asset_type=asset_type,
            start_time=start_time, end_time=end_time, on_chunk_done=on_chunk_done,
        ):
            if not frame.is_empty():
                found = True
                yield frame
        # 非股票源沿用指数/ETF 原有的「明确不支持时回退」契约。
        if found or asset_type == "stock":
            return
        logger.warning("日K源 %s 未返回 %s 数据, 修补回退 TickFlow", provider_name, asset_type)
    elif provider_name != "tickflow":
        logger.warning("数据源 %s 未提供 daily 能力, 修补回退 TickFlow", provider_name)
    if not capset.has(Cap.KLINE_DAILY_BATCH):
        raise DailySourceUnavailableError("当前日K数据源不可用, 请检查数据源配置")
    limit = resolve_limit(capset, Cap.KLINE_DAILY_BATCH, default_batch=100)
    failures: list[str] = []
    frame = sync_daily_batch(
        symbols, start_time=start_time, end_time=end_time,
        batch_size=limit.batch, rpm=limit.rpm, on_chunk_done=on_chunk_done, failed_out=failures,
    )
    if failures:
        raise RuntimeError(f"日K获取失败: {', '.join(failures[:10])}")
    if not frame.is_empty():
        yield frame


def sync_and_persist_daily_batch(
    symbols: list[str],
    repo: KlineRepository,
    capset: CapabilitySet,
    count: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    on_chunk_done: Callable[[int, int], None] | None = None,
    replace_existing: bool = False,
    zero_row_out: list[str] | None = None,
    failed_out: list[str] | None = None,
) -> int:
    """批量同步日 K 并落到 Parquet。返回写入的行数。

    start_date/end_date: 外部传入的时间范围(由 pipeline 根据已有数据计算)。
    未传入时默认拉最近 1 年。
    zero_row_out: 可选出参。本轮实际入库 0 行的标的(含无法识别被跳过的裸符号、
                  上游 200 空 payload 的静默 0 行)按调用方原始写法追加进该 list,
                  供上层 fail-loud 展示, 不再"显示成功实则全空" (#302)。
    failed_out: 可选出参。分块拉取失败(超时/异常)的标的, 供上层把单票失败
                上浮为 stage_errors 而非静默当成功 (#514)。
    """
    seen: set[str] = set()
    failed_syms: list[str] = []
    skipped_bare: list[str] = []

    def _finalize(written: int) -> int:
        if failed_out is not None and failed_syms:
            # 单票失败上浮给调用方 (daily_pipeline 汇入 stage_errors, #514)
            failed_out.extend(failed_syms)
        zero_full = [s for s in requested if s not in seen and s not in failed_syms]
        if zero_full or skipped_bare:
            zero = [orig_by_full.get(s, s) for s in zero_full] + skipped_bare
            logger.warning(
                "日K批量同步: %d/%d 标的本轮入库 0 行 (符号无法识别/停牌/窗口内无数据; 样例: %s)",
                len(zero), len(requested) + len(skipped_bare), zero[:10])
            if zero_row_out is not None:
                zero_row_out.extend(zero)
        return written

    original = list(symbols)
    symbols, resolved, skipped_bare = _normalize_bare_symbols(symbols, repo.store.data_dir)
    requested = symbols
    # 完整符号 → 调用方原始写法: 0 行回报用用户输入的形式, 裸符号也能对上
    orig_by_full: dict[str, str] = {}
    for orig in original:
        orig_by_full.setdefault(resolved.get(orig, orig), orig)

    if not symbols:
        return _finalize(0)

    provider_name = preferences.get_daily_data_provider()
    custom_df = pl.DataFrame()
    if provider_name != "tickflow":
        from app.data_providers import custom as custom_sources

        # 日K是可按批次替换的低频数据: 主自定义源明确失败/空结果时,
        # 仅尝试雪球作为末级备源；主源已有有效结果时不混源覆盖。
        candidates = [provider_name]
        if provider_name != "xueqiu":
            candidates.append("xueqiu")
        end_time = end_date or datetime.now()
        days = count or 365
        start_time = start_date or (end_time - timedelta(days=days))
        for candidate in candidates:
            if not custom_sources.provider_has_dataset(candidate, "daily"):
                continue
            streaming_attempt = False
            try:
                provider = custom_sources.get_provider(candidate)
                if candidate == "xueqiu" and not getattr(provider, "history_auth_configured", lambda: True)():
                    logger.info("雪球未配置历史会话凭据，跳过日K备源")
                    continue
                iter_daily = getattr(provider, "iter_daily", None)
                if callable(iter_daily) and not replace_existing:
                    streaming_attempt = True

                    def _tracking_chunks(chunks):
                        for df in chunks:
                            if not df.is_empty() and "symbol" in df.columns:
                                seen.update(df["symbol"].cast(pl.Utf8).unique().to_list())
                            yield df

                    return _finalize(_persist_daily_chunks(
                        _tracking_chunks(iter_daily(
                            symbols,
                            start_time=start_time,
                            end_time=end_time,
                            on_chunk_done=on_chunk_done,
                        )),
                        repo,
                    ))
                candidate_df = provider.get_daily(
                    symbols,
                    start_time=start_time,
                    end_time=end_time,
                    on_chunk_done=on_chunk_done,
                )
            except Exception as exc:
                if streaming_attempt:
                    raise
                logger.warning("日K源 %s 请求失败，尝试下一源: %s", candidate, type(exc).__name__)
                continue
            if candidate_df is not None and not candidate_df.is_empty():
                custom_df = candidate_df
                if candidate != provider_name:
                    logger.warning("日K主源 %s 无有效结果，已切换雪球末级备源", provider_name)
                break

        if not custom_df.is_empty():
            df = custom_df
            seen.update(df["symbol"].cast(pl.Utf8).unique().to_list())
            if replace_existing:
                if "quote_ts" in df.columns:
                    df = df.drop("quote_ts")
                _replace_existing_daily_partitions(repo, df)
            repo.append_daily(df)
            d = repo.store.data_dir.as_posix()
            repo.refresh_view("kline_daily", f"{d}/kline_daily/**/*.parquet")
            return _finalize(df.height)
        # 自定义源未配置 daily → 回退 TickFlow

    if not capset.has(Cap.KLINE_DAILY_BATCH):
        _log_cap_unavailable("daily", "日K同步")
        return _finalize(0)

    limit = resolve_limit(capset, Cap.KLINE_DAILY_BATCH, default_batch=100)

    end_time = end_date or datetime.now()
    start_time = start_date or (end_time - timedelta(days=365))

    df = sync_daily_batch(
        symbols, count=count, batch_size=limit.batch, rpm=limit.rpm,
        start_time=start_time, end_time=end_time,
        on_chunk_done=on_chunk_done,
        failed_out=failed_syms,
    )

    if not df.is_empty() and "symbol" in df.columns:
        seen.update(df["symbol"].cast(pl.Utf8).unique().to_list())

    if df.is_empty():
        return _finalize(0)

    if replace_existing:
        if "quote_ts" in df.columns:
            df = df.drop("quote_ts")
        _replace_existing_daily_partitions(repo, df)

    repo.append_daily(df)

    d = repo.store.data_dir.as_posix()
    repo.refresh_view("kline_daily", f"{d}/kline_daily/**/*.parquet")

    return _finalize(df.height)


def _persist_daily_chunks(chunks, repo: KlineRepository) -> int:
    """先把流式 provider 结果写入私有 staging,完整取数后再提交正式分区。"""
    staging_base = repo.store.data_dir / ".daily_sync_staging"
    _sweep_stale_daily_staging(staging_base)
    root = staging_base / uuid.uuid4().hex
    written = 0
    try:
        for index, df in enumerate(chunks):
            if df.is_empty():
                continue
            df = repo._validate_asset_identity("stock", df)
            for date_df in df.partition_by("date"):
                dt = date_df["date"][0]
                ds = dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
                out = root / f"date={ds}" / f"part-{index}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                date_df.write_parquet(out)
                written += date_df.height

        date_dirs = sorted(root.glob("date=*"))
        for date_dir in date_dirs:
            files = sorted(date_dir.glob("*.parquet"))
            if not files:
                continue
            target_dir = repo.store.data_dir / "kline_daily" / date_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(target_dir.glob("*.parquet"))
            if not existing:
                for index, staged in enumerate(files):
                    replace_with_retry(staged, target_dir / f"part-stream-{index}.parquet")
                continue
            # 当天分区可能混有实时快照写的 quote_ts 列, 按存储 schema 读才不抛 SchemaError。
            base = scan_daily_parquet(existing).collect(engine="streaming")
            incoming = pl.scan_parquet(files).collect(engine="streaming")
            merged = pl.concat([base, incoming], how="diagonal_relaxed").unique(
                subset=["symbol", "date"], keep="last"
            )
            repo.append_daily(merged)
            for part in target_dir.glob("part-stream-*.parquet"):
                part.unlink(missing_ok=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        with contextlib.suppress(OSError):
            root.parent.rmdir()

    return written


def _sweep_stale_daily_staging(staging_base, max_age_s: int = 24 * 60 * 60) -> None:
    """清理崩溃遗留的旧同步目录,不碰仍可能活跃的新目录。"""
    if not staging_base.exists():
        return
    cutoff = time.time() - max_age_s
    for run_dir in staging_base.iterdir():
        try:
            if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff:
                shutil.rmtree(run_dir)
        except OSError:
            logger.warning("failed to clean stale daily staging: %s", run_dir)


def sync_daily_by_quotes(repo: KlineRepository) -> int:
    """用实时行情接口拉全市场当日数据,覆写 kline_daily 今天分区。

    一个请求覆盖 ~5500 只股票,比 batch K-line 快几个数量级。
    返回写入的行数。
    """
    from datetime import date as _date

    from app.tickflow.client import get_client

    tf = get_client()
    try:
        resp = tf.quotes.get_by_universes(universes=["CN_Equity_A"])
    except Exception as e:
        logger.warning("get_by_universes failed: %s", e)
        return 0

    if not resp:
        logger.warning("get_by_universes returned empty")
        return 0

    records = []
    for q in resp:
        ext = q.get("ext") or {}
        records.append({
            "symbol": q.get("symbol"),
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("last_price"),
            "volume": q.get("volume"),
            "amount": q.get("amount"),
            # 快照时刻标记: data_integrity 靠 quote_ts 区分盘中快照与盘后权威历史,
            # 缺失会让盘中覆写的分区在停机后被当成完整历史, 永远不进修复。
            "quote_ts": q.get("timestamp"),
        })

    df = pl.DataFrame(records)
    if df.is_empty():
        return 0
    df = df.with_columns(pl.col("quote_ts").cast(pl.Int64, strict=False))

    # 分区日期用北京交易日 (与 quote_service._build_daily 的 cn_today 一致),
    # 避免 UTC 服务器在盘中把日分区写成服务器本地日期。
    today = cn_today()
    daily_df = df.with_columns(pl.lit(today).cast(pl.Date).alias("date"))

    # 过滤停牌 (open/high 为 0; close 可能被填充为前收盘价, 不能用全零判断)
    daily_df = filter_halt_days(daily_df)

    repo.flush_live_daily(daily_df)
    logger.info("sync_daily_by_quotes: %d symbols flushed for %s", daily_df.height, today)
    return daily_df.height


# 除权事件明细列 (可选, 缺省为 null): 等差显示投影 (原始价 - Σ未来每股分红,
# 分红按其后送转比例缩放) 与官方金额全精度因子链重建使用。
# 存量 all.parquet 只有规范三列, 合并用 how="diagonal" 自动补 null;
# 读取方 (_apply_adj_factor / minute_adjust) 只按规范三列消费, 不受影响。
ADJ_DETAIL_COLS = ("dividend", "bonus", "allot", "allot_price", "prev_close")


def _normalize_adj_factor(raw) -> pl.DataFrame:
    """Normalize SDK ex_factors response to symbol/trade_date/ex_factor."""
    if raw is None or len(raw) == 0:
        return pl.DataFrame()
    if isinstance(raw, dict):
        rows: list[dict] = []
        for sym, values in raw.items():
            for item in values or []:
                row = dict(item or {})
                row.setdefault("symbol", sym)
                rows.append(row)
        df = pl.DataFrame(rows) if rows else pl.DataFrame()
    elif isinstance(raw, pl.DataFrame):
        df = raw
    else:
        df = pl.from_pandas(raw.reset_index() if hasattr(raw, "reset_index") else raw)
    if df.is_empty():
        return df
    # rename: timestamp/date → trade_date, adj_factor → ex_factor
    # 注意: 新版 SDK 可能同时返回 timestamp 和 trade_date (或 adj_factor 和 ex_factor),
    # 直接 rename 会产生重复列报错。仅当目标列不存在时才 rename。
    rename_map: dict[str, str] = {}
    for src, dst in (("timestamp", "trade_date"), ("date", "trade_date"), ("adj_factor", "ex_factor")):
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst
    df = df.rename(rename_map)
    if "trade_date" in df.columns:
        if df.schema["trade_date"] in {pl.Int64, pl.Int32, pl.UInt64, pl.UInt32, pl.Float64, pl.Float32}:
            # 毫秒时间戳 → 北京墙钟日期。不能直接 from_epoch().dt.date():
            # 那是 UTC 日期, 除权事件时间戳为北京零点 (= UTC 前一日 16:00),
            # 会整体早一天 (与 SDK True 路径 fromtimestamp(ts/1000, Asia/Shanghai) 不一致)。
            df = df.with_columns(
                _timestamp_to_beijing_datetime(pl.col("trade_date")).dt.date().alias("trade_date")
            )
        else:
            df = df.with_columns(pl.col("trade_date").cast(pl.Date, strict=False))
    if "ex_factor" in df.columns:
        df = df.with_columns(pl.col("ex_factor").cast(pl.Float64, strict=False))
    if not all(c in df.columns for c in ("symbol", "trade_date", "ex_factor")):
        return pl.DataFrame()
    # 明细列存在则透传 (缺省不影响规范三列); drop_nulls 只作用于规范三列 ——
    # 明细列合法可为 null (如纯送转事件的 dividend)。
    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False)
        for c in ADJ_DETAIL_COLS if c in df.columns
    ])
    cols = ["symbol", "trade_date", "ex_factor"] + [
        c for c in ADJ_DETAIL_COLS if c in df.columns
    ]
    return df.select(cols).drop_nulls(subset=["symbol", "trade_date", "ex_factor"])


def _align_adj_factor_frame(df: pl.DataFrame) -> pl.DataFrame:
    """除权因子帧对齐为统一 schema: 规范三列 + 明细列 (缺则补 null 列)。"""
    df = df.with_columns(
        pl.col("trade_date").cast(pl.Date, strict=False),
        pl.col("ex_factor").cast(pl.Float64, strict=False),
        *[pl.col(c).cast(pl.Float64, strict=False)
          for c in ADJ_DETAIL_COLS if c in df.columns],
    )
    df = df.with_columns([
        pl.lit(None, dtype=pl.Float64).alias(c)
        for c in ADJ_DETAIL_COLS if c not in df.columns
    ])
    return df.select(["symbol", "trade_date", "ex_factor", *ADJ_DETAIL_COLS])


def _merge_adj_factor_store(out: Path, new_data: pl.DataFrame) -> tuple[int, list[str]]:
    """合并写入 adj_factor 的 all.parquet, 返回 (新增行数, 因子发生变化的 symbol 列表)。

    how="diagonal" 兼容存量三列 schema (旧行明细自动为 null);
    unique keep="last" → 重复 (symbol, trade_date) 以本次同步为准,
    重跑一次同步即完成存量数据的明细回填。

    变化判定只看 ex_factor (新键 + 因子值差异 > 1e-9): 仅明细列回填
    (dividend/prev_close 等补 null) 不改变复权价格 → 不计入变化列表,
    enriched 局部重算因此不会被明细回填触发全市场重算。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    new_data = _align_adj_factor_frame(new_data)
    if not out.exists():
        _atomic_write_parquet(new_data.sort(["symbol", "trade_date"]), out)
        return new_data.height, sorted(new_data["symbol"].unique().to_list())
    existing = pl.read_parquet(out)
    before = existing.height
    # ex_factor 变化判定 (明细回填不算): 与存量按 (symbol, trade_date) 对齐,
    # 旧值缺失 (新事件) 或值差超容差即视为变化。
    diffed = new_data.select(["symbol", "trade_date", "ex_factor"]).join(
        existing.select(["symbol", "trade_date", "ex_factor"]),
        on=["symbol", "trade_date"], how="left", suffix="_old",
    ).filter(
        pl.col("ex_factor_old").is_null()
        | ((pl.col("ex_factor") - pl.col("ex_factor_old")).abs() > 1e-9)
    )
    changed_symbols = sorted(set(diffed["symbol"].to_list()))
    merged = pl.concat([existing, new_data], how="diagonal").unique(
        subset=["symbol", "trade_date"], keep="last",
    ).sort(["symbol", "trade_date"])
    _atomic_write_parquet(merged, out)
    return merged.height - before, changed_symbols


def _changed_factor_symbols(existing, incoming, *, cumulative=False) -> list[str]:
    """Compare effective factors, not the list of symbols returned by the API."""
    if incoming.is_empty():
        return []
    incoming = incoming.unique(["symbol", "trade_date"], keep="last").sort(["symbol", "trade_date"])
    if existing.is_empty():
        return sorted(incoming["symbol"].unique().to_list())
    old = existing.select("symbol", "trade_date", "ex_factor").unique(
        ["symbol", "trade_date"], keep="last").sort(["symbol", "trade_date"])
    if cumulative:
        reference = old.rename({"ex_factor": "_previous"})
        compared = incoming.join_asof(reference, on="trade_date", by="symbol", strategy="backward",
                                      check_sortedness=False)
        first = old.group_by("symbol").agg(pl.col("ex_factor").first().alias("_first"))
        compared = compared.join(first, on="symbol", how="left").with_columns(
            pl.coalesce("_previous", "_first").alias("_previous"))
    else:
        compared = incoming.join(old.rename({"ex_factor": "_previous"}),
                                 on=["symbol", "trade_date"], how="left").with_columns(
                                     pl.col("_previous").fill_null(1.0))
    changed = compared.filter(pl.col("_previous").is_null() | (
        (pl.col("ex_factor") - pl.col("_previous")).abs()
        > pl.col("_previous").abs().clip(lower_bound=1.0) * 1e-12))
    return sorted(changed["symbol"].unique().to_list())


def _persist_adj_factors(repo, new_data, *, asset_type, cumulative=False):
    new_data = _normalize_adj_factor(new_data)
    if not {"symbol", "trade_date", "ex_factor"} <= set(new_data.columns):
        raise ValueError("Incomplete adjustment factor schema")
    if new_data.filter(pl.col("ex_factor").is_null() | ~pl.col("ex_factor").is_finite()
                       | (pl.col("ex_factor") <= 0)).height:
        raise ValueError("Invalid adjustment factors")
    new_data = new_data.unique(["symbol", "trade_date"], keep="last").sort(["symbol", "trade_date"])
    factor_dir = "adj_factor_etf" if asset_type == "etf" else "adj_factor"
    out = repo.store.data_dir / factor_dir / "all.parquet"
    existing = pl.read_parquet(out) if out.exists() else new_data.clear()
    affected = _changed_factor_symbols(existing, new_data, cumulative=cumulative)
    merged = pl.concat([existing, new_data], how="diagonal_relaxed").unique(
        ["symbol", "trade_date"], keep="last").sort(["symbol", "trade_date"])
    if merged.equals(existing.sort(["symbol", "trade_date"])):
        return 0, []
    # Persist invalidation before publishing factors: a crash must not lose required recomputation.
    if affected and asset_type == "stock":
        from app.services.maintenance_queue import enqueue
        enqueue(repo.store.data_dir, "adjustment", affected)
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(merged, out)
    logger.info("adj_factor: %d new rows, %d genuinely changed symbols",
                merged.height - existing.height, len(affected))
    return merged.height - existing.height, affected


def sync_adj_factor(symbols: list[str], repo: KlineRepository,
                    capset: CapabilitySet,
                    start_time: datetime | None = None,
                    end_time: datetime | None = None,
                    on_chunk_done: Callable[[int, int], None] | None = None,
                    asset_type: str = "stock") -> tuple[int, list[str]]:
    """同步除权因子(Starter+)。SDK 接口:`tf.klines.ex_factors(symbols=...)`。

    支持增量: 传 start_time/end_time 只拉取该时间范围内的新除权事件。
    返回 (写入行数, 受影响的 symbol 列表) — 供 enriched 局部重算使用。
    受影响 = 因子新增或值变化 (ex_factor diff) 的标的; 仅事件明细列
    (dividend 等) 回填不改变复权价, 不计入受影响列表。
    """
    if not symbols:
        return 0, []

    provider_name = preferences.get_adj_factor_provider()
    if provider_name != "tickflow":
        from app.data_providers import custom as custom_sources
        if custom_sources.provider_has_dataset(provider_name, "adj_factor"):
            provider = custom_sources.get_provider(provider_name)
            new_data = provider.get_adj_factors(
                symbols,
                start_time=start_time,
                end_time=end_time,
                asset_type=asset_type,
                on_chunk_done=on_chunk_done,
            )
            if new_data.is_empty():
                # 扶摇等自定义源对 ETF 直接空返回, 与「该 ETF 无除权」无法区分;
                # 有 TickFlow 除权能力时回退, 否则 ETF 日K永远不复权。
                if asset_type == "etf":
                    logger.info(
                        "custom adj_factor provider %s returned no ETF rows, falling back to TickFlow",
                        provider_name,
                    )
                else:
                    return 0, []
            else:
                return _persist_adj_factors(
                    repo, new_data, asset_type=asset_type,
                    cumulative=getattr(provider, "adj_factor_kind", "event") == "cumulative",
                )
        # 自定义源未配置 adj_factor → 回退 TickFlow

    if not capset.has(Cap.ADJ_FACTOR):
        _log_cap_unavailable("adj_factor", "除权因子同步")
        return 0, []

    tf = get_client()
    limit = resolve_limit(
        capset,
        Cap.ADJ_FACTOR,
        default_batch=50,
        default_rpm=30,
        default_rpm_when_unset=False,
    )

    # 构建 SDK 参数 (False: _normalize_adj_factor 的 dict 分支原生支持)
    sdk_kwargs: dict = {"as_dataframe": False, "batch_size": limit.batch, "show_progress": False}
    if start_time:
        sdk_kwargs["start_time"] = _datetime_to_ms(start_time)
    if end_time:
        sdk_kwargs["end_time"] = _datetime_to_ms(end_time)

    chunks = chunked(symbols, limit.batch)
    all_dfs: list[pl.DataFrame] = []
    failed_syms: list[str] = []

    for i, chunk in enumerate(chunks):
        sleep_between_batches(i, limit.rpm)
        try:
            raw = _sdk_call_with_deadline(lambda chunk=chunk: tf.klines.ex_factors(chunk, **sdk_kwargs))
            normalized = _normalize_adj_factor(raw)
            if not normalized.is_empty():
                all_dfs.append(normalized)
            logger.debug("adj_factor chunk %d/%d: %d symbols", i + 1, len(chunks), len(chunk))
        except SDKCallTimeoutError as e:
            # on_chunk_done 在 try 外, 成败都会走到 → 心跳/取消检查点不缺
            logger.warning("adj_factor chunk %d/%d timed out: %s", i + 1, len(chunks), e)
            failed_syms.extend(chunk)
        except Exception as e:  # noqa: BLE001
            logger.warning("adj_factor chunk %d/%d failed: %s", i + 1, len(chunks), e)
            failed_syms.extend(chunk)

        if on_chunk_done:
            on_chunk_done(i + 1, len(chunks))

    # 部分失败可见化: 失败分块的标的不在 affected 里 → enriched 不会重算它们,
    # 它们会保持**旧的前复权价**直到下次成功同步。聚合一条 WARNING 让其可见。
    if failed_syms:
        logger.warning("adj_factor 同步部分失败: %d/%d 标的未获取复权因子, 将保持旧复权价 (样例: %s)",
                       len(failed_syms), len(symbols), failed_syms[:10])

    if not all_dfs:
        return 0, []

    new_data = pl.concat(all_dfs, how="diagonal_relaxed") if len(all_dfs) > 1 else all_dfs[0]

    return _persist_adj_factors(repo, new_data, asset_type=asset_type)


# ===== 分钟 K 同步 =====

def _minute_identity_lookup(minute_dir: Path, asset_type: str) -> dict[str, str]:
    """Load the dimension used to validate a minute write."""
    from app.services.security_identity import build_instrument_lookup

    data_dir = Path(minute_dir).parent
    if asset_type == "etf":
        path = data_dir / "instruments_etf" / "instruments_etf.parquet"
    elif asset_type == "index":
        path = data_dir / "instruments_index" / "instruments_index.parquet"
    else:
        path = data_dir / "instruments" / "instruments.parquet"
    paths = [path]
    if asset_type == "stock":
        # Delisted securities must remain outside the live instrument universe,
        # but their historical bars still need a verified code/exchange identity.
        paths.append(data_dir / "instruments_history" / "instruments_history.parquet")
    frames = []
    try:
        for candidate in paths:
            if candidate.exists():
                frames.append(pl.read_parquet(candidate))
        return build_instrument_lookup(frames)
    except Exception:  # noqa: BLE001
        return {}



def _ensure_minute_identity_columns(
    df: pl.DataFrame,
    *,
    asset_type: str,
    identity_lookup: dict[str, str] | None,
    allow_existing_identity: bool = False,
) -> pl.DataFrame:
    """Normalize minute rows to the storage identity schema before merging.

    Online providers commonly return only the eight OHLCV columns while local
    partitions also contain the four identity columns.  Keeping this
    normalization at the write boundary prevents an 8-column delta from
    colliding with an existing 12-column partition.
    """
    if df.is_empty():
        return df
    from app.services.security_identity import parse_identity

    identities = []
    for row in df.iter_rows(named=True):
        value = row.get("symbol")
        lookup = identity_lookup
        # A scoped repair lookup only contains the current batch. Existing
        # rows from the same partition already carry an independently
        # validated identity and must not be revalidated against that scope.
        if allow_existing_identity and row.get("code") and row.get("exchange"):
            stored_symbol = f"{row['code']}.{row['exchange']}"
            lookup = {str(row["code"]): stored_symbol}
            value = stored_symbol
        identities.append(
            parse_identity(
                value,
                asset_type=asset_type,
                lookup=lookup,
                # Existing partitions may predate the identity schema and
                # contain a qualified or structurally unambiguous symbol
                # without a current dimension row. Incoming deltas remain
                # fail-closed because this flag is only used for stored rows.
                require_lookup=not allow_existing_identity,
            )
        )
    identity_frame = pl.DataFrame(
        {
            "symbol": [item.symbol for item in identities],
            "asset_type": [item.asset_type for item in identities],
            "exchange": [item.exchange for item in identities],
            "code": [item.code for item in identities],
            "security_key": [item.security_key for item in identities],
        },
        schema={
            "symbol": pl.String,
            "asset_type": pl.String,
            "exchange": pl.String,
            "code": pl.String,
            "security_key": pl.String,
        },
    )
    # Replace identity fields in place so normalizing an existing partition
    # retains its column order and repeated normalization is idempotent.
    return df.with_columns(identity_frame.get_columns())



def _write_minute_partition(
    df: pl.DataFrame, minute_dir: Path, *, missing_only: bool = False,
    asset_type: AssetType | None = None, identity_lookup: dict[str, str] | None = None,
) -> int:
    """按日期原子合并。普通同步返回写入分区行数，定日修补与旧分区迁移只统计新增。"""
    if df.is_empty():
        return 0
    minute_dir = Path(minute_dir)
    if asset_type is None:
        asset_type = "etf" if minute_dir.name == "kline_etf_minute" else "stock"
    from app.services.security_identity import validate_frame_identity

    lookup = identity_lookup if identity_lookup is not None else _minute_identity_lookup(minute_dir, asset_type)
    validate_frame_identity(df, asset_type=asset_type, lookup=lookup, require_lookup=True)
    df = _ensure_minute_identity_columns(df, asset_type=asset_type, identity_lookup=lookup)
    keys = ["symbol", "datetime"]
    df = df.drop_nulls(keys)
    if df.is_empty():
        return 0
    df = df.with_columns(pl.col("datetime").dt.date().alias("_trade_date"))
    written = 0
    for day_df in df.partition_by("_trade_date"):
        out = minute_dir / f"date={day_df['_trade_date'][0]}" / "part.parquet"
        incoming = day_df.drop("_trade_date")
        if missing_only:
            incoming = incoming.unique(subset=keys, keep="last")
        if out.exists():
            existing = pl.read_parquet(out).filter(pl.col("datetime").is_not_null())
            existing = _ensure_minute_identity_columns(
                existing, asset_type=asset_type, identity_lookup=lookup, allow_existing_identity=True,
            )
            if missing_only:
                # 老分区可能用毫秒，新数据用微秒。按北京墙钟对齐后仅插入缺失键。
                incoming = incoming.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
                existing_keys = existing.select("symbol", pl.col("datetime").cast(pl.Datetime("us")))
                incoming = incoming.join(existing_keys, on=keys, how="anti")
                if incoming.is_empty():
                    continue
                merged = pl.concat([existing, incoming], how="diagonal_relaxed")
            else:
                # 普通同步允许更新已存在的分钟，只对触及的标的去重。
                touched = pl.col("symbol").is_in(incoming["symbol"].unique().to_list())
                updated = pl.concat([existing.filter(touched), incoming], how="diagonal_relaxed")
                merged = pl.concat([existing.filter(~touched), updated.unique(subset=keys, keep="last")],
                                   how="diagonal_relaxed")
        else:
            merged = incoming
        out.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_parquet(merged.sort(keys), out)
        written += incoming.height if missing_only else merged.height
    return written


def _selected_minute_provider_name() -> str:
    """识别偏好 getter 静默回退的失效自定义分钟源。"""
    saved = str(preferences.load().get("minute_data_provider", "tickflow") or "tickflow").lower()
    selected = preferences.get_minute_data_provider()
    if saved != "tickflow" and selected == "tickflow":
        return f"unavailable:{saved}"
    return selected


class MinuteSourceStamp(NamedTuple):
    data_dir: Path
    provider: str
    fingerprint: bytes


def minute_source_stamp() -> MinuteSourceStamp:
    """分钟请求只随当前分钟源、该源的配置和凭据失效。"""
    from app import secrets_store
    from app.config import settings

    name = _selected_minute_provider_name()
    if name == "tickflow":
        from app.tickflow.client import PAID_ENDPOINT

        endpoint = secrets_store.load().get("tickflow_base_url") or PAID_ENDPOINT
        config = (endpoint, secrets_store.get_tickflow_key())
        fingerprint = hashlib.sha256(json.dumps(config).encode()).digest()
    else:
        from app.data_providers import custom

        fingerprint = custom.minute_config_fingerprint(name)
    return MinuteSourceStamp(settings.data_dir, name, fingerprint)


def _partition_dates(root: Path) -> list[date]:
    """Return sorted date=YYYY-MM-DD partition dates without reading parquet data."""
    if not root.exists():
        return []
    out: list[date] = []
    for entry in root.iterdir():
        if not (entry.is_dir() and entry.name.startswith("date=")):
            continue
        try:
            out.append(date.fromisoformat(entry.name[5:]))
        except ValueError:
            continue
    return sorted(set(out))


def _parquet_row_count(path: Path) -> int | None:
    """Read parquet row count from metadata; fall back to None when unavailable."""
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001
        try:
            return int(pl.scan_parquet(str(path)).select(pl.len()).collect().item())
        except Exception:  # noqa: BLE001
            return None


def _symbol_counts_in_parquet(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    df = read_parquet_safe(path, columns=["symbol"])
    if df is None or df.is_empty() or "symbol" not in df.columns:
        return {}
    counts: dict[str, int] = {}
    for value in df["symbol"].to_list():
        symbol = _normalize_sync_symbol(value)
        if not symbol:
            continue
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _missing_minute_symbols(
    repo: KlineRepository,
    trade_day: date,
    symbols: list[str],
    *,
    asset_type: AssetType = "stock",
    min_symbol_rows: int = 1,
) -> list[str]:
    """Return tracked symbols that have daily rows but missing minute rows."""
    tracked_symbols = set(_dedupe_symbols(symbols))
    if not tracked_symbols:
        return []
    daily_root = repo.store.data_dir / ("kline_etf_daily" if asset_type == "etf" else "kline_daily")
    daily_part = daily_root / f"date={trade_day.isoformat()}" / "part.parquet"
    daily_symbols = set(_symbol_counts_in_parquet(daily_part))
    required_symbols = tracked_symbols & daily_symbols if daily_symbols else tracked_symbols
    if not required_symbols:
        return []
    minute_part = _minute_table_dir(repo, asset_type) / f"date={trade_day.isoformat()}" / "part.parquet"
    minute_counts = _symbol_counts_in_parquet(minute_part)
    min_symbol_rows = max(1, int(min_symbol_rows or 1))
    return sorted(symbol for symbol in required_symbols if minute_counts.get(symbol, 0) < min_symbol_rows)


def minute_gap_days(
    repo: KlineRepository,
    *,
    start_date: date,
    end_date: date,
    min_rows: int = 0,
    symbols: list[str] | None = None,
    min_symbol_rows: int = 1,
    newest_first: bool = False,
    limit: int | None = None,
    asset_type: AssetType = "stock",
) -> list[date]:
    """Find trading days with missing or obviously incomplete minute partitions.

    Daily K partitions define the local trading calendar. For small universes,
    pass ``symbols`` so a date partition is not considered complete merely
    because some other symbol already wrote rows for that day. ``min_rows`` is
    only a coarse full-market guard; keep it at 0 unless explicitly validating a
    full-market partition.
    """
    data_dir = repo.store.data_dir
    daily_root = data_dir / ("kline_etf_daily" if asset_type == "etf" else "kline_daily")
    minute_root = _minute_table_dir(repo, asset_type)
    expected = [d for d in _partition_dates(daily_root) if start_date <= d <= end_date]
    if not expected:
        return []
    tracked_symbols = set(_dedupe_symbols(symbols or []))
    min_symbol_rows = max(1, int(min_symbol_rows or 1))
    scan_days = list(reversed(expected)) if newest_first else expected
    limit_value = int(limit) if limit is not None else 0

    # For a bounded universe, aggregate all relevant partitions in one
    # DuckDB scan.  The previous implementation opened the daily and minute
    # parquet file once per date and then counted rows in Python; that made a
    # two-year Beijing repair spend several minutes before the first QMT
    # request.  Codes are normalized here because local minute rows may carry
    # an exchange suffix while daily rows usually do not.
    if tracked_symbols:
        try:
            root = data_dir.as_posix()
            code_sql = ",".join("'" + str(code).replace("'", "''") + "'" for code in sorted(tracked_symbols))
            date_filter = f"BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'"
            daily_sql = f"""
                SELECT CAST(regexp_extract(filename, 'date=([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})', 1) AS DATE) AS trade_day,
                       regexp_replace(CAST(symbol AS VARCHAR), '\\\\..*$', '') AS code
                FROM read_parquet('{root}/{'kline_etf_daily' if asset_type == 'etf' else 'kline_daily'}/date=*/part.parquet',
                                  union_by_name=true, filename=true)
                WHERE CAST(regexp_extract(filename, 'date=([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})', 1) AS DATE) {date_filter}
                  AND regexp_replace(CAST(symbol AS VARCHAR), '\\\\..*$', '') IN ({code_sql})
            """
            minute_sql = f"""
                SELECT CAST(regexp_extract(filename, 'date=([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})', 1) AS DATE) AS trade_day,
                       regexp_replace(CAST(symbol AS VARCHAR), '\\\\..*$', '') AS code,
                       COUNT(*) AS rows
                FROM read_parquet('{root}/{'kline_etf_minute' if asset_type == 'etf' else 'kline_minute'}/date=*/part.parquet',
                                  union_by_name=true, filename=true)
                WHERE CAST(regexp_extract(filename, 'date=([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})', 1) AS DATE) {date_filter}
                  AND regexp_replace(CAST(symbol AS VARCHAR), '\\\\..*$', '') IN ({code_sql})
                GROUP BY 1, 2
            """
            daily_rows = repo.db.execute(daily_sql).fetchall()
            minute_rows = repo.db.execute(minute_sql).fetchall()
            required_by_day: dict[date, set[str]] = {}
            for trade_day, code in daily_rows:
                if trade_day is not None and code:
                    required_by_day.setdefault(trade_day, set()).add(str(code))
            counts_by_day: dict[date, dict[str, int]] = {}
            for trade_day, code, count in minute_rows:
                if trade_day is not None and code:
                    counts_by_day.setdefault(trade_day, {})[str(code)] = int(count or 0)
            gaps = []
            for trade_day in scan_days:
                required = required_by_day.get(trade_day, set())
                counts = counts_by_day.get(trade_day, {})
                if any(counts.get(code, 0) < min_symbol_rows for code in required):
                    gaps.append(trade_day)
                    if limit_value > 0 and len(gaps) >= limit_value:
                        break
            return gaps
        except Exception as exc:  # noqa: BLE001
            logger.warning("bulk minute gap scan failed, falling back to partition scan: %s", exc)

    gaps: list[date] = []
    for trade_day in scan_days:
        part = minute_root / f"date={trade_day.isoformat()}" / "part.parquet"
        if not part.exists():
            gaps.append(trade_day)
            if limit_value > 0 and len(gaps) >= limit_value:
                break
            continue
        if tracked_symbols:
            if _missing_minute_symbols(
                repo,
                trade_day,
                list(tracked_symbols),
                asset_type=asset_type,
                min_symbol_rows=min_symbol_rows,
            ):
                gaps.append(trade_day)
                if limit_value > 0 and len(gaps) >= limit_value:
                    break
                continue
        if min_rows > 0:
            rows = _parquet_row_count(part)
            if rows is None or rows < min_rows:
                gaps.append(trade_day)
                if limit_value > 0 and len(gaps) >= limit_value:
                    break
    return gaps


def _group_trade_day_ranges(days: list[date], calendar: list[date]) -> list[tuple[date, date]]:
    if not days:
        return []
    selected = sorted(set(days))
    index = {d: i for i, d in enumerate(calendar)}
    ranges: list[tuple[date, date]] = []
    start = prev = selected[0]
    for day in selected[1:]:
        if day in index and prev in index and index[day] == index[prev] + 1:
            prev = day
            continue
        ranges.append((start, prev))
        start = prev = day
    ranges.append((start, prev))
    return ranges


def sync_missing_minute_days(
    symbols: list[str],
    repo: KlineRepository,
    capset: CapabilitySet,
    *,
    start_date: date,
    end_date: date,
    max_missing_days: int = 1,
    min_rows: int = 0,
    newest_first: bool = True,
    on_range_done: Callable[[int, int, str], None] | None = None,
    asset_type: AssetType = "stock",
    provider_name: str | None = None,
) -> dict:
    """Backfill missing minute-K trading days only.

    This is different from ``history_increment`` (latest tail) and
    ``history_backfill`` (extend before earliest local date): it repairs holes
    inside the existing history window.
    """
    if not symbols:
        return {
            "written_rows": 0,
            "missing_days_before": 0,
            "selected_days": [],
            "ranges": [],
            "remaining_missing_days": 0,
        }

    daily_root = repo.store.data_dir / ("kline_etf_daily" if asset_type == "etf" else "kline_daily")
    calendar = [d for d in _partition_dates(daily_root) if start_date <= d <= end_date]
    coverage_symbols = symbols if len(symbols) <= 1000 else None
    gaps = minute_gap_days(
        repo,
        start_date=start_date,
        end_date=end_date,
        min_rows=min_rows,
        symbols=coverage_symbols,
        newest_first=newest_first,
        limit=max(1, int(max_missing_days)),
        asset_type=asset_type,
    )
    ordered = sorted(gaps, reverse=newest_first)
    selected = sorted(ordered[: max(1, int(max_missing_days))])
    # A local provider such as QMT can return a bounded multi-day history per
    # symbol.  Merge consecutive missing dates even for a symbol-scoped
    # repair; the old one-day-per-request path multiplied the same QMT lookup
    # hundreds of times for the Beijing market.  The range still contains only
    # dates selected by the gap detector, so complete dates remain untouched.
    ranges = _group_trade_day_ranges(selected, calendar)
    minute_dir = _minute_table_dir(repo, asset_type)
    written_box = [0]
    requested_symbol_days = 0

    def _persist(seg_df: pl.DataFrame) -> None:
        with repo._write_lock:
            from app.config import settings
            if asset_type == "stock" and settings.minute_fallback_provider == "tdx" and preferences.get_minute_data_provider() == "tushare":
                pieces = []
                for key, part in seg_df.with_columns(pl.col("datetime").dt.date().alias("_day")).partition_by("_day", as_dict=True).items():
                    day = key[0]
                    part = part.drop("_day")
                    allow_qmt_partial = (
                        str(provider_name or "").strip().lower() == "qmt"
                        and os.getenv("QMT_IMPORT_PARTIAL", "0").strip().lower()
                        in {"1", "true", "yes"}
                    )
                    if (
                        datetime.combine(day, dt_time(15, 10)) <= cn_now().replace(tzinfo=None)
                        and not allow_qmt_partial
                    ):
                        dp = repo.store.data_dir / "kline_daily" / f"date={day}" / "part.parquet"
                        mp = minute_dir / f"date={day}" / "part.parquet"
                        if not dp.exists():
                            continue
                        part = _verified_closed_minute_additions(
                            part, pl.read_parquet(mp) if mp.exists() else part.clear(),
                            pl.read_parquet(dp), day, data_dir=repo.store.data_dir,
                            validation_source=(provider_name or preferences.get_minute_data_provider()),
                        )
                    if not part.is_empty():
                        pieces.append(part)
                if not pieces:
                    return
                seg_df = pl.concat(pieces)
            written_box[0] += _write_minute_partition(
                seg_df,
                minute_dir,
                asset_type=asset_type,
            )

    limit = resolve_limit(
        capset,
        Cap.KLINE_MINUTE_BATCH,
        default_batch=100,
        default_rpm=30,
        default_rpm_when_unset=False,
    )
    total_ranges = len(ranges)
    for idx, (first_day, last_day) in enumerate(ranges, start=1):
        range_symbols = symbols
        if coverage_symbols:
            # Query only symbols that are incomplete on at least one date in
            # this contiguous range.  This keeps the request bounded while
            # still allowing QMT to return several historical sessions at
            # once for each symbol.
            pending = set()
            for trade_day in calendar:
                if first_day <= trade_day <= last_day:
                    pending.update(_missing_minute_symbols(
                        repo,
                        trade_day,
                        coverage_symbols,
                        asset_type=asset_type,
                    ))
            range_symbols = sorted(pending)
            if not range_symbols:
                continue
            requested_symbol_days += len(range_symbols) * max(
                1, sum(first_day <= trade_day <= last_day for trade_day in calendar)
            )
        else:
            requested_symbol_days += len(symbols)
        start_time = datetime.combine(first_day, dt_time(9, 25))
        end_time = datetime.combine(last_day, dt_time(15, 5))
        label = f"{first_day.isoformat()}~{last_day.isoformat()}"

        def _range_chunk_done(cur: int, total: int, seg_label: str) -> None:
            if on_range_done is None:
                return
            local = cur / max(total, 1)
            done = int(((idx - 1) + local) / max(total_ranges, 1) * 100)
            on_range_done(done, 100, f"{label} · {seg_label}")

        sync_minute_batch(
            range_symbols,
            start_time=start_time,
            end_time=end_time,
            batch_size=limit.batch,
            rpm=limit.rpm,
            on_chunk_done=_range_chunk_done,
            segment_trading_days=preferences.get_minute_sync_segment_days(),
            on_segment=_persist,
            asset_type=asset_type,
            provider_name=provider_name,
        )

    remaining = minute_gap_days(
        repo,
        start_date=start_date,
        end_date=end_date,
        min_rows=min_rows,
        symbols=coverage_symbols,
        newest_first=newest_first,
        limit=max(10, int(max_missing_days)),
        asset_type=asset_type,
    )
    if written_box[0] > 0:
        try:
            d = repo.store.data_dir.as_posix()
            table = "kline_etf_minute" if asset_type == "etf" else "kline_minute"
            repo.db.execute(
                f"""CREATE OR REPLACE VIEW {table} AS
                    SELECT * FROM read_parquet('{d}/{table}/**/*.parquet', union_by_name=true)"""
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("refresh minute gap view failed: %s", e)

    return {
        "written_rows": written_box[0],
        "missing_days_before": len(gaps),
        "selected_days": [d.isoformat() for d in selected],
        "ranges": [(a.isoformat(), b.isoformat()) for a, b in ranges],
        "remaining_missing_days": len(remaining),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "min_rows": int(min_rows),
        "symbol_coverage": len(coverage_symbols or []),
        "requested_symbol_days": requested_symbol_days,
    }


def _resolve_minute_provider(
    provider_name: str,
) -> tuple[object | None, bool, str | None]:
    """统一解析 custom minute provider, 把所有 resolver 调用纳入同一异常边界。

    供分钟拉取和落盘共用，解析失败统一返回错误，不跨源补数据。

    返回 (provider, should_fallback_to_tickflow, error_msg):
      - provider_name == "tickflow" → (None, True, None)
      - 自定义源未配置 minute dataset 或解析失败 → (None, False, error_msg)
      - 自定义源可用 → (provider, False, None)

    只有用户明确选择 TickFlow 时才请求 TickFlow,避免静默跨源。
    注意: provider.get_minute() 仍由调用方在自身 try 块内调用 (业务异常, 非解析异常)。
    """
    if provider_name == "tickflow":
        return (None, True, None)
    from app.data_providers import custom as custom_sources
    try:
        provider = custom_sources.get_provider(provider_name)
        datasets = provider.config.datasets
    except Exception as e:  # noqa: BLE001
        return (None, False, str(e))
    if "minute" not in datasets:
        return (None, False, f"provider {provider_name} does not configure minute data")
    return (provider, False, None)


def _try_custom_minute(
    symbols: list[str],
    start_time: datetime | None,
    end_time: datetime | None,
    asset_type: AssetType,
    freq: str = "1m",
    on_chunk_done: Callable[[int, int, str], None] | None = None,
    provider_name: str | None = None,
    on_segment: Callable[[pl.DataFrame], None] | None = None,
) -> tuple[pl.DataFrame | None, bool]:
    """尝试从自定义分钟源拉取。返回 (df, should_fallback_to_tickflow)。

    返回契约:
      (None, True)   → 用户明确选择 TickFlow
      (df, False)    → 自定义源成功(含空 df)或失败 → 直接使用结果, 不回退

    用户明确选择了自定义分钟源后,其请求失败或解析异常均 fail-closed;
    避免自定义源增广的 batch 能力被误认为 TickFlow 原生权限,造成静默跨源请求。

    resolver 异常边界由 _resolve_minute_provider 统一兜底; 业务调用
    (provider.get_minute) 仍在本函数 try 块内, 与 resolver 异常分离
    便于日志区分 ("resolution failed" vs "call failed")。

    on_chunk_done 适配: 上层回调是 3 参 (cur, total, seg_label), provider
    实现内部以 2 参 (cur, total) 调用。这里包装一层, provider 调 2 参时补
    默认 seg_label="custom" 转发给上层, 保证进度展示不降级。
    """
    provider_name = provider_name or _selected_minute_provider_name()
    if provider_name.startswith("unavailable:"):
        logger.warning("custom minute provider %s is unavailable; not falling back to TickFlow",
                       provider_name.removeprefix("unavailable:"))
        return (pl.DataFrame(), False)
    provider, fallback, err = _resolve_minute_provider(provider_name)
    if fallback:
        if err is not None:
            logger.warning(
                "custom minute provider %s unavailable; not falling back to TickFlow: %s",
                provider_name, err,
            )
            return (pl.DataFrame(), False)
        return (None, True)
    if err is not None or provider is None:
        logger.warning("custom minute provider %s has no usable minute dataset: %s", provider_name, err)
        return (pl.DataFrame(), False)

    # 包装 on_chunk_done: provider 调 2 参 → 补 seg_label="custom" → 转发上层 3 参
    wrapped_cb: Callable[[int, int], None] | None = None
    callback_failed = False
    if on_chunk_done is not None:
        def _wrapped_cb(cur: int, total: int) -> None:
            nonlocal callback_failed
            try:
                on_chunk_done(cur, total, "custom")
            except BaseException:
                callback_failed = True
                raise
        wrapped_cb = _wrapped_cb

    if on_segment is not None:
        stream = iter_minute_frames(
            provider, symbols, start_time=start_time, end_time=end_time,
            asset_type=asset_type, freq=freq, on_chunk_done=wrapped_cb,
        )
        try:
            while True:
                try:
                    incoming = next(stream)
                except StopIteration:
                    break
                except ProviderRequestError:
                    raise
                except Exception as exc:  # iterator errors happen on next()
                    if callback_failed:
                        raise
                    raise ProviderRequestError(f"分钟源 {provider_name} 取数失败: {type(exc).__name__}") from exc
                try:
                    if not isinstance(incoming, pl.DataFrame):
                        raise ValueError("分钟源未返回标准数据帧")
                    df = _normalize_minute(incoming, source=provider_name)
                    if not df.is_empty():
                        required = set(CANONICAL_MINUTE_COLS) - {"amount"}
                        if required - set(df.columns):
                            raise ValueError("分钟数据缺少必要字段")
                        if df["symbol"].null_count() or df["datetime"].null_count():
                            raise ValueError("分钟数据标的或时间为空")
                except Exception as exc:  # malformed frames must not be published
                    raise ProviderRequestError(f"分钟源 {provider_name} 数据校验失败") from exc
                if not df.is_empty():
                    on_segment(df)
        finally:
            stream.close()
        return (pl.DataFrame(), False)

    try:
        from app.services import minute_failover
        if minute_failover.enabled(provider_name, provider):
            df, _ = minute_failover.fetch_minutes(
                provider, symbols, start_time, end_time, asset_type=asset_type,
                freq=freq, on_chunk_done=wrapped_cb,
            )
        else:
            df = provider.get_minute(
                symbols, start_time=start_time, end_time=end_time,
                asset_type=asset_type, freq=freq, on_chunk_done=wrapped_cb,
            )
    except ProviderRequestError:
        raise
    except Exception as e:
        if callback_failed:
            raise
        logger.warning(
            "custom minute provider %s call failed; not falling back to TickFlow: %s",
            provider_name, e,
        )
        return (pl.DataFrame(), False)
    try:
        # 时区契约守卫: 插件/自定义源帧同样收口为北京墙钟 (CONTRIBUTING §3.3)
        df = _normalize_minute(df, source=provider_name)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "custom minute provider %s datetime 契约校验失败; not falling back to TickFlow: %s",
            provider_name, e,
        )
        return (pl.DataFrame(), False)
    return (df, False)


def _try_custom_minute_batched(
    symbols: list[str],
    start_time: datetime | None,
    end_time: datetime | None,
    asset_type: AssetType,
    freq: str = "1m",
    batch_size: int | None = None,
    rpm: int | None = None,
    on_chunk_done: Callable[[int, int, str], None] | None = None,
    on_segment: Callable[[pl.DataFrame], None] | None = None,
    provider_name: str | None = None,
    on_source_issues: Callable[[list[dict]], None] | None = None,
) -> tuple[pl.DataFrame | None, bool]:
    """分批调用自定义分钟源。

    Tushare/Promax 的历史分钟接口通常是逐股请求。旧逻辑把全市场股票
    一次交给 provider，只有全部完成后才落盘；长任务中途失败会白跑。
    这里按 symbol chunk 分批调用并在每批结束后落盘，保证可取消、可续跑。
    """
    provider_name = provider_name or _selected_minute_provider_name()
    provider, fallback, err = _resolve_minute_provider(provider_name)
    if fallback:
        return (None, True)
    if provider is None or err:
        raise ProviderRequestError(f"分钟源 {provider_name} 不可用: {err}")
    if not symbols:
        return (pl.DataFrame(), False)

    chunks = chunked(symbols, batch_size)
    total_symbols = len(symbols)
    processed_symbols = 0
    hard_errors = 0
    successful_calls = 0
    out: list[pl.DataFrame] = []
    validation_issues: list[dict] = []
    received_rows = 0

    for chunk_index, chunk in enumerate(chunks):
        sleep_between_batches(chunk_index, rpm)

        def _wrapped_cb(cur: int, total: int) -> None:
            if on_chunk_done is None:
                return
            done = min(processed_symbols + max(0, int(cur)), total_symbols)
            on_chunk_done(done, total_symbols, "custom")

        try:
            from app.services import minute_failover
            if minute_failover.enabled(provider_name, provider):
                df, issues = minute_failover.fetch_minutes(
                    provider, chunk, start_time, end_time, asset_type=asset_type,
                    freq=freq, on_chunk_done=_wrapped_cb,
                )
                if issues and on_source_issues:
                    on_source_issues(issues)
                if df.is_empty() and issues:
                    failure = minute_failover.MinuteSourcesUnavailableError(issues)
                    if failure.reason == "upstream_unavailable":
                        raise failure
                    validation_issues.extend(issues)
                    logger.warning("%s", failure)
                    if on_chunk_done:
                        on_chunk_done(processed_symbols, total_symbols, str(failure))
            else:
                df = provider.get_minute(
                    chunk,
                    start_time=start_time,
                    end_time=end_time,
                    asset_type=asset_type,
                    freq=freq,
                    on_chunk_done=_wrapped_cb,
                )
            successful_calls += 1
        except Exception as e:  # noqa: BLE001
            if getattr(e, "reason", None) in {
                "source_pending", "rate_limited", "upstream_unavailable", "request_timeout", "transport_error",
            }:
                # Retry the unfinished increment later, not every remaining
                # stock batch (nor an unrelated provider) in this run.
                raise
            hard_errors += 1
            logger.warning(
                "custom minute provider %s call failed for %d symbols: %s",
                provider_name,
                len(chunk),
                e,
            )
            processed_symbols += len(chunk)
            if on_chunk_done:
                on_chunk_done(min(processed_symbols, total_symbols), total_symbols, "custom")
            continue

        try:
            df = _enforce_minute_beijing_wallclock(df, source=provider_name)
        except Exception as e:  # noqa: BLE001
            hard_errors += 1
            logger.warning(
                "custom minute provider %s datetime contract failed for %d symbols: %s",
                provider_name,
                len(chunk),
                e,
            )
            processed_symbols += len(chunk)
            if on_chunk_done:
                on_chunk_done(min(processed_symbols, total_symbols), total_symbols, "custom")
            continue

        df = df if df is not None else pl.DataFrame()
        if not df.is_empty():
            received_rows += df.height
            if on_segment:
                on_segment(df)
            else:
                out.append(df)

        processed_symbols += len(chunk)
        if on_chunk_done:
            on_chunk_done(min(processed_symbols, total_symbols), total_symbols, "custom")

    if not received_rows and validation_issues:
        raise minute_failover.MinuteSourcesUnavailableError(validation_issues)
    if hard_errors and successful_calls == 0:
        raise ProviderRequestError(f"分钟源 {provider_name} 全部请求失败")
    if not out:
        return (pl.DataFrame(), False)
    return (pl.concat(out, how="diagonal_relaxed"), False)


def sync_minute_batch(
    symbols: list[str],
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    count: int | None = None,
    batch_size: int | None = None,
    rpm: int | None = None,
    on_chunk_done: Callable[[int, int, str] | None] | None = None,
    segment_trading_days: int = 20,
    on_segment: Callable[[pl.DataFrame], None] | None = None,
    asset_type: AssetType = "stock",
    on_source_issues: Callable[[list[dict]], None] | None = None,
    provider_name: str | None = None,
    raw_basis: bool = False,
) -> pl.DataFrame:
    """向选中的分钟源交付完整标的列表和时间窗;消费源返回的标准帧。

    核心不拆网络请求。原生 iter_minute 由源管理标的分批、时间分段、限速、
    重试和有界输出;旧 get_minute 仍一次收到完整请求,返回一帧。
    batch_size / rpm / count / segment_trading_days / raw_basis 为旧 TickFlow
    调用兼容选项,仅交给 TickFlow Provider,不覆盖第三方源配置。
    on_chunk_done(cur,total,label) 转发源进度;回调异常原样透传以支持取消。
    on_segment 每收到一帧立即调用(并非固定时间段),用于有界流式落盘;
    不传时保持原返回契约,合并帧后返回 DataFrame。
    """
    if not symbols:
        return pl.DataFrame()
    provider_name = provider_name or _selected_minute_provider_name()
    from app.services import minute_failover
    provider, _, _ = _resolve_minute_provider(provider_name)
    if minute_failover.enabled(provider_name, provider):
        df, fallback = _try_custom_minute_batched(
            symbols, start_time=start_time, end_time=end_time,
            asset_type=asset_type, freq="1m", batch_size=batch_size, rpm=rpm,
            on_chunk_done=on_chunk_done, on_segment=on_segment,
            on_source_issues=on_source_issues, provider_name=provider_name,
        )
    else:
        df, fallback = _try_custom_minute(
            symbols, start_time=start_time, end_time=end_time,
            asset_type=asset_type, freq="1m", on_chunk_done=on_chunk_done, on_segment=on_segment,
            provider_name=provider_name,
        )
    if not fallback:
        if on_segment is not None:
            if df is not None and not df.is_empty():
                on_segment(df)
            return pl.DataFrame()
        return df if df is not None else pl.DataFrame()

    # 旧业务入口只转交选项;SDK 请求分批与时间分段由 TickFlow Provider 决定。
    from app.data_providers.tickflow_provider import TickFlowMinuteOptions, TickFlowProvider

    provider = TickFlowProvider(
        minute_options=TickFlowMinuteOptions(
            batch_size=batch_size, rpm=rpm, count=count,
            segment_trading_days=segment_trading_days, raw_basis=raw_basis,
        ),
        minute_client=get_client(), minute_call=_sdk_call_with_deadline,
        minute_progress=on_chunk_done,
    )
    stream = provider.iter_minute(symbols, start_time, end_time, asset_type=asset_type)
    try:
        if on_segment is not None:
            for frame in stream:
                on_segment(frame)
            return pl.DataFrame()
        parts = [frame for frame in stream if not frame.is_empty()]
        return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    finally:
        stream.close()


def intraday_monitor_support(capset: CapabilitySet | None) -> dict[str, object]:
    """返回分时信号监控可用的数据能力和单轮标的上限。"""
    provider_name = _selected_minute_provider_name()
    _, fallback, error = _resolve_minute_provider(provider_name)
    if error is not None:
        logger.warning("minute provider resolution failed while checking monitor support: %s", error)
        return {
            "available": False, "source": None, "max_symbols": 0,
            "reason": "当前分钟数据源不可用，请检查数据源配置",
        }
    if not fallback:
        return {
            "available": True, "source": "custom_minute", "max_symbols": 100,
            "reason": "使用已配置的分钟数据插件",
        }
    if capset is None:
        return {
            "available": False, "source": None, "max_symbols": 0,
            "reason": "需要分钟 K 或日内分时数据权限",
        }
    for cap, source in (
        (Cap.INTRADAY_BATCH, "intraday_batch"),
        (Cap.KLINE_MINUTE_BATCH, "minute_batch"),
    ):
        if capset.has(cap):
            limits = capset.limits(cap)
            return {
                "available": True, "source": source,
                "max_symbols": max(1, int(limits.batch or 100)) if limits else 100,
                "reason": "日内分时数据可用" if cap == Cap.INTRADAY_BATCH else "分钟 K 数据可用",
            }
    for cap, source in (
        (Cap.INTRADAY, "intraday_single"),
        (Cap.KLINE_MINUTE_BY_SYMBOL, "minute_single"),
    ):
        if capset.has(cap):
            return {
                "available": True, "source": source, "max_symbols": 1,
                "reason": "当前权限仅支持单标的分时监控",
            }
    return {
        "available": False, "source": None, "max_symbols": 0,
        "reason": "需要分钟 K 或日内分时数据权限",
    }


def fetch_intraday_monitor_batch(
    symbols: list[str], capset: CapabilitySet | None, *, now: datetime | None = None,
) -> pl.DataFrame:
    """按当前能力获取分时信号所需的当日分钟数据，不落盘。"""
    if not symbols:
        return pl.DataFrame()
    support = intraday_monitor_support(capset)
    if not support["available"] or len(symbols) > int(support["max_symbols"]):
        return pl.DataFrame()

    now = now or cn_now()
    start_time = now.replace(hour=9, minute=25, second=0, microsecond=0)
    source = support["source"]
    if source in {"custom_minute", "minute_batch"}:
        limits = capset.limits(Cap.KLINE_MINUTE_BATCH) if capset and capset.has(Cap.KLINE_MINUTE_BATCH) else None
        return sync_minute_batch(
            symbols, start_time=start_time, end_time=now,
            batch_size=limits.batch if limits else None,
            rpm=limits.rpm if limits else None,
        )

    tf = get_client()
    frames: list[pl.DataFrame] = []
    try:
        if source == "intraday_batch":
            limits = capset.limits(Cap.INTRADAY_BATCH) if capset else None
            raw = tf.klines.intraday_batch(
                symbols, count=300, as_dataframe=False, show_progress=False,
                batch_size=limits.batch if limits and limits.batch else 100,
            )
            df = _normalize_minute(_compact_klines_to_df(raw))
        elif source == "intraday_single":
            raw = tf.klines.intraday(symbols[0], count=300, as_dataframe=False)
            df = _normalize_minute(_compact_klines_to_df(raw, default_symbol=symbols[0]))
        elif source == "minute_single":
            raw = tf.klines.get(symbols[0], period="1m", count=300, as_dataframe=False)
            df = _normalize_minute(_compact_klines_to_df(raw, default_symbol=symbols[0]))
        else:
            df = pl.DataFrame()
        if not df.is_empty():
            frames.append(df)
    except Exception as e:  # noqa: BLE001
        logger.warning("intraday monitor fetch failed (%s, %d symbols): %s", source, len(symbols), e)
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def fetch_intraday_full_market_burst(
    symbols: list[str],
    capset: CapabilitySet | None,
    *,
    count: int = 300,
) -> tuple[pl.DataFrame, int]:
    """全市场当日分钟K并发脉冲拉取 (盘中增量刷新专用, 不落盘)。

    与 fetch_intraday_monitor_batch 的区别:
    - 监控路径每轮只拉少量标的 (≤ batch 上限, 单请求);
      本函数按 batch_size 把全市场切块后用线程池一次全部打出
      (5546/200 = 28 并发), 配合 >=60s 的固定轮节奏, 任何 60s
      滑动窗口至多一个脉冲 (28 < 48 安全 rpm)。
    - 单块失败不拖垮整轮: 成功块照常返回落盘; 失败块立即单独重试一次,
      仍失败则跳过该块 (本函数每轮拉全天, 下一轮天然自愈)。失败块过多
      (>4, 系统性故障/限流风暴) 时跳过重试, 避免向已过载的服务端加压。

    限流口径: 只用 intraday.batch 独立池 (Cap.INTRADAY_BATCH, Expert 专有),
    不与 kline.minute.batch (盘后分钟同步) 共享配额。
    返回 (当日全市场分钟K, 请求数)。
    """
    if not symbols:
        return (pl.DataFrame(), 0)
    limits = capset.limits(Cap.INTRADAY_BATCH) if capset and capset.has(Cap.INTRADAY_BATCH) else None
    batch_size = max(1, int(limits.batch) if limits and limits.batch else 200)
    chunks = list(chunked(symbols, batch_size))
    if not chunks:
        return (pl.DataFrame(), 0)

    from concurrent.futures import ThreadPoolExecutor

    tf = get_client()

    def _fetch(chunk: list[str]) -> tuple[list[pl.DataFrame], Exception | None]:
        # 单块独立容错: 异常作为返回值上交而不是抛出, 避免一个块把整轮
        # 已成功的数据一起拖垮 (pool.map 迭代中抛异常会废弃全部已收 frames)
        try:
            raw = tf.klines.intraday_batch(
                chunk, count=count, as_dataframe=False, show_progress=False,
                batch_size=len(chunk),
            )
            seg = _normalize_minute(_compact_klines_to_df(raw))
            return ([seg] if not seg.is_empty() else [], None)
        except Exception as e:
            return ([], e)

    frames: list[pl.DataFrame] = []
    failed: list[list[str]] = []
    with ThreadPoolExecutor(max_workers=min(len(chunks), 32)) as pool:
        for chunk, (sub, err) in zip(chunks, pool.map(_fetch, chunks), strict=True):
            if err is not None:
                failed.append(chunk)
            else:
                frames.extend(sub)

    requests = len(chunks)
    if failed:
        if len(failed) > 4:
            missed = sum(len(chunk) for chunk in failed)
            logger.warning(
                "intraday burst: %d/%d chunks failed (%d symbols), systemic — skip retry, next round re-pulls full day",
                len(failed), len(chunks), missed,
            )
        else:
            logger.warning("intraday burst: %d/%d chunks failed, retrying once", len(failed), len(chunks))
            for chunk in failed:
                sub, err = _fetch(chunk)
                requests += 1
                if err is None:
                    frames.extend(sub)
                else:
                    logger.warning(
                        "intraday burst: chunk retry still failed, skip %d symbols this round: %s",
                        len(chunk), err,
                    )
    if not frames:
        return (pl.DataFrame(), requests)
    return (pl.concat(frames, how="diagonal_relaxed"), requests)


def fetch_intraday_universe_increment(
    universe: str = "CN_Equity_A",
    *,
    count: int = 3,
) -> tuple[pl.DataFrame, int]:
    """全市场当日分钟K增量拉取 (盘中稳态轮专用, 不落盘)。

    /v1/klines/intraday/universe: 传 universe ID 一次请求返回全市场每只标的
    最新 count 根分钟K (服务端实测上限 3 根/标的), 替代稳态场景下 28 块并发
    的 intraday.batch 脉冲 (请求量 28→1, 传输量 ~40 倍降)。缺口回补
    (冷启动/长时间断档/全天修复) 仍走 fetch_intraday_full_market_burst。
    返回 (增量分钟K, 请求数); 拉取失败返回空 df 由调用方按失败轮处理。

    as_dataframe=False: raw 的 CompactKlineData 已按列组织 (字段→数组),
    直接用列数组建 polars 帧 — 无 pandas 中转、无逐行转换、无 _resolve_names
    名称解析, 全市场单轮本地转换 ~1.1s → ~0.1s。时区/dtype/列序契约
    复用 _normalize_minute (timestamp→北京墙钟 datetime, 输出 canonical 8 列)。
    """
    tf = get_client()
    try:
        raw = tf.klines.intraday_universe(universe, count=count, as_dataframe=False)
    except Exception as e:
        logger.warning("intraday universe fetch failed (%s): %s", universe, e)
        return (pl.DataFrame(), 0)
    seg = _compact_klines_to_df(raw)
    if seg.is_empty():
        return (pl.DataFrame(), 0)
    return (_normalize_minute(seg), 1)


def fetch_custom_intraday_increment(
    symbols: list[str],
    repo: KlineRepository,
    capset: CapabilitySet | None,
    *,
    full_day: bool = False,
    now: datetime | None = None,
) -> tuple[pl.DataFrame, int]:
    """Fetch today's minute increment from the configured custom minute source.

    TickFlow Expert has intraday.universe for one-shot full-market increments.
    Tushare/TDX/custom providers are symbol-scoped, so the realtime refresher
    uses the configured minute universe and each symbol's local latest bar.
    """
    universe = _dedupe_symbols(symbols)
    if not universe:
        return (pl.DataFrame(), 0)
    now_value = now or cn_now()
    if now_value.tzinfo is not None:
        end_time = now_value.replace(tzinfo=None)
    else:
        end_time = now_value
    day_start = end_time.replace(hour=9, minute=25, second=0, microsecond=0)
    if full_day:
        latest_by_symbol: dict[str, datetime] = {}
    else:
        latest_by_symbol = _latest_minute_datetime_by_symbol(
            repo,
            universe,
            start_floor=day_start,
            end_time=end_time,
        )
    groups = _minute_increment_groups(universe, latest_by_symbol, default_start=day_start)

    limits = capset.limits(Cap.KLINE_MINUTE_BATCH) if capset and capset.has(Cap.KLINE_MINUTE_BATCH) else None
    frames: list[pl.DataFrame] = []
    requests = 0
    for group_symbols, start_time, _label in groups:
        if start_time >= end_time:
            continue
        df = sync_minute_batch(
            group_symbols,
            start_time=start_time,
            end_time=end_time,
            batch_size=limits.batch if limits else None,
            rpm=limits.rpm if limits else None,
            asset_type="stock",
        )
        requests += max(1, len(group_symbols))
        if not df.is_empty():
            frames.append(df)
    if not frames:
        return (pl.DataFrame(), requests)
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .unique(subset=["symbol", "datetime"], keep="last")
        .sort(["symbol", "datetime"]),
        requests,
    )
def _resolve_full_minute_provider(
    provider_name: str,
) -> tuple[object | None, bool, str | None]:
    """解析全量分钟生效的自定义源。返回 (provider, should_use_tickflow, error_msg):
返回 (provider, should_use_tickflow, error_msg):

    - 明确选择 tickflow → (None, True, None)
    - 自定义源缺少 full_minute 或解析失败 → (None, False, error_msg)
    - 成功 → (provider, False, None)
    """
    if provider_name == "tickflow":
        return (None, True, None)
    from app.data_providers import custom as custom_sources
    try:
        provider = custom_sources.get_provider(provider_name)
        datasets = provider.config.datasets
    except Exception as e:  # noqa: BLE001
        return (None, False, str(e))
    if "full_minute" not in datasets:
        return (None, False, f"provider {provider_name} does not configure full_minute data")
    return (provider, False, None)


def fetch_intraday_custom_batch(
    provider: object,
    provider_name: str,
    symbols: list[str],
) -> tuple[pl.DataFrame, int]:
    """自定义源全量分钟修复轮: 当日窗口全市场批量拉取 (不落盘)。

    provider 契约 (见 docs/plugin-development.md):
    - get_intraday_batch(symbols, count, asset_type) 优先 — 源自管批量端点;
    - 未实现则回退 get_minute(symbols, 当日窗口) — 复用逐标的分钟K机制,
      请求数按 chunk 回调统计。
    帧统一过北京墙钟守卫 (与 _try_custom_minute 同纪律)。
    返回 (当日分钟K, 请求数); 失败返回空 df 由调用方按空轮处理。
    """
    try:
        method = getattr(provider, "get_intraday_batch", None)
        if callable(method):
            df = method(symbols)
            requests = 1
        else:
            counted = {"requests": 0}

            def _count_requests(cur: int, total: int) -> None:
                counted["requests"] = max(counted["requests"], int(total))

            end = cn_now()
            start = end.replace(hour=0, minute=0, second=0, microsecond=0)
            df = provider.get_minute(
                symbols, start_time=start, end_time=end,
                asset_type="stock", freq="1m", on_chunk_done=_count_requests,
            )
            requests = counted["requests"]
    except Exception as e:  # noqa: BLE001
        logger.warning("custom full_minute batch via %s failed: %s", provider_name, e)
        return (pl.DataFrame(), 0)
    try:
        df = _enforce_minute_beijing_wallclock(df, source=provider_name)
    except Exception as e:  # noqa: BLE001
        logger.warning("custom full_minute datetime 契约校验失败 (%s): %s", provider_name, e)
        return (pl.DataFrame(), 0)
    return (df, max(requests, 1))


def fetch_intraday_custom_latest(
    provider: object,
    provider_name: str,
    *,
    count: int = 3,
) -> tuple[pl.DataFrame, int] | None:
    """自定义源全量分钟稳态增量轮 (不落盘)。

    provider 可选实现 get_intraday_latest(symbols=None, count) → 每只标的最新
    count 根分钟K, 尽量单请求/低请求量 (TickFlow 的 intraday.universe 同义)。
    未实现返回 None — 调用方降级为仅修复轮模式。失败返回 (空 df, 0) 按空轮处理。
    """
    method = getattr(provider, "get_intraday_latest", None)
    if not callable(method):
        return None
    try:
        df = method(count=count)
    except Exception as e:  # noqa: BLE001
        logger.warning("custom full_minute latest via %s failed: %s", provider_name, e)
        return (pl.DataFrame(), 0)
    try:
        df = _enforce_minute_beijing_wallclock(df, source=provider_name)
    except Exception as e:  # noqa: BLE001
        logger.warning("custom full_minute datetime 契约校验失败 (%s): %s", provider_name, e)
        return (pl.DataFrame(), 0)
    return (df, 1)


def fetch_minute_single(
    symbol: str,
    trade_date: date,
    asset_type: AssetType = "stock",
    *,
    capset: CapabilitySet | None = None,
    raw_basis: bool = False,
) -> pl.DataFrame:
    source = minute_source_stamp()
    key = (source, capset, symbol, trade_date, asset_type, raw_basis)

    def fetch():
        frame = _fetch_minute_single(symbol, trade_date, asset_type, capset=capset, raw_basis=raw_basis)
        if source != minute_source_stamp():
            raise WorkBusyError("分钟数据源配置已变化，请重新加载")
        return frame

    return _single_minute_work.run(key, fetch, ttl_s=2)


def _fetch_minute_single(
    symbol: str, trade_date: date, asset_type: AssetType, *, capset: CapabilitySet, raw_basis: bool,
) -> pl.DataFrame:
    """实时拉取单股单日分钟 K(不写入本地)。

    优先使用当前自定义分钟源。只有明确选择 TickFlow 且具备原生权限时才请求 TickFlow；
    已选自定义源缺少能力或请求失败时 fail-closed，不混用来源。
    """
    from datetime import datetime
    # 北京时间窗口必须带时区: naive datetime 会被 .timestamp() 按服务器本地时区解释,
    # UTC 容器上窗口整体偏移 8 小时, 分时补拉必然为空。
    start_time = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 25, 0, tzinfo=CN_TZ)
    end_time = datetime(trade_date.year, trade_date.month, trade_date.day, 15, 5, 0, tzinfo=CN_TZ)

    minute_provider = preferences.get_minute_data_provider()
    realtime_provider = preferences.get_realtime_data_provider()
    from app.services import minute_failover
    candidate, candidate_fallback, _ = _resolve_minute_provider(minute_provider)
    failover_active = not candidate_fallback and minute_failover.enabled(minute_provider, candidate)

    def _try_named_minute(provider_name: str) -> tuple[pl.DataFrame, bool]:
        df, fallback = _try_custom_minute(
            [symbol], start_time=start_time, end_time=end_time,
            asset_type=asset_type, freq="1m", provider_name=provider_name,
        )
        return (df if df is not None else pl.DataFrame(), fallback)

    if not failover_active and trade_date == cn_today() and realtime_provider != minute_provider:
        rt_df, rt_fallback = _try_named_minute(realtime_provider)
        if not rt_fallback:
            if not rt_df.is_empty():
                logger.info(
                    "fetch_minute_single(%s, %s): realtime provider %s returned %d rows",
                    symbol, trade_date, realtime_provider, rt_df.height,
                )
                return rt_df

    # 自定义数据源分流: 与 sync_minute_batch 一致, 配了自定义分钟源时走 custom provider,
    # 避免无 TickFlow Pro+ 权限的用户分时图首次打开(本地无数据)时补拉失败返回空。
    df, fallback = _try_named_minute(minute_provider)
    if not fallback:
        # 见 sync_minute_batch 同分支注释: df 在此必非 None。
        if not df.is_empty():
            return df
        # Tushare rt_min_daily 会对部分标的返回旧交易日分钟数据；规范化过滤后为空。
        # 单股详情页此时应继续尝试本地 TDX，而不是把空数组直接交给前端。
        if trade_date != cn_today() or failover_active:
            return df

    if trade_date == cn_today() and asset_type == "stock":
        for fallback_provider in ("tdx",):
            if fallback_provider in {str(minute_provider).lower(), str(realtime_provider).lower()}:
                continue
            alt_df, alt_fallback = _try_named_minute(fallback_provider)
            if not alt_fallback and not alt_df.is_empty():
                logger.info(
                    "fetch_minute_single(%s, %s): provider %s returned empty, fallback %s returned %d rows",
                    symbol, trade_date, minute_provider, fallback_provider, alt_df.height,
                )
                return alt_df

    if not fallback or capset is None or not capset.has(Cap.KLINE_MINUTE_BY_SYMBOL):
        return pl.DataFrame()

    tf = get_client()
    try:
        raw = tf.klines.batch(
            [symbol], period="1m",
            start_time=_datetime_to_ms(start_time),
            end_time=_datetime_to_ms(end_time),
            count=10000,
            adjust="none" if raw_basis else "forward",
            as_dataframe=False, show_progress=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("fetch_minute_single(%s, %s) failed: %s", symbol, trade_date, e)
        return pl.DataFrame()

    return _normalize_minute(_compact_klines_to_df(raw, default_symbol=symbol))


def fetch_adj_factor_single(
    symbol: str, *, asset_type: AssetType = "stock", capset: CapabilitySet | None = None, strict: bool = False,
) -> pl.DataFrame:
    """按已配置的数据源获取单标的除权因子，不写入本地。

    返回结构: symbol, trade_date, ex_factor (空 DataFrame 表示无除权事件或拉取失败)。
    与 _apply_adj_factor / compute_enriched 的 factors 参数格式一致。
    """
    provider_name = preferences.get_adj_factor_provider()
    if provider_name != "tickflow":
        from app.data_providers import custom

        if custom.provider_has_dataset(provider_name, "adj_factor"):
            frame = custom.get_provider(provider_name).get_adj_factors(
                [symbol], start_time=None, end_time=None, asset_type=asset_type,
            )
            if not frame.is_empty() or asset_type != "etf":
                return frame
    if capset is not None and not capset.has(Cap.ADJ_FACTOR):
        return pl.DataFrame()
    tf = get_client()
    try:
        raw = tf.klines.ex_factors([symbol], as_dataframe=False, show_progress=False)
    except Exception as e:
        logger.warning("fetch_adj_factor_single(%s) failed: %s", symbol, e)
        if strict:
            raise ProviderRequestError("复权因子获取失败") from e
        return pl.DataFrame()
    return _normalize_adj_factor(raw)


def _minute_table_dir(repo: KlineRepository, asset_type: AssetType = "stock") -> Path:
    dirname = "kline_minute" if asset_type == "stock" else f"kline_{asset_type}_minute"
    return repo.store.data_dir / dirname


def _minute_symbol_key_expr(column: str = "symbol") -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.replace(r"^(SH|SZ|BJ)", "")
        .str.replace(r"\..*$", "")
        .alias("_symbol_key")
    )


def _minute_partition_files(
    repo: KlineRepository,
    *,
    asset_type: AssetType = "stock",
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[Path]:
    minute_dir = _minute_table_dir(repo, asset_type)
    if not minute_dir.exists():
        return []
    files: list[tuple[date, Path]] = []
    for entry in minute_dir.iterdir():
        if not (entry.is_dir() and entry.name.startswith("date=")):
            continue
        try:
            trade_date = date.fromisoformat(entry.name[5:])
        except ValueError:
            continue
        if start_date is not None and trade_date < start_date:
            continue
        if end_date is not None and trade_date > end_date:
            continue
        part = entry / "part.parquet"
        if part.exists():
            files.append((trade_date, part))
    files.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in files]


def _latest_minute_datetime_by_symbol(
    repo: KlineRepository,
    symbols: list[str],
    *,
    asset_type: AssetType = "stock",
    start_floor: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[str, datetime]:
    """Return local latest minute timestamp per symbol.

    The historical sync must not use one global max(datetime): a single already
    fresh symbol would make every lagging symbol start too late. We scan recent
    date partitions newest-first and stop as soon as all requested symbols are
    found. Invalid half-written parquet files are quarantined by read_parquet_safe.
    """
    keys = _dedupe_symbols(symbols)
    if not keys:
        return {}
    lookup = _minute_identity_lookup(repo.store.data_dir / "kline_minute", asset_type)
    aliases = {}
    requested = {}
    for key in keys:
        code = key.split(".", 1)[0]
        native = lookup.get(code)
        if native and key not in {code, native}:
            continue
        canonical = native or key
        requested.setdefault(canonical, set()).add(key)
        aliases[key] = canonical
        if native:
            aliases[code] = canonical
            aliases[native] = canonical
    wanted = set(requested)
    latest: dict[str, datetime] = {}
    found = set()
    files = _minute_partition_files(
        repo,
        asset_type=asset_type,
        start_date=start_floor.date() if start_floor else None,
        end_date=end_time.date() if end_time else None,
    )
    if not files:
        return {}

    for part in files:
        if wanted <= found:
            break
        df = read_parquet_safe(part, columns=["symbol", "datetime"])
        if df is None or df.is_empty() or "symbol" not in df.columns or "datetime" not in df.columns:
            continue
        try:
            sub = (
                df.with_columns(
                    pl.col("symbol").cast(pl.String).str.to_uppercase()
                    .str.replace(r"^(SH|SZ|BJ)(\d{6})$", "$2.$1")
                    .replace_strict(aliases, default=None).alias("_symbol_key")
                )
                .filter(pl.col("_symbol_key").is_in(list(wanted - found)))
                .filter(pl.col("datetime").is_not_null())
            )
            if start_floor is not None:
                sub = sub.filter(pl.col("datetime") >= start_floor)
            if end_time is not None:
                sub = sub.filter(pl.col("datetime") <= end_time)
            if sub.is_empty():
                continue
            grouped = sub.group_by("_symbol_key").agg(pl.col("datetime").max().alias("datetime"))
            for row in grouped.to_dicts():
                value = row.get("datetime")
                key = _normalize_sync_symbol(row.get("_symbol_key"))
                if key and isinstance(value, datetime):
                    found.add(key)
                    for requested_key in requested[key]:
                        latest[requested_key] = value
        except Exception as exc:  # noqa: BLE001
            logger.debug("latest minute scan skipped %s: %s", part, exc)
            continue
    return latest


def _minute_increment_groups(
    symbols: list[str],
    latest_by_symbol: dict[str, datetime],
    *,
    default_start: datetime,
) -> list[tuple[list[str], datetime, str]]:
    buckets: dict[datetime, list[str]] = {}
    for symbol in _dedupe_symbols(symbols):
        start = latest_by_symbol.get(symbol) or default_start
        buckets.setdefault(start.replace(second=0, microsecond=0), []).append(symbol)
    groups: list[tuple[list[str], datetime, str]] = []
    for start, group_symbols in sorted(buckets.items(), key=lambda item: item[0]):
        if start == default_start.replace(second=0, microsecond=0):
            label = f"缺口回补 {start:%m-%d %H:%M}"
        else:
            label = f"增量 {start:%m-%d %H:%M}"
        groups.append((group_symbols, start, label))
    return groups


def _closed_minute_gap_starts(repo, symbols, start: datetime, end: datetime) -> dict[str, datetime]:
    """A 15:00 watermark alone does not prove that all 240 minutes exist."""
    from app.services.minute_repair import daily_reference, keyed, regular
    from app.services.security_identity import IdentityError, parse_identity
    lookup = _minute_identity_lookup(repo.store.data_dir / "kline_minute", "stock")
    requested = {}
    for symbol in _dedupe_symbols(symbols):
        try:
            identity = parse_identity(symbol, asset_type="stock", lookup=lookup)
        except IdentityError:
            continue
        requested.setdefault(identity.code, []).append(symbol)
    missing = {}
    for day in _partition_dates(repo.store.data_dir / "kline_daily"):
        if not start.date() <= day <= end.date() or datetime.combine(day, dt_time(15)) > end:
            continue
        daily_path = repo.store.data_dir / "kline_daily" / f"date={day}" / "part.parquet"
        references = daily_reference(pl.read_parquet(daily_path), data_dir=repo.store.data_dir)
        active = set(requested) & set(references)
        minute_path = repo.store.data_dir / "kline_minute" / f"date={day}" / "part.parquet"
        complete = set()
        if minute_path.exists():
            frame = keyed(pl.read_parquet(minute_path, columns=["symbol", "datetime"]))
            counts = regular(frame, day).group_by("_code").agg(pl.col("datetime").n_unique().alias("n"))
            complete = set(counts.filter(pl.col("n") == 240)["_code"])
        for code in active - complete:
            for symbol in requested[code]:
                missing[symbol] = min(missing.get(symbol, end), datetime.combine(day, dt_time.min))
    return missing


def _verified_closed_minute_additions(incoming, existing, daily, day, *, data_dir=None, validation_source=None):
    from collections import Counter
    from app.services.minute_repair import checked_additions, daily_reference, keyed

    references = daily_reference(daily, data_dir=data_dir)
    old = {key[0]: part.drop("_code") for key, part in keyed(existing).partition_by("_code", as_dict=True).items()}
    additions, rejected = [], Counter()
    for key, part in keyed(incoming).partition_by("_code", as_dict=True).items():
        code = key[0]
        source = part.drop("_code")
        delta, reason = checked_additions(
            source, old.get(code, source.clear()), references.get(code), day,
            source=validation_source,
        )
        if not delta.is_empty():
            additions.append(delta)
        elif reason != "already_complete":
            rejected[reason] += 1
    if rejected:
        logger.warning("closed minute gap %s left unchanged after verification: %s", day, dict(rejected))
    return pl.concat(additions) if additions else incoming.clear()
def _as_beijing(d: datetime) -> datetime:
    """落盘的分钟 datetime 是北京墙钟 naive, 带上北京时区再交给取数窗口。

    naive 值经 _datetime_to_ms 会被 .timestamp() 按服务器本地时区解释, 与同
    窗口另一端的服务器本地时间混用后整体错位 (UTC 容器上错 8 小时)。
    """
    return d if d.tzinfo is not None else d.replace(tzinfo=CN_TZ)


def _latest_minute_datetime(repo: KlineRepository, asset_type: AssetType = "stock") -> datetime | None:
    """本地分钟 K 数据的最新时间 (北京时区)。"""
    try:
        table = "kline_etf_minute" if asset_type == "etf" else "kline_minute"
        res = repo.execute_one(f"SELECT max(datetime) FROM {table}")
        if res and res[0]:
            d = res[0]
            if isinstance(d, datetime):
                return _as_beijing(d)
            return _as_beijing(datetime.fromisoformat(str(d)))
    except Exception:  # noqa: BLE001
        pass
    return None


def _earliest_minute_datetime(repo: KlineRepository, asset_type: AssetType = "stock") -> datetime | None:
    """本地分钟 K 数据的最早时间 (北京时区, 用于向前扩展的起点)。"""
    try:
        table = "kline_etf_minute" if asset_type == "etf" else "kline_minute"
        res = repo.execute_one(f"SELECT min(datetime) FROM {table}")
        if res and res[0]:
            d = res[0]
            if isinstance(d, datetime):
                return _as_beijing(d)
            return _as_beijing(datetime.fromisoformat(str(d)))
    except Exception:  # noqa: BLE001
        pass
    return None


def _cleanup_null_datetime_minute(repo: KlineRepository) -> None:
    """检测并清除 datetime 全为 null 的旧版分钟 K 数据(迁移用)。"""
    minute_dir = repo.store.data_dir / "kline_minute"
    if not minute_dir.exists():
        return
    try:
        row = repo.execute_one(
            "SELECT count(*) AS total, count(datetime) AS non_null FROM kline_minute"
        )
        if row and row[0] > 0 and (row[1] is None or row[1] == 0):
            # 视图可能滞后于刚写入的分区，在写锁内按实际文件确认再清理。
            with repo._write_lock:
                for part in minute_dir.rglob("*.parquet"):
                    times = pl.read_parquet(part, columns=["datetime"])["datetime"]
                    if len(times) and times.null_count() == len(times):
                        part.unlink()
    except Exception as e:  # noqa: BLE001
        logger.debug("minute cleanup check failed: %s", e)


def _migrate_symbol_to_date_partition(repo: KlineRepository) -> None:
    """将旧版 symbol= 分区迁移为 date= 分区。迁移完成后删除旧目录。"""
    minute_dir = repo.store.data_dir / "kline_minute"
    if next(minute_dir.glob("symbol=*/*.parquet"), None) is None:
        return
    with repo._write_lock:
        sources = sorted(minute_dir.glob("symbol=*/*.parquet"))
        frames = []
        for part in sources:
            df = pl.read_parquet(part)
            if "datetime" not in df.columns:
                raise ValueError(f"旧分钟分区缺少 datetime: {part.name}")
            symbol = part.parent.name.removeprefix("symbol=")
            df = df.with_columns(
                (pl.col("symbol").fill_null(symbol) if "symbol" in df.columns else pl.lit(symbol)).alias("symbol")
            )
            frames.append(df)
        if not frames:
            return
        # 全部读入且写入成功才删除源文件。中途失败可以安全重试。
        added = _write_minute_partition(pl.concat(frames, how="diagonal_relaxed"), minute_dir, missing_only=True)
        for part in sources:
            part.unlink()
        for folder in {part.parent for part in sources}:
            if not any(folder.iterdir()):
                folder.rmdir()
        logger.info("minute-K migration done: %d rows added", added)


def sync_and_persist_minute(
    symbols: list[str],
    repo: KlineRepository,
    capset: CapabilitySet,
    days: int = 5,
    on_chunk_done: Callable[[int, int, str], None] | None = None,
    extend_backward: bool = False,
    force_full_days: bool = False,
    as_of: datetime | None = None,
    on_source_issues: Callable[[list[dict]], None] | None = None,
    *,
    asset_type: AssetType | None = None,
    repair_date: date | None = None,
    expected_source: MinuteSourceStamp | None = None,
    expected_raw_basis: bool | None = None,
) -> int:
    """同步分钟 K 并存到 Parquet。返回写入行数。

    存储口径由基准标记决定 (services/minute_adjust): 存量未迁移 → SDK adjust=qfq
    前复权 (旧行为); 已迁移 → adjust='none' 原始价落盘, 复权读取时投影。
    使用 start_time / end_time 区间拉取, 确保所有标的覆盖同一时间段。
    on_chunk_done(current, total) 每个 chunk 完成后回调。
    force_full_days=True 时强制回溯 days 自然日 (不增量补, 用于个股补齐历史)。
    """
    source = minute_source_stamp()
    if expected_source is not None and source != expected_source:
        raise WorkBusyError("分钟数据源配置已变化，请重新补齐")
    raw_basis = minute_adjust.minute_basis_is_raw(repo.store.data_dir)
    if expected_raw_basis is not None and raw_basis != expected_raw_basis:
        raise WorkBusyError("分钟价格基准已变化，请重新补齐")
    if asset_type is None:
        etfs = getattr(repo, "get_etf_symbol_set", lambda: set())()
        if etfs and any(symbol in etfs for symbol in symbols):
            return sum(
                sync_and_persist_minute(
                    group, repo, capset, days, on_chunk_done, extend_backward, force_full_days,
                    as_of, on_source_issues, asset_type=asset, repair_date=repair_date, expected_source=source,
                    expected_raw_basis=raw_basis,
                )
                for asset, group in (
                    ("stock", [s for s in symbols if s not in etfs]),
                    ("etf", [s for s in symbols if s in etfs]),
                )
                if group
            )
        asset_type = "stock"
    if on_chunk_done:
        on_chunk_done(0, 100, "检查分钟数据源")
    minute_provider = _selected_minute_provider_name()
    if minute_provider.startswith('unavailable:'):
        raise ProviderRequestError('配置的分钟数据源不可用')
    # resolver 调用统一走 _resolve_minute_provider, 与 _try_custom_minute 共用异常边界。
    # resolver 异常时视为非 custom (minute_is_custom=False), 走 capset 检查 →
    # sync_minute_batch 内 _try_custom_minute 会再次 resolver 异常 → fallback TickFlow。
    provider, fallback, resolve_err = _resolve_minute_provider(minute_provider)
    minute_is_custom = not fallback
    if resolve_err is not None:
        logger.warning("custom minute provider %s unavailable: %s", minute_provider, resolve_err)
        return 0
    if not symbols:
        return 0
    if not minute_is_custom and not capset.has(Cap.KLINE_MINUTE_BATCH):
        return 0
    from app.services import minute_failover
    wall_now = cn_now().replace(tzinfo=None)
    if as_of is not None:
        as_of = _as_beijing(as_of).astimezone(CN_TZ).replace(tzinfo=None)
    now = min(as_of, wall_now) if as_of else wall_now
    if minute_failover.enabled(minute_provider, provider):
        closed_request = (now.time() >= dt_time(15)
                          and datetime.combine(now.date(), dt_time(15, 10)) <= wall_now)
        deferred = minute_failover.retry_state(repo.store.data_dir, include_local=closed_request)
        if deferred["source_retry"]:
            raise minute_failover.MinuteSourcesUnavailableError(deferred["source_errors"])

    # Legacy full-history audits/migrations are offline maintenance only.
    # Daily batches must neither scan nor delete unrelated historical partitions.

    minute_dir = _minute_table_dir(repo, asset_type)
    written_box = [0]  # list 闭包, 绕过 Python 闭包外层赋值
    repair_dates: set[date] = set()
    repair_backup_dir = None

    def _persist(seg_df: pl.DataFrame) -> None:
        nonlocal repair_backup_dir
        # 单股自动补齐可能与另一个补齐请求同时写同一日期分区。Windows 不允许
        # 替换仍被另一写入占用的临时文件,因此读-改-写必须复用仓库写锁。
        with repo._write_lock:
            if minute_source_stamp() != source:
                raise WorkBusyError("分钟数据源配置已变化，请重新补齐")
            if minute_adjust.minute_basis_is_raw(repo.store.data_dir) != raw_basis:
                raise WorkBusyError("分钟价格基准已变化，请重新补齐")
            if repair_date is not None:
                seg_df = seg_df.filter(
                    pl.col("symbol").is_in(symbols)
                    & pl.col("datetime").is_between(start_time, end_time)
                )
            if seg_df.is_empty():
                return
            from app.config import settings
            if asset_type == "stock" and settings.minute_fallback_provider == "tdx" and minute_provider == "tushare":
                repair_dates.update(
                    day for day in seg_df["datetime"].dt.date().drop_nulls().unique()
                    if datetime.combine(day, dt_time(15, 10)) <= cn_now().replace(tzinfo=None)
                )
            if repair_dates:
                pieces = []
                dated = seg_df.with_columns(pl.col("datetime").dt.date().alias("_sync_day"))
                for key, part in dated.partition_by("_sync_day", as_dict=True).items():
                    day = key[0]
                    part = part.drop("_sync_day")
                    if day in repair_dates:
                        daily_path = repo.store.data_dir / "kline_daily" / f"date={day}" / "part.parquet"
                        existing_path = minute_dir / f"date={day}" / "part.parquet"
                        if not daily_path.exists():
                            logger.warning("closed minute gap %s has no daily reference; not written", day)
                            continue
                        part = _verified_closed_minute_additions(
                            part, pl.read_parquet(existing_path) if existing_path.exists() else part.clear(),
                            pl.read_parquet(daily_path), day, data_dir=repo.store.data_dir,
                        )
                    if not part.is_empty():
                        pieces.append(part)
                if not pieces:
                    return
                seg_df = pl.concat(pieces)
            for day in set(seg_df["datetime"].dt.date().drop_nulls()) & repair_dates:
                import shutil
                import uuid
                existing = minute_dir / f"date={day}" / "part.parquet"
                if existing.exists():
                    if repair_backup_dir is None:
                        repair_backup_dir = repo.store.data_dir / "import_backups" / ("minute_incremental_" + uuid.uuid4().hex[:12])
                    backup = repair_backup_dir / f"date={day}" / "part.parquet"
                    if not backup.exists():
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(existing, backup)
            written_box[0] += _write_minute_partition(
                seg_df,
                minute_dir,
                asset_type=asset_type, missing_only=repair_date is not None,
            )
    if repair_date is not None:
        start_time = datetime.combine(repair_date, dt_time(9, 25))
        end_time = min(datetime.combine(repair_date, dt_time(15, 5)), now)
        groups = [(symbols, start_time, repair_date.isoformat())]
    elif extend_backward:
        # 向前扩展模式: 从本地最早数据往前补, 叠加已有数据避免缺口。
        earliest_dt = _earliest_minute_datetime(repo, asset_type)
        # 按交易日换算自然日 (7/5 系数)。>41 交易日时 +10 天余量覆盖节假日。
        # (分段由 sync_minute_batch 的 segment_trading_days 控制, 与此处的区间天数独立。)
        calendar_days = int(days * 7 / 5) + (10 if days > 41 else 0)
        if earliest_dt:
            end_time = _as_beijing(earliest_dt).astimezone(CN_TZ).replace(tzinfo=None)
            start_time = end_time - timedelta(days=calendar_days)
        else:
            # 本地无数据 → 从今天往前拉
            start_time = now - timedelta(days=calendar_days)
            end_time = now
        start_time = datetime.combine(start_time.date(), dt_time(9, 25))
        groups = [(symbols, start_time, f"向前扩展 {start_time:%m-%d}~{end_time:%m-%d}")]
    else:
        # 默认增量模式: 首次拉取回溯 N 天, 已有数据则按「每只股票自己的最新时间」
        # 分组增量补到今天。不能用全局 max(datetime), 否则一只新鲜票会让所有
        # 滞后票跳过历史缺口。
        # force_full_days=True: 强制回溯 days 自然日 (个股补齐历史, 不增量)
        if force_full_days:
            # 按交易日换算自然日 (7/5 系数), 确保覆盖足够交易日
            calendar_days = int(days * 7 / 5) + 5
            start_time = datetime.combine((now - timedelta(days=calendar_days)).date(), dt_time(9, 25))
            end_time = now
            groups = [(symbols, start_time, f"强制补齐 {start_time:%m-%d}~{end_time:%m-%d}")]
        else:
            start_time = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = min(now, now.replace(hour=15, minute=0, second=0, microsecond=0))
            if dt_time(11, 30) <= now.time() < dt_time(13):
                end_time = now.replace(hour=11, minute=30, second=0, microsecond=0)
            if now.time() < dt_time(9, 30) or now.weekday() >= 5:
                prior = [d for d in _partition_dates(repo.store.data_dir / "kline_daily") if d < now.date()]
                if prior:
                    end_time = datetime.combine(max(prior), dt_time(15))
                    start_time = min(start_time, datetime.combine(end_time.date(), dt_time.min))
            if on_chunk_done:
                on_chunk_done(0, 100, f"检查最近分区 {start_time:%Y-%m-%d}~{end_time:%Y-%m-%d}")
            latest_by_symbol = _latest_minute_datetime_by_symbol(
                repo,
                symbols,
                start_floor=start_time,
                end_time=end_time,
                asset_type=asset_type,
            )
            if on_chunk_done:
                on_chunk_done(0, 100, f"校验本批 {len(symbols)} 只股票的分钟缺口")
            gap_starts = _closed_minute_gap_starts(repo, symbols, start_time, end_time) if asset_type == "stock" else {}
            if gap_starts:
                first_gap = min(v.date() for v in gap_starts.values())
                repair_dates.update(
                    first_gap + timedelta(days=i)
                    for i in range((end_time.date() - first_gap).days + 1)
                    if datetime.combine(first_gap + timedelta(days=i), dt_time(15)) <= now
                )
            for symbol, gap_start in gap_starts.items():
                latest_by_symbol[symbol] = min(latest_by_symbol.get(symbol, gap_start), gap_start)
            groups = _minute_increment_groups(
                symbols,
                latest_by_symbol,
                default_start=start_time,
            )

    limit = resolve_limit(
        capset,
        Cap.KLINE_MINUTE_BATCH,
        default_batch=100,
        default_rpm=30,
        default_rpm_when_unset=False,
    )

    segment_days = preferences.get_minute_sync_segment_days()
    progress_units_per_group = 100
    total_units = max(1, len(groups) * progress_units_per_group)

    # 流式落盘: 每段拉完立即写盘, 内存峰值 = 单段 (而非全量)。
    # 全量攒内存曾导致 1 年全市场分钟 K OOM 卡死 (3 亿行 / 数十 GB)。
    for group_index, (group_symbols, start_time, label) in enumerate(groups):
        if start_time >= end_time:
            continue

        def _group_chunk_done(cur: int, total: int, seg_label: str) -> None:
            if on_chunk_done is None:
                return
            local = int(progress_units_per_group * cur / max(total, 1))
            done = min(group_index * progress_units_per_group + local, total_units)
            on_chunk_done(done, total_units, f"{label} · {seg_label}".strip(" ·"))

        try:
            _group_chunk_done(0, 100, f"请求分钟源, 本组 {len(group_symbols)} 只")
            sync_minute_batch(
                group_symbols, start_time=_as_beijing(start_time), end_time=_as_beijing(end_time),
                batch_size=limit.batch, rpm=limit.rpm,
                on_chunk_done=_group_chunk_done,
                segment_trading_days=segment_days,
                on_segment=_persist,
                asset_type=asset_type,
                on_source_issues=on_source_issues,
                raw_basis=raw_basis,
            )
        except Exception as exc:
            if getattr(exc, "reason", None) != "minute_validation_failed":
                exc.written_rows = written_box[0]
                if written_box[0]:
                    _refresh_minute_view(repo, asset_type)
                raise
            # The maintenance post-check retains these stocks in the retry queue.
            # Data conflicts in an earlier start-date group must not starve later groups.
            logger.warning("%s", exc)
            _group_chunk_done(1, 1, str(exc))

    if written_box[0] == 0:
        return 0
    written = written_box[0]

    _refresh_minute_view(repo, asset_type)

    logger.info("minute K synced: %d rows (%d symbols)", written, len(symbols))
    return written



def _refresh_minute_view(repo, asset_type="stock"):
    # Publish successful streamed segments even when a later source request fails.
    from app.jobs.daily_pipeline import _refresh_single_view

    table = _minute_table_dir(repo, asset_type).name
    try:
        _refresh_single_view(repo, table)
    except Exception as e:  # noqa: BLE001
        logger.warning("refresh %s view failed: %s", table, e)
