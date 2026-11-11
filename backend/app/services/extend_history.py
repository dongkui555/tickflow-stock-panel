"""向前扩展历史数据 — 完全独立于 daily_pipeline 的盘后管道。

用户从日 K 卡片手动触发,指定往前补的时长 (x 天/月/年)。
流程:
  1. 获取当前最早日期
  2. 向前拉日 K batch (start = 最早日期 - offset, end = 最早日期)
  3. 向前拉除权因子 (同范围)
  4. 全量重算 enriched
  5. 刷新视图 + 缓存

⚠️ 本模块不导入 daily_pipeline 的任何函数,只复用基础设施:
  - kline_sync.sync_and_persist_daily_batch / sync_adj_factor
  - indicators.pipeline.run_pipeline
  - pipeline_jobs.JobStore
  - tickflow.repository.KlineRepository
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta

import polars as pl

from app.services import kline_sync
from app.services.pipeline_jobs import job_store
from app.tickflow.capabilities import Cap, CapabilitySet
from app.tickflow.repository import KlineRepository

logger = logging.getLogger(__name__)


def _noop(stage: str, pct: int, msg: str, **kwargs) -> None:  # noqa: ARG001
    pass


def _invalidate(table: str | None = None) -> None:
    from app.api.data import invalidate_data_cache
    invalidate_data_cache(table)


def _resolve_universe(capset: CapabilitySet, repo=None) -> list[str]:
    """解析标的池 — 与 daily_pipeline 同口径: instruments 维表 + 自选 (单一来源)。

    不再拉 CN_Equity_A 池 (含 SW1 聚合兜底); 维表由盘后管道同步维护。
    repo 传入时过滤自选里的指数 symbol (指数日K走独立存储)。
    """
    from pathlib import Path

    import polars as pl

    from app.config import settings
    from app.tickflow.pools import get_pool

    base: set[str] = set(get_pool("watchlist"))
    inst_path = Path(settings.data_dir) / "instruments" / "instruments.parquet"
    if inst_path.exists():
        try:
            inst = pl.read_parquet(inst_path, columns=["symbol"])
            base.update(inst["symbol"].to_list())
        except Exception as e:  # noqa: BLE001
            logger.warning("instruments 维表读取失败, 标的池仅剩自选: %s", e)
    if repo is not None:
        base -= set(repo.get_index_symbol_set())
    return sorted(base)


_FULL_MARKET_MIN_SYMBOLS = 1000
_FULL_HISTORY_SEGMENT_DAYS = 62
_INCREMENTAL_REFRESH_DAYS = 10
_EVENT_REFRESH_DAYS = 14
_FINANCIAL_REFRESH_DAYS = 240
_EDGE_TOLERANCE_DAYS = 10


def _symbols_from_instruments(repo: KlineRepository) -> list[str]:
    """从刚落盘的股票维表读取标的，避免命中旧的 repository 内存缓存。"""
    from pathlib import Path

    import polars as pl

    path = Path(repo.store.data_dir) / "instruments" / "instruments.parquet"
    if not path.exists():
        return []
    try:
        frame = pl.read_parquet(path, columns=["symbol"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("read instruments parquet failed: %s", exc)
        return []
    if frame.is_empty() or "symbol" not in frame.columns:
        return []
    return sorted({
        str(symbol).strip()
        for symbol in frame["symbol"].drop_nulls().to_list()
        if str(symbol).strip()
    })


def _resolve_full_market_universe(
    repo: KlineRepository,
    capset: CapabilitySet,
) -> tuple[list[str], str]:
    """解析全市场 A 股标的池，拒绝退化为演示或自选标的池。"""
    symbols = _symbols_from_instruments(repo)
    if len(symbols) >= _FULL_MARKET_MIN_SYMBOLS:
        return symbols, "股票维表"

    from app.services import preferences

    provider_name = preferences.get_daily_data_provider()
    if provider_name != "tickflow":
        try:
            from app.data_providers import custom as custom_sources

            if custom_sources.provider_has_dataset(provider_name, "daily"):
                provider = custom_sources.get_provider(provider_name)
                instruments = provider.get_instruments("stock") or []
                symbols = sorted({
                    str(item.get("symbol") or item.get("code") or "").strip()
                    for item in instruments
                    if item.get("symbol") or item.get("code")
                })
                if len(symbols) >= _FULL_MARKET_MIN_SYMBOLS:
                    return symbols, f"{provider_name} 标的接口"
        except Exception as exc:  # noqa: BLE001
            logger.warning("full market universe from provider failed: %s", exc)

    if capset.has(Cap.KLINE_DAILY_BATCH):
        try:
            from app.tickflow.pools import get_pool

            symbols = sorted(set(get_pool("CN_Equity_A", refresh=True)))
            if len(symbols) >= _FULL_MARKET_MIN_SYMBOLS:
                return symbols, "TickFlow CN_Equity_A"
        except Exception as exc:  # noqa: BLE001
            logger.warning("full market universe from TickFlow failed: %s", exc)

    raise RuntimeError(
        "无法获取完整 A 股标的池（至少需要 1000 只）。请检查日线数据源、"
        "Tushare Token / 聚合接口及股票维表同步权限。"
    )


def _date_segments(start: date, end: date, days: int = _FULL_HISTORY_SEGMENT_DAYS):
    """把长时间区间切成较小批次，控制全市场历史同步的内存占用。"""
    cursor = start
    while cursor <= end:
        segment_end = min(cursor + timedelta(days=days - 1), end)
        yield cursor, segment_end
        cursor = segment_end + timedelta(days=1)


def _merge_ranges(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    valid = sorted((start, end) for start, end in ranges if start <= end)
    if not valid:
        return []
    merged = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + timedelta(days=1):
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _segments_from_ranges(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    segments: list[tuple[date, date]] = []
    for start, end in _merge_ranges(ranges):
        segments.extend(_date_segments(start, end))
    return segments


def _partition_date_bounds(data_dir, table: str) -> tuple[date, date] | None:
    """从 date=YYYY-MM-DD 分区目录快速判断本地覆盖范围。"""
    root = data_dir / table
    if not root.exists():
        return None
    dates: list[date] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("date="):
            continue
        try:
            dates.append(date.fromisoformat(child.name[5:]))
        except ValueError:
            continue
    return (min(dates), max(dates)) if dates else None


def _partition_trade_dates(data_dir, table: str, start: date, end: date) -> list[date]:
    """读取已落盘分区日期，作为不依赖上游交易日历的同步驱动。"""
    root = data_dir / table
    if not root.exists():
        return []
    dates: list[date] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("date="):
            continue
        try:
            value = date.fromisoformat(child.name[5:])
        except ValueError:
            continue
        if start <= value <= end:
            dates.append(value)
    return sorted(set(dates))


def _parquet_date_bounds(path, column: str) -> tuple[date, date] | None:
    """从单文件 parquet 的日期列判断覆盖范围，用于复权/财务这类非分区表。"""
    if not path.exists():
        return None
    try:
        import polars as pl

        row = (
            pl.scan_parquet(path)
            .select(
                pl.col(column).cast(pl.Date, strict=False).min().alias("start"),
                pl.col(column).cast(pl.Date, strict=False).max().alias("end"),
            )
            .collect()
            .row(0)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("scan %s bounds failed: %s", path, exc)
        return None
    if not row or not row[0] or not row[1]:
        return None
    return row[0], row[1]


def _incremental_ranges_from_bounds(
    target_start: date,
    target_end: date,
    bounds: tuple[date, date] | None,
    *,
    refresh_days: int,
    fill_head: bool = True,
    edge_tolerance_days: int = _EDGE_TOLERANCE_DAYS,
) -> list[tuple[date, date]]:
    """生成增量请求区间：空库全量，已有库补历史头部 + 刷新最近窗口。"""
    if target_start > target_end:
        return []
    if bounds is None:
        return [(target_start, target_end)]
    earliest, latest = bounds
    if latest < target_start or earliest > target_end:
        return [(target_start, target_end)]

    ranges: list[tuple[date, date]] = []
    if fill_head and earliest > target_start + timedelta(days=edge_tolerance_days):
        ranges.append((target_start, min(target_end, earliest - timedelta(days=1))))

    # 无论最新日期是否已到 today，都刷新一小段尾部，覆盖 T+0/T+1 发布、停机恢复和数据修正。
    anchor = min(max(latest, target_start), target_end)
    refresh_start = max(target_start, anchor - timedelta(days=refresh_days))
    ranges.append((refresh_start, target_end))
    return _merge_ranges(ranges)


def _partitioned_incremental_segments(
    data_dir,
    table: str,
    target_start: date,
    target_end: date,
    *,
    refresh_days: int = _INCREMENTAL_REFRESH_DAYS,
    fill_head: bool = True,
) -> list[tuple[date, date]]:
    return _segments_from_ranges(
        _incremental_ranges_from_bounds(
            target_start,
            target_end,
            _partition_date_bounds(data_dir, table),
            refresh_days=refresh_days,
            fill_head=fill_head,
        )
    )


def _historical_adj_segments(data_dir, start: date, end: date, *, cumulative: bool = False) -> list[tuple[date, date]]:
    import polars as pl

    path = data_dir / "adj_factor" / "all.parquet"
    ranges = _incremental_ranges_from_bounds(
        start, end, _parquet_date_bounds(path, "trade_date"),
        refresh_days=30, fill_head=True, edge_tolerance_days=0,
    )
    # Only explicit historical repair scans interior dates. A partial download
    # must not hide gaps merely by moving the oldest/newest boundary.
    if not cumulative:
        return _segments_from_ranges(ranges)
    known = set(pl.read_parquet(path, columns=["trade_date"])["trade_date"].drop_nulls()) if path.exists() else set()
    for partition in (data_dir / "kline_daily").glob("date=*/part.parquet"):
        day = date.fromisoformat(partition.parent.name[5:])
        if start <= day <= end and day not in known:
            ranges.append((day, day))
    return _segments_from_ranges(_merge_ranges(ranges))


def _range_label(segments: list[tuple[date, date]]) -> str:
    if not segments:
        return "无需拉取"
    if len(segments) == 1:
        return f"{segments[0][0]} 至 {segments[0][1]}"
    return f"{len(segments)} 段，{segments[0][0]} 至 {segments[-1][1]}"


def _dedupe_unavailable(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for item in items:
        dataset = str(item.get("dataset") or "").strip()
        reason = str(item.get("reason") or "").strip()
        key = (dataset, reason)
        if not dataset or not reason or key in seen:
            continue
        seen.add(key)
        result.append({"dataset": dataset, "reason": reason})
    return result


def _drop_enriched_partitions(data_dir, segments: list[tuple[date, date]]) -> int:
    """删除被刷新日K覆盖的 enriched 日期分区，让增量计算重建这些日期。"""
    import shutil

    root = data_dir / "kline_daily_enriched"
    if not root.exists():
        return 0
    dropped = 0
    for start, end in _merge_ranges(segments):
        cursor = start
        while cursor <= end:
            partition = root / f"date={cursor.isoformat()}"
            if partition.exists():
                shutil.rmtree(partition)
                dropped += 1
            cursor += timedelta(days=1)
    return dropped


def _financial_history_bounds(data_dir) -> tuple[date, date] | None:
    bounds: list[tuple[date, date]] = []
    for table in ("metrics", "income", "balance_sheet", "cash_flow", "shares"):
        table_bounds = _parquet_date_bounds(
            data_dir / "financials" / table / "part.parquet",
            "period_end",
        )
        if table_bounds:
            bounds.append(table_bounds)
    if not bounds:
        return None
    return min(start for start, _ in bounds), max(end for _, end in bounds)


def _refresh_single_view(repo: KlineRepository, name: str) -> None:
    """刷新单个 DuckDB 视图。"""
    d = repo.store.data_dir.as_posix()
    paths = {
        "kline_daily": f"{d}/kline_daily/**/*.parquet",
        "kline_enriched": f"{d}/kline_daily_enriched/**/*.parquet",
        "kline_minute": f"{d}/kline_minute/**/*.parquet",
        "adj_factor": f"{d}/adj_factor/**/*.parquet",
        "instruments": f"{d}/instruments/**/*.parquet",
    }
    path = paths.get(name)
    if not path:
        return
    repo.refresh_view(name, path)


def compute_offset(value: int, unit: str) -> timedelta:
    """将用户输入的 value + unit 转成 timedelta。"""
    if unit == "day":
        return timedelta(days=value)
    elif unit == "month":
        return timedelta(days=value * 30)
    elif unit == "year":
        return timedelta(days=value * 365)
    else:
        raise ValueError(f"不支持的单位: {unit}")


def run_extend_history(
    repo: KlineRepository,
    capset: CapabilitySet,
    value: int,
    unit: str,
    on_progress: Callable | None = None,
) -> dict:
    """向前扩展历史数据的主函数。

    完全独立于 daily_pipeline.run_now(),不调用其任何逻辑。
    返回结果 dict 供 job_store 记录。
    """
    emit = on_progress or _noop

    # 0. 计算时间偏移
    offset = compute_offset(value, unit)
    today = date.today()

    # 1. 获取当前最早日期
    emit("extend_history", 2, "检查当前数据范围…")
    earliest = repo.earliest_daily_date()

    if not earliest:
        return {"error": "本地无日K数据,请先执行一次完整同步"}

    new_start = earliest - offset
    # 不能超过今天
    if new_start >= earliest:
        return {"error": "扩展范围无效,请增大时间跨度"}

    # 2. 解析标的池
    emit("extend_history", 5, "解析标的池…")
    universe = _resolve_universe(capset, repo)
    if not universe:
        return {"error": "标的池为空"}
    emit("extend_history", 8, f"标的池: {len(universe)} 只")

    start_str = new_start.strftime("%Y-%m-%d")
    end_str = earliest.strftime("%Y-%m-%d")

    # 3. 拉日 K
    emit("extend_history", 10, f"获取日K [{start_str} ~ {end_str}]…")
    logger.info("extend_history: daily K [%s ~ %s], %d symbols", start_str, end_str, len(universe))

    def _daily_chunk(cur: int, tot: int) -> None:
        emit("extend_history", 10 + int(35 * cur / tot),
             f"日K 批次 {cur}/{tot}", stage_pct=int(100 * cur / tot), skip_log=True)

    written_daily = kline_sync.sync_and_persist_daily_batch(
        universe, repo, capset,
        start_date=datetime.combine(new_start, datetime.min.time()),
        end_date=datetime.combine(earliest, datetime.min.time()),
        on_chunk_done=_daily_chunk,
    )
    emit("extend_history", 45, f"日K 完成,写入 {written_daily} 行")
    logger.info("extend_history: daily K done, %d rows", written_daily)
    _refresh_single_view(repo, "kline_daily")
    _invalidate("daily")

    # 4. 拉除权因子 (新范围)
    written_adj = 0
    adj_start = datetime.combine(new_start, datetime.min.time())
    adj_end = datetime.combine(today, datetime.min.time())
    adj_start_str = new_start.strftime("%Y-%m-%d")
    adj_end_str = today.strftime("%Y-%m-%d")

    from app.services import preferences as _prefs
    adj_provider = _prefs.get_adj_factor_provider()
    can_sync_adj = capset.has(Cap.ADJ_FACTOR) or adj_provider != "tickflow"
    if can_sync_adj:
        emit("extend_history", 48, f"获取除权因子 [{adj_start_str} ~ {adj_end_str}]…")
        logger.info("extend_history: adj_factor [%s ~ %s]", adj_start_str, adj_end_str)

        def _adj_chunk(cur: int, tot: int) -> None:
            emit("extend_history", 48 + int(10 * cur / tot),
                 f"除权因子批次 {cur}/{tot}", stage_pct=int(100 * cur / tot), skip_log=True)

        written_adj, _affected = kline_sync.sync_adj_factor(
            universe, repo, capset,
            start_time=adj_start, end_time=adj_end,
            on_chunk_done=_adj_chunk,
        )
        emit("extend_history", 60, f"除权因子完成,{written_adj} 行")
        logger.info("extend_history: adj_factor done, %d rows", written_adj)
        _refresh_single_view(repo, "adj_factor")
        _invalidate("adj_factor")
    else:
        emit("extend_history", 60, "除权因子跳过(无权限)")
        logger.info("extend_history: adj_factor skipped, no ADJ_FACTOR capability")

    # 5. 全量重算 enriched
    emit("extend_history", 65, "全量计算 enriched…")
    logger.info("extend_history: full enriched rebuild start")

    from app.indicators.pipeline import run_pipeline
    written_enriched = run_pipeline()

    enriched_dir = repo.store.data_dir / "kline_daily_enriched"
    enriched_days = len(list(enriched_dir.glob("date=*"))) if enriched_dir.exists() else 0
    emit("extend_history", 92, f"enriched 完成,覆盖 {enriched_days} 天")
    logger.info("extend_history: enriched done, %d days", enriched_days)
    _refresh_single_view(repo, "kline_enriched")
    _invalidate("enriched")

    # 6. 刷新视图
    emit("extend_history", 95, "刷新视图…")
    _refresh_single_view(repo, "kline_daily")
    _refresh_single_view(repo, "kline_enriched")
    _refresh_single_view(repo, "adj_factor")
    _invalidate(None)

    # 7. 统计结果
    daily_dir = repo.store.data_dir / "kline_daily"
    daily_days = len(list(daily_dir.glob("date=*"))) if daily_dir.exists() else 0

    emit("extend_history", 100, f"完成,已扩展至 {new_start}")

    return {
        "earliest_before": earliest.isoformat(),
        "earliest_after": new_start.isoformat(),
        "daily_rows": written_daily,
        "daily_days": daily_days,
        "adj_factor_rows": written_adj,
        "enriched_days": enriched_days,
        "universe_size": len(universe),
    }


def run_sync_recent_history(
    repo: KlineRepository,
    capset: CapabilitySet,
    years: int = 2,
    on_progress: Callable | None = None,
) -> dict:
    """同步近 N 年全市场 A 股日线、复权因子和技术指标。

    首次空库可直接运行。同步按小时间段顺序落盘，既避免一次性占用过多内存，
    也使再次执行时能复用已经落盘的分区数据。
    """
    if years < 1 or years > 5:
        raise ValueError("years 仅支持 1 到 5 年")

    emit = on_progress or _noop
    today = date.today()
    start = today - timedelta(days=years * 365)

    # 先更新维表，确保 Tushare 等自定义数据源的全市场标的可立即用于后续同步。
    emit("sync_instruments", 2, "同步全市场股票维表…")
    from app.services import instrument_sync

    instrument_rows = instrument_sync.sync_instruments(repo.store.data_dir)
    if instrument_rows:
        _refresh_single_view(repo, "instruments")
        _invalidate("instruments")
    emit("sync_instruments", 8, f"股票维表完成，{instrument_rows} 只标的")

    emit("resolve_universe", 10, "校验全市场标的池…")
    universe, universe_source = _resolve_full_market_universe(repo, capset)
    emit("resolve_universe", 12, f"全市场标的池：{len(universe)} 只（{universe_source}）")

    daily_segments = _partitioned_incremental_segments(
        repo.store.data_dir,
        "kline_daily",
        start,
        today,
        refresh_days=_INCREMENTAL_REFRESH_DAYS,
        fill_head=True,
    )
    total_segments = max(len(daily_segments), 1)
    written_daily = 0

    emit("sync_daily", 12, f"日 K 增量计划：{_range_label(daily_segments)}")
    for index, (segment_start, segment_end) in enumerate(daily_segments, start=1):
        base_pct = 12 + int(43 * (index - 1) / total_segments)
        span_pct = max(1, int(43 / total_segments))
        emit(
            "sync_daily",
            base_pct,
            f"同步日 K {index}/{total_segments}：{segment_start} 至 {segment_end}",
        )

        def _daily_chunk(cur: int, total: int, *, _base: int = base_pct, _span: int = span_pct) -> None:
            emit(
                "sync_daily",
                min(55, _base + int(_span * cur / max(total, 1))),
                f"日 K {index}/{total_segments}，数据批次 {cur}/{total}",
                stage_pct=int(100 * cur / max(total, 1)),
                skip_log=True,
            )

        written_daily += kline_sync.sync_and_persist_daily_batch(
            universe,
            repo,
            capset,
            start_date=datetime.combine(segment_start, datetime.min.time()),
            end_date=datetime.combine(segment_end, datetime.min.time()),
            on_chunk_done=_daily_chunk,
        )

    daily_bounds_after = _partition_date_bounds(repo.store.data_dir, "kline_daily")
    daily_has_recent = bool(
        daily_bounds_after
        and daily_bounds_after[0] <= start + timedelta(days=_EDGE_TOLERANCE_DAYS)
        and daily_bounds_after[1] >= today - timedelta(days=_INCREMENTAL_REFRESH_DAYS)
    )
    if written_daily <= 0 and not daily_has_recent:
        return {
            "error": (
                "未获取到任何日 K。请确认已将“日线数据源”切换为可用的 Tushare，"
                "并检查 Token、聚合接口地址及对应 daily 权限。"
            ),
        }

    _refresh_single_view(repo, "kline_daily")
    _invalidate("daily")
    emit("sync_daily", 55, f"日 K 完成，增量请求写入 {written_daily:,} 行")

    written_adj = 0
    adj_symbols = 0
    affected_adj_symbols: set[str] = set()
    from app.services import preferences as _prefs

    adj_provider = _prefs.get_adj_factor_provider()
    if adj_provider == "same_as_daily":
        adj_provider = _prefs.get_daily_data_provider()
    can_sync_adj = capset.has(Cap.ADJ_FACTOR) or adj_provider != "tickflow"

    adj_segments = _historical_adj_segments(
        repo.store.data_dir, start, today, cumulative=adj_provider == "tushare",
    )
    if can_sync_adj:
        emit("sync_adj", 57, f"复权因子增量计划：{_range_label(adj_segments)}")
        total_adj_segments = max(len(adj_segments), 1)
        for index, (segment_start, segment_end) in enumerate(adj_segments, start=1):
            base_pct = 57 + int(18 * (index - 1) / total_adj_segments)
            span_pct = max(1, int(18 / total_adj_segments))
            emit(
                "sync_adj",
                base_pct,
                f"同步复权因子 {index}/{total_adj_segments}：{segment_start} 至 {segment_end}",
            )

            def _adj_chunk(cur: int, total: int, *, _base: int = base_pct, _span: int = span_pct) -> None:
                emit(
                    "sync_adj",
                    min(75, _base + int(_span * cur / max(total, 1))),
                    f"复权因子 {index}/{total_adj_segments}，数据批次 {cur}/{total}",
                    stage_pct=int(100 * cur / max(total, 1)),
                    skip_log=True,
                )

            rows, affected = kline_sync.sync_adj_factor(
                universe,
                repo,
                capset,
                start_time=datetime.combine(segment_start, datetime.min.time()),
                end_time=datetime.combine(segment_end, datetime.min.time()),
                on_chunk_done=_adj_chunk,
            )
            written_adj += rows
            adj_symbols += len(affected)
            affected_adj_symbols.update(str(symbol) for symbol in affected)
        _refresh_single_view(repo, "adj_factor")
        _invalidate("adj_factor")
        emit("sync_adj", 75, f"复权因子完成，增量新增 {written_adj:,} 行")
    else:
        emit("sync_adj", 75, "复权因子跳过（当前数据源无权限）")

    enriched_bounds_before = _partition_date_bounds(repo.store.data_dir, "kline_daily_enriched")
    needs_full_enriched = (
        enriched_bounds_before is None
        or any(segment_start < enriched_bounds_before[0] for segment_start, _ in daily_segments)
    )
    emit(
        "compute_enriched",
        78,
        "全量重算技术指标和交易信号…"
        if needs_full_enriched
        else "增量计算技术指标和交易信号…",
    )
    from app.indicators.pipeline import run_pipeline

    if needs_full_enriched:
        written_enriched = run_pipeline(data_dir=repo.store.data_dir)
    else:
        dropped_days = _drop_enriched_partitions(repo.store.data_dir, daily_segments)
        written_enriched = run_pipeline(
            data_dir=repo.store.data_dir,
            new_dates_only=True,
            symbols=sorted(affected_adj_symbols) or None,
        )
        if dropped_days:
            logger.info("enriched incremental refresh: dropped %d partitions", dropped_days)
    enriched_dir = repo.store.data_dir / "kline_daily_enriched"
    enriched_days = len(list(enriched_dir.glob("date=*"))) if enriched_dir.exists() else 0
    emit("compute_enriched", 94, f"技术指标完成，写入 {written_enriched:,} 行")

    emit("refresh_views", 96, "刷新本地数据视图和缓存…")
    repo.rebuild_views()
    repo.clear_cache()
    repo.refresh_cache(background=True)
    _invalidate(None)

    daily_dir = repo.store.data_dir / "kline_daily"
    daily_days = len(list(daily_dir.glob("date=*"))) if daily_dir.exists() else 0
    emit("done", 100, f"近 {years} 年全市场历史同步完成")

    return {
        "start_date": start.isoformat(),
        "end_date": today.isoformat(),
        "years": years,
        "segments": total_segments,
        "instrument_rows": instrument_rows,
        "universe_size": len(universe),
        "universe_source": universe_source,
        "daily_rows": written_daily,
        "daily_days": daily_days,
        "adj_factor_rows": written_adj,
        "adj_factor_symbols": adj_symbols,
        "enriched_rows": written_enriched,
        "enriched_days": enriched_days,
    }


def _get_daily_custom_provider(dataset: str):
    """返回当前日线源中可复用的数据集 provider。

    两年建库以日线源为唯一主数据源。Tushare 同时支持 daily、auction、moneyflow，
    因而无需再要求用户为资金和竞价单独配置另一个数据源。
    """
    from app.services import preferences
    from app.data_providers import custom as custom_sources

    provider_name = preferences.get_daily_data_provider()
    if provider_name == "tickflow":
        return None, "当前日线数据源为 TickFlow，未配置可复用的集合竞价/资金流数据源"
    if not custom_sources.provider_has_dataset(provider_name, dataset):
        return None, f"{provider_name} 未声明 {dataset} 数据集"
    try:
        return custom_sources.get_provider(provider_name), ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("resolve %s provider failed: %s", dataset, exc)
        return None, f"{provider_name} 初始化失败：{exc}"


def _sync_partitioned_dataset(
    *,
    repo: KlineRepository,
    provider,
    universe: list[str],
    segments: list[tuple[date, date]],
    method_name: str,
    persist: Callable,
    stage: str,
    label: str,
    start_pct: int,
    end_pct: int,
    emit: Callable,
    extra_kwargs: dict | None = None,
    trade_dates_provider: Callable[[date, date], list[date]] | None = None,
) -> int:
    """分段拉取按交易日返回的全市场数据并按日期落库。"""
    method = getattr(provider, method_name, None)
    if method is None:
        raise RuntimeError(f"当前数据源未实现 {method_name}")
    try:
        accepts_trade_dates = "trade_dates" in inspect.signature(method).parameters
    except Exception:
        accepts_trade_dates = False

    total_segments = max(len(segments), 1)
    written = 0
    for index, (segment_start, segment_end) in enumerate(segments, start=1):
        base_pct = start_pct + int((end_pct - start_pct) * (index - 1) / total_segments)
        span_pct = max(1, int((end_pct - start_pct) / total_segments))
        emit(stage, base_pct, f"{label} {index}/{total_segments}：{segment_start} 至 {segment_end}")

        def _chunk(
            current: int,
            total: int,
            *,
            _base: int = base_pct,
            _span: int = span_pct,
        ) -> None:
            ratio = current / max(total, 1)
            emit(
                stage,
                min(end_pct, _base + int(_span * ratio)),
                f"{label} {index}/{total_segments}，数据批次 {current}/{total}",
                stage_pct=int(100 * ratio),
                skip_log=True,
            )

        kwargs = dict(extra_kwargs or {})
        if trade_dates_provider is not None and accepts_trade_dates:
            kwargs["trade_dates"] = trade_dates_provider(segment_start, segment_end)
        df = method(
            universe,
            start_time=datetime.combine(segment_start, datetime.min.time()),
            end_time=datetime.combine(segment_end, datetime.min.time()),
            on_chunk_done=_chunk,
            **kwargs,
        )
        if df is None or df.is_empty():
            continue
        persist(df)
        written += df.height

    return written


def run_sync_recent_enhanced_tail(
    repo: KlineRepository,
    capset: CapabilitySet,
    days: int = _INCREMENTAL_REFRESH_DAYS,
    on_progress: Callable | None = None,
) -> dict:
    """轻量补齐资金流、集合竞价和新闻尾部数据。

    用于普通日更/手动刷新后的保险补差：只刷新最近 N 个自然日覆盖的交易日，
    避免因为盘后全量增强任务被重启打断后，moneyflow 永远停在 T-1。
    """
    emit = on_progress or _noop
    tail_days = max(1, min(60, int(days or _INCREMENTAL_REFRESH_DAYS)))
    end = date.today()
    start = end - timedelta(days=tail_days)
    universe = _symbols_from_instruments(repo) or _resolve_universe(capset)
    result = {
        "moneyflow_rows": 0,
        "auction_rows": 0,
        "news_rows": {"domestic": 0, "overseas": 0},
        "unavailable_datasets": [],
    }
    unavailable: list[dict[str, str]] = []
    if not universe:
        unavailable.append({"dataset": "enhanced_tail", "reason": "股票维表为空，无法补齐增强数据"})
        result["unavailable_datasets"] = unavailable
        return result

    def _tail_segments(table: str) -> list[tuple[date, date]]:
        bounds = _partition_date_bounds(repo.store.data_dir, table)
        if bounds is None:
            return _segments_from_ranges([(start, end)])
        latest = bounds[1]
        if latest >= end:
            tail_start = end
        else:
            # 刷新最新已有日 + 缺失日：既补 T 日，也覆盖 T-1 盘后修正。
            tail_start = max(start, latest)
        return _segments_from_ranges([(tail_start, end)])

    trade_dates_provider = lambda s, e: _partition_trade_dates(repo.store.data_dir, "kline_daily", s, e)

    provider, reason = _get_daily_custom_provider("moneyflow")
    if provider is None:
        unavailable.append({"dataset": "moneyflow", "reason": reason})
    else:
        try:
            moneyflow_segments = _tail_segments("moneyflow")
            emit("sync_moneyflow_tail", 0, f"补齐主力/大单资金流尾部：{_range_label(moneyflow_segments)}")
            rows = _sync_partitioned_dataset(
                repo=repo,
                provider=provider,
                universe=universe,
                segments=moneyflow_segments,
                method_name="get_moneyflow",
                persist=repo.append_moneyflow,
                stage="sync_moneyflow_tail",
                label="补齐主力/大单资金流",
                start_pct=0,
                end_pct=45,
                emit=emit,
                trade_dates_provider=trade_dates_provider,
            )
            result["moneyflow_rows"] = rows
            if rows:
                _invalidate("moneyflow")
            else:
                unavailable.append({
                    "dataset": "moneyflow",
                    "reason": "资金流尾部接口未返回数据；请确认 moneyflow 权限、网关接口和当天发布时间。",
                })
        except Exception as exc:  # noqa: BLE001
            logger.exception("sync moneyflow tail failed")
            unavailable.append({"dataset": "moneyflow", "reason": f"资金流尾部同步失败：{exc}"})

    provider, reason = _get_daily_custom_provider("auction")
    if provider is None:
        unavailable.append({"dataset": "auction_open", "reason": reason})
    else:
        try:
            auction_segments = _tail_segments("auction_open")
            emit("sync_auction_tail", 45, f"补齐开盘集合竞价尾部：{_range_label(auction_segments)}")
            rows = _sync_partitioned_dataset(
                repo=repo,
                provider=provider,
                universe=universe,
                segments=auction_segments,
                method_name="get_auction",
                persist=repo.append_auction_open,
                stage="sync_auction_tail",
                label="补齐开盘集合竞价",
                start_pct=45,
                end_pct=75,
                emit=emit,
                extra_kwargs={"market": "o"},
                trade_dates_provider=trade_dates_provider,
            )
            result["auction_rows"] = rows
            if rows:
                _invalidate("auction_open")
            else:
                unavailable.append({
                    "dataset": "auction_open",
                    "reason": "集合竞价尾部接口未返回数据；请确认 stk_auction_o 权限、网关接口和查询日期范围。",
                })
        except Exception as exc:  # noqa: BLE001
            logger.exception("sync auction tail failed")
            unavailable.append({"dataset": "auction_open", "reason": f"集合竞价尾部同步失败：{exc}"})

    try:
        from app.services.news_sync import sync_news_history

        emit("sync_news_tail", 75, f"补齐国内外新闻尾部：{start} 至 {end}")
        news_result = sync_news_history(repo.store.data_dir, start, end, repo)
        news_rows = {
            "domestic": int(news_result.get("domestic_rows", 0)),
            "overseas": int(news_result.get("overseas_rows", 0)),
        }
        result["news_rows"] = news_rows
        unavailable.extend(news_result.get("unavailable", []))
        if sum(news_rows.values()):
            _invalidate("news")
    except Exception as exc:  # noqa: BLE001
        logger.exception("sync news tail failed")
        unavailable.append({"dataset": "news", "reason": f"新闻尾部同步失败：{exc}"})

    from app.market_time import cn_now
    now = cn_now()
    daily_days = _partition_trade_dates(repo.store.data_dir, "kline_daily", start, end)
    for dataset, hour in (("moneyflow", 19), ("auction_open", 15)):
        expected = [d for d in daily_days if d < now.date() or now.hour >= hour]
        actual = set(_partition_trade_dates(repo.store.data_dir, dataset, start, end))
        missing = sorted(set(expected) - actual)
        if missing:
            unavailable.append({"dataset": dataset, "reason": f"实际交易日仍缺 {', '.join(map(str, missing))}，旧数据写入不代表当日更新成功"})
    emit("sync_enhanced_tail", 100, "资金/竞价/新闻尾部补差完成，缺项见分项状态")
    result["unavailable_datasets"] = _dedupe_unavailable(unavailable)
    return result


def run_sync_recent_all_history(
    repo: KlineRepository,
    capset: CapabilitySet,
    years: int = 2,
    on_progress: Callable | None = None,
) -> dict:
    """建立近 N 年可落库的 A 股核心研究数据。

    包含：股票维表、日线、复权因子、技术指标、财报/股本、开盘集合竞价、主力资金流和国内外新闻。
    全市场两年分钟 K 仍需要独立的大容量/授权数据源；新闻会按已配置的 Tushare 国内源与海外新闻源
    分别同步，未授权或未返回的覆盖范围会明确列为待补，绝不会把没有同步的数据标记为完成。
    """
    emit = on_progress or _noop

    def _core_progress(
        stage: str,
        pct: int,
        msg: str,
        stage_pct: int | None = None,
        skip_log: bool = False,
    ) -> None:
        mapped_stage = "sync_core_history" if stage == "done" else stage
        emit(
            mapped_stage,
            min(60, int(pct * 0.60)),
            msg,
            stage_pct=stage_pct,
            skip_log=skip_log,
        )

    base_result = run_sync_recent_history(
        repo,
        capset,
        years=years,
        on_progress=_core_progress,
    )
    if "error" in base_result:
        return base_result

    start = date.fromisoformat(base_result["start_date"])
    end = date.fromisoformat(base_result["end_date"])
    universe = _symbols_from_instruments(repo)
    if len(universe) < _FULL_MARKET_MIN_SYMBOLS:
        return {
            "error": "日线同步完成后未能读取完整 A 股标的池，无法继续同步资金、竞价与财务数据。",
        }
    unavailable: list[dict[str, str]] = []

    # 财务报表按报告期存储；Tushare 配置为日线源时会自动复用为财务源。
    financial_rows: dict[str, int] = {}
    try:
        from app.services.financial_sync import sync_recent_history as sync_financial_history

        financial_tables = ("metrics", "income", "balance_sheet", "cash_flow", "shares")
        financial_bounds = _financial_history_bounds(repo.store.data_dir)
        if financial_bounds is None or financial_bounds[0] > start + timedelta(days=_EDGE_TOLERANCE_DAYS):
            financial_start = start
        else:
            financial_start = max(
                start,
                min(financial_bounds[1], end) - timedelta(days=_FINANCIAL_REFRESH_DAYS),
            )

        def _financial_progress(table: str, current: int, total: int) -> None:
            try:
                table_index = financial_tables.index(table)
            except ValueError:
                table_index = 0
            ratio = (table_index + current / max(total, 1)) / len(financial_tables)
            emit(
                "sync_financials",
                60 + int(18 * ratio),
                f"同步财务历史：{table} {current}/{total}",
                stage_pct=int(100 * current / max(total, 1)),
                skip_log=True,
            )

        emit("sync_financials", 60, f"同步财务增量：{financial_start} 至 {end}，并保留近两年")
        financial_rows = sync_financial_history(
            repo.store.data_dir,
            capset,
            start_time=datetime.combine(financial_start, datetime.min.time()),
            end_time=datetime.combine(end, datetime.min.time()),
            on_progress=_financial_progress,
            retention_start_time=datetime.combine(start, datetime.min.time()),
            retention_end_time=datetime.combine(end, datetime.min.time()),
        )
        financial_total = sum(financial_rows.values())
        if financial_total:
            emit("sync_financials", 78, f"财务历史完成，写入 {financial_total:,} 行")
        else:
            unavailable.append({
                "dataset": "financials",
                "reason": "当前财务数据源未返回近两年报告期；请确认 Tushare 财务接口权限和财务数据源设置。",
            })
            emit("sync_financials", 78, "财务历史未返回数据（已记录为待补）")
    except Exception as exc:  # noqa: BLE001
        logger.exception("sync recent financial history failed")
        unavailable.append({"dataset": "financials", "reason": f"财务历史同步失败：{exc}"})
        emit("sync_financials", 78, "财务历史同步失败（已记录为待补）")

    moneyflow_rows = 0
    moneyflow_segments = _partitioned_incremental_segments(
        repo.store.data_dir,
        "moneyflow",
        start,
        end,
        refresh_days=_INCREMENTAL_REFRESH_DAYS,
        fill_head=True,
    )
    provider, reason = _get_daily_custom_provider("moneyflow")
    if provider is None:
        unavailable.append({"dataset": "moneyflow", "reason": reason})
        emit("sync_moneyflow", 85, "主力资金流跳过（当前数据源不支持）")
    else:
        try:
            emit("sync_moneyflow", 78, f"主力/大单资金流增量计划：{_range_label(moneyflow_segments)}")
            moneyflow_rows = _sync_partitioned_dataset(
                repo=repo,
                provider=provider,
                universe=universe,
                segments=moneyflow_segments,
                method_name="get_moneyflow",
                persist=repo.append_moneyflow,
                stage="sync_moneyflow",
                label="同步主力/大单资金流",
                start_pct=78,
                end_pct=85,
                emit=emit,
                trade_dates_provider=lambda s, e: _partition_trade_dates(repo.store.data_dir, "kline_daily", s, e),
            )
            if moneyflow_rows:
                _invalidate("moneyflow")
                emit("sync_moneyflow", 85, f"主力/大单资金流完成，写入 {moneyflow_rows:,} 行")
            else:
                unavailable.append({
                    "dataset": "moneyflow",
                    "reason": "资金流接口未返回数据；请确认 moneyflow 权限、网关接口和查询日期范围。",
                })
                emit("sync_moneyflow", 85, "主力/大单资金流未返回数据（已记录为待补）")
        except Exception as exc:  # noqa: BLE001
            logger.exception("sync recent moneyflow failed")
            unavailable.append({"dataset": "moneyflow", "reason": f"主力资金流同步失败：{exc}"})
            emit("sync_moneyflow", 85, "主力/大单资金流同步失败（已记录为待补）")

    auction_rows = 0
    auction_segments = _partitioned_incremental_segments(
        repo.store.data_dir,
        "auction_open",
        start,
        end,
        refresh_days=_INCREMENTAL_REFRESH_DAYS,
        fill_head=True,
    )
    provider, reason = _get_daily_custom_provider("auction")
    if provider is None:
        unavailable.append({"dataset": "auction_open", "reason": reason})
        emit("sync_auction", 91, "开盘集合竞价跳过（当前数据源不支持）")
    else:
        try:
            emit("sync_auction", 85, f"开盘集合竞价增量计划：{_range_label(auction_segments)}")
            auction_rows = _sync_partitioned_dataset(
                repo=repo,
                provider=provider,
                universe=universe,
                segments=auction_segments,
                method_name="get_auction",
                persist=repo.append_auction_open,
                stage="sync_auction",
                label="同步开盘集合竞价",
                start_pct=85,
                end_pct=91,
                emit=emit,
                extra_kwargs={"market": "o"},
                trade_dates_provider=lambda s, e: _partition_trade_dates(repo.store.data_dir, "kline_daily", s, e),
            )
            if auction_rows:
                _invalidate("auction_open")
                emit("sync_auction", 91, f"开盘集合竞价完成，写入 {auction_rows:,} 行")
            else:
                unavailable.append({
                    "dataset": "auction_open",
                    "reason": "集合竞价接口未返回数据；请确认 stk_auction_o 权限、网关接口和查询日期范围。",
                })
                emit("sync_auction", 91, "开盘集合竞价未返回数据（已记录为待补）")
        except Exception as exc:  # noqa: BLE001
            logger.exception("sync recent auction failed")
            unavailable.append({"dataset": "auction_open", "reason": f"开盘集合竞价同步失败：{exc}"})
            emit("sync_auction", 91, "开盘集合竞价同步失败（已记录为待补）")

    news_rows = {"domestic": 0, "overseas": 0}
    try:
        from app.services.news_sync import sync_news_history

        news_segments = _partitioned_incremental_segments(
            repo.store.data_dir,
            "news",
            start,
            end,
            refresh_days=_EVENT_REFRESH_DAYS,
            fill_head=False,
        )

        def _news_progress(
            stage: str,
            pct: int,
            msg: str,
        ) -> None:
            emit(
                stage,
                91 + int(6 * pct / 100),
                msg,
                stage_pct=pct,
                skip_log=True,
            )

        emit("sync_news", 91, f"国内外新闻增量计划：{_range_label(news_segments)}")
        merged_news_result = {
            "domestic_rows": 0,
            "overseas_rows": 0,
            "unavailable": [],
        }
        for segment_start, segment_end in news_segments:
            news_result = sync_news_history(
                repo.store.data_dir,
                segment_start,
                segment_end,
                repo,
                on_progress=_news_progress,
            )
            merged_news_result["domestic_rows"] += int(news_result.get("domestic_rows", 0))
            merged_news_result["overseas_rows"] += int(news_result.get("overseas_rows", 0))
            merged_news_result["unavailable"].extend(news_result.get("unavailable", []))
        news_rows = {
            "domestic": int(merged_news_result.get("domestic_rows", 0)),
            "overseas": int(merged_news_result.get("overseas_rows", 0)),
        }
        unavailable.extend(merged_news_result.get("unavailable", []))
        if sum(news_rows.values()):
            _invalidate("news")
            emit(
                "sync_news",
                97,
                f"国内外新闻完成，写入 {sum(news_rows.values()):,} 行",
            )
        else:
            emit("sync_news", 97, "国内外新闻未返回数据（已记录为待补）")
    except Exception as exc:  # noqa: BLE001
        logger.exception("sync recent news failed")
        unavailable.append({"dataset": "news", "reason": f"国内外新闻同步失败：{exc}"})
        emit("sync_news", 97, "国内外新闻同步失败（已记录为待补）")

    tushare_ext_rows: dict[str, int] = {}
    try:
        from app.services.tushare_ext_sync import sync_tushare_ext_tail

        emit("sync_tushare_ext", 97, "同步 Tushare 扩展数据增量…")

        def _tushare_ext_progress(stage: str, pct: int, msg: str) -> None:
            emit(
                stage,
                97 + int(1 * min(max(pct, 0), 100) / 100),
                msg,
                stage_pct=pct,
                skip_log=True,
            )

        tushare_ext = sync_tushare_ext_tail(
            repo.store.data_dir,
            days=_EVENT_REFRESH_DAYS,
            max_catchup_days=max(90, years * 365),
            on_progress=_tushare_ext_progress,
        )
        tushare_ext_rows = {
            str(k): int(v or 0)
            for k, v in (tushare_ext.get("rows") or {}).items()
        }
        unavailable.extend(tushare_ext.get("unavailable") or [])
        emit(
            "sync_tushare_ext",
            98,
            f"Tushare 扩展数据完成，写入 {int(tushare_ext.get('total_rows', 0) or 0):,} 行",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("sync recent Tushare extension data failed")
        unavailable.append({"dataset": "tushare_ext", "reason": f"Tushare 扩展数据同步失败：{exc}"})
        emit("sync_tushare_ext", 98, "Tushare 扩展数据同步失败（已记录为待补）")

    index_rows = 0
    index_sync_result = None
    etf_rows = 0
    index_count = 0
    etf_count = 0
    try:
        from app.services import index_incremental, index_sync, preferences

        pull_index = preferences.get_pipeline_pull_index()
        pull_etf = preferences.get_pipeline_pull_etf()
        if (capset.has(Cap.KLINE_DAILY_BATCH) or index_sync.custom_asset_provider() is not None) and (pull_index or pull_etf):
            emit("sync_index", 97, "同步指数/ETF历史增量…")
            if pull_index:
                index_count = index_sync.sync_index_instruments(repo, pull_index=True, pull_etf=False)
                def _index_progress(current: int, total: int, text: str) -> None:
                    emit(
                        "sync_index",
                        97,
                        text,
                        stage_pct=int(100 * current / max(total, 1)),
                    )

                instruments = repo.get_index_instruments()
                if 'asset_type' in instruments.columns:
                    instruments = instruments.filter(pl.col('asset_type') == 'index')
                index_sync_result = index_incremental.sync(
                    repo, capset, symbols=instruments['symbol'].to_list(),
                    start_date=start, end_date=end, on_progress=_index_progress,
                )
                index_rows = index_sync_result['written_rows']
                if not index_sync_result['complete']:
                    unavailable.append({'dataset': 'index_daily', 'reason': index_incremental.message(index_sync_result)})
            if pull_etf:
                etf_count = index_sync.sync_etf_instruments(repo)
                etf_segments = _partitioned_incremental_segments(
                    repo.store.data_dir,
                    "kline_etf_daily",
                    start,
                    end,
                    refresh_days=_INCREMENTAL_REFRESH_DAYS,
                    fill_head=True,
                )

                def _etf_progress(current: int, total: int) -> None:
                    emit(
                        "sync_index",
                        98,
                        f"ETF日K批次 {current}/{total}",
                        stage_pct=int(100 * current / max(total, 1)),
                        skip_log=True,
                    )

                for segment_start, segment_end in etf_segments:
                    etf_rows += index_sync.sync_and_persist_etf_daily(
                        repo,
                        capset,
                        start_date=datetime.combine(segment_start, datetime.min.time()),
                        end_date=datetime.combine(segment_end, datetime.min.time()),
                        on_chunk_done=_etf_progress,
                    )
            if index_rows or etf_rows:
                _invalidate("index_daily")
                _invalidate("index_enriched")
                _invalidate("etf_daily")
                _invalidate("etf_enriched")
            emit("sync_index", 98,
                 (index_incremental.message(index_sync_result) if index_sync_result else '指数未启用')
                 + f"; ETF {etf_count} 只/{etf_rows:,} 行")
        elif pull_index or pull_etf:
            unavailable.append({
                "dataset": "index_etf",
                "reason": "指数/ETF 历史同步需要可用的批量日线能力，请检查 TickFlow 或专用数据源权限。",
            })
    except Exception as exc:  # noqa: BLE001
        logger.exception("sync recent index/ETF history failed")
        unavailable.append({"dataset": "index_etf", "reason": f"指数/ETF 历史同步失败：{exc}"})
        emit("sync_index", 98, "指数/ETF历史同步失败（已记录为待补）")

    minute_policy = {
        "dataset": "minute",
        "mode": "separate_tasks",
        "reason": "分钟K已拆为“历史全量补齐”和“盘中增量刷新”两个独立任务，本次增强历史建库不自动拉全市场分钟K。",
    }

    emit("refresh_views", 98, "刷新本地数据视图和缓存…")
    repo.rebuild_views()
    repo.clear_cache()
    repo.refresh_cache(background=True)
    _invalidate(None)
    emit("done", 100, f"近 {years} 年 A 股核心历史建库完成")

    unavailable = _dedupe_unavailable(unavailable)
    return {
        **base_result,
        "financial_rows": financial_rows,
        "moneyflow_rows": moneyflow_rows,
        "auction_rows": auction_rows,
        "index_count": index_count,
        "index_daily_rows": index_rows,
        "index_sync": index_sync_result,
        "etf_count": etf_count,
        "etf_daily_rows": etf_rows,
        "news_rows": news_rows,
        "tushare_ext_rows": tushare_ext_rows,
        "minute_policy": minute_policy,
        "unavailable_datasets": unavailable,
    }
