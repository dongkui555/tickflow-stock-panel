"""Tushare extension-data incremental sync.

Offline Tushare packages are useful for quickly seeding history, but the data is
only valuable for trading research if the same datasets keep updating.  This
module uses the configured Tushare provider and writes the high-value event,
sector, chips, and cross-border-flow datasets into ext_data timeseries
partitions.
"""
from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import polars as pl

from app.data_providers import custom as custom_sources
from app.services import preferences
from app.services.ext_data import ExtConfig, ExtConfigStore, ExtField

logger = logging.getLogger(__name__)


Progress = Callable[[str, int, str], None]


class ExtSyncPartialError(RuntimeError):
    def __init__(self, rows_written: int, cause: Exception):
        self.rows_written = rows_written
        super().__init__(f"已保存 {rows_written} 行, 后续日期失败: {type(cause).__name__}")


@dataclass(frozen=True)
class TushareExtSpec:
    api: str
    config_id: str
    label: str
    date_col: str
    query_mode: str
    fields: tuple[tuple[str, str, str], ...]
    unique_fields: tuple[str, ...] = ()


EXT_SPECS: dict[str, TushareExtSpec] = {
    "stk_limit": TushareExtSpec(
        api="stk_limit",
        config_id="ext_tushare_stk_limit",
        label="Tushare 涨跌停价",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("symbol", "string", "标的代码"),
            ("trade_date", "string", "交易日期"),
            ("up_limit", "float", "涨停价"),
            ("down_limit", "float", "跌停价"),
        ),
        unique_fields=("symbol", "trade_date"),
    ),
    "limit_list_d": TushareExtSpec(
        api="limit_list_d",
        config_id="ext_tushare_limit_list_d",
        label="Tushare 涨跌停明细",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("symbol", "string", "标的代码"),
            ("trade_date", "string", "交易日期"),
            ("name", "string", "名称"),
            ("industry", "string", "行业"),
            ("close", "float", "收盘价"),
            ("pct_chg", "float", "涨跌幅"),
            ("amount", "float", "成交额"),
            ("limit_amount", "float", "封单额"),
            ("float_mv", "float", "流通市值"),
            ("total_mv", "float", "总市值"),
            ("turnover_rate", "float", "换手率"),
            ("fd_amount", "float", "封单金额"),
            ("first_time", "string", "首次封板"),
            ("last_time", "string", "最后封板"),
            ("open_times", "int", "开板次数"),
            ("up_stat", "string", "连板状态"),
            ("limit_times", "float", "封板次数"),
            ("limit", "string", "涨跌停方向"),
        ),
        unique_fields=("symbol", "trade_date", "limit"),
    ),
    "top_list": TushareExtSpec(
        api="top_list",
        config_id="ext_tushare_top_list",
        label="Tushare 龙虎榜",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("symbol", "string", "标的代码"),
            ("trade_date", "string", "交易日期"),
            ("name", "string", "名称"),
            ("close", "float", "收盘价"),
            ("pct_change", "float", "涨跌幅"),
            ("turnover_rate", "float", "换手率"),
            ("amount", "float", "成交额"),
            ("l_sell", "float", "龙虎榜卖出额"),
            ("l_buy", "float", "龙虎榜买入额"),
            ("l_amount", "float", "龙虎榜成交额"),
            ("net_amount", "float", "净买入额"),
            ("net_rate", "float", "净买入占比"),
            ("amount_rate", "float", "成交额占比"),
            ("float_values", "float", "流通市值"),
            ("reason", "string", "上榜原因"),
        ),
        unique_fields=("symbol", "trade_date", "reason"),
    ),
    "top_inst": TushareExtSpec(
        api="top_inst",
        config_id="ext_tushare_top_inst",
        label="Tushare 龙虎榜营业部",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("symbol", "string", "标的代码"),
            ("trade_date", "string", "交易日期"),
            ("exalter", "string", "营业部"),
            ("buy", "float", "买入额"),
            ("buy_rate", "float", "买入占比"),
            ("sell", "float", "卖出额"),
            ("sell_rate", "float", "卖出占比"),
            ("net_buy", "float", "净买入额"),
            ("side", "string", "买卖方向"),
            ("reason", "string", "上榜原因"),
        ),
        unique_fields=("symbol", "trade_date", "exalter", "side", "reason"),
    ),
    "moneyflow_dc": TushareExtSpec(
        api="moneyflow_dc",
        config_id="ext_tushare_moneyflow_dc",
        label="Tushare 东方财富资金流",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("symbol", "string", "标的代码"),
            ("trade_date", "string", "交易日期"),
            ("name", "string", "名称"),
            ("pct_change", "float", "涨跌幅"),
            ("close", "float", "收盘价"),
            ("net_amount", "float", "净流入额"),
            ("net_amount_rate", "float", "净流入占比"),
            ("buy_elg_amount", "float", "超大单净额"),
            ("buy_elg_amount_rate", "float", "超大单占比"),
            ("buy_lg_amount", "float", "大单净额"),
            ("buy_lg_amount_rate", "float", "大单占比"),
            ("buy_md_amount", "float", "中单净额"),
            ("buy_md_amount_rate", "float", "中单占比"),
            ("buy_sm_amount", "float", "小单净额"),
            ("buy_sm_amount_rate", "float", "小单占比"),
        ),
        unique_fields=("symbol", "trade_date"),
    ),
    "moneyflow_ths": TushareExtSpec(
        api="moneyflow_ths",
        config_id="ext_tushare_moneyflow_ths",
        label="Tushare 同花顺资金流",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("symbol", "string", "标的代码"),
            ("trade_date", "string", "交易日期"),
            ("name", "string", "名称"),
            ("pct_change", "float", "涨跌幅"),
            ("latest", "float", "最新价"),
            ("net_amount", "float", "净流入额"),
            ("net_d5_amount", "float", "5日净流入"),
            ("buy_lg_amount", "float", "大单净额"),
            ("buy_lg_amount_rate", "float", "大单占比"),
            ("buy_md_amount", "float", "中单净额"),
            ("buy_md_amount_rate", "float", "中单占比"),
            ("buy_sm_amount", "float", "小单净额"),
            ("buy_sm_amount_rate", "float", "小单占比"),
        ),
        unique_fields=("symbol", "trade_date"),
    ),
    "dc_daily": TushareExtSpec(
        api="dc_daily",
        config_id="ext_tushare_dc_daily",
        label="Tushare 东方财富板块日线",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("ts_code", "string", "板块代码"),
            ("trade_date", "string", "交易日期"),
            ("name", "string", "板块名称"),
            ("open", "float", "开盘"),
            ("high", "float", "最高"),
            ("low", "float", "最低"),
            ("close", "float", "收盘"),
            ("pct_change", "float", "涨跌幅"),
            ("vol", "float", "成交量"),
            ("amount", "float", "成交额"),
        ),
        unique_fields=("ts_code", "trade_date"),
    ),
    "ths_daily": TushareExtSpec(
        api="ths_daily",
        config_id="ext_tushare_ths_daily",
        label="Tushare 同花顺板块日线",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("ts_code", "string", "板块代码"),
            ("trade_date", "string", "交易日期"),
            ("name", "string", "板块名称"),
            ("open", "float", "开盘"),
            ("high", "float", "最高"),
            ("low", "float", "最低"),
            ("close", "float", "收盘"),
            ("pct_change", "float", "涨跌幅"),
            ("vol", "float", "成交量"),
            ("amount", "float", "成交额"),
        ),
        unique_fields=("ts_code", "trade_date"),
    ),
    "sw_daily": TushareExtSpec(
        api="sw_daily",
        config_id="ext_tushare_sw_daily",
        label="Tushare 申万行业日线",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("ts_code", "string", "行业代码"),
            ("trade_date", "string", "交易日期"),
            ("name", "string", "行业名称"),
            ("open", "float", "开盘"),
            ("high", "float", "最高"),
            ("low", "float", "最低"),
            ("close", "float", "收盘"),
            ("pct_change", "float", "涨跌幅"),
            ("vol", "float", "成交量"),
            ("amount", "float", "成交额"),
        ),
        unique_fields=("ts_code", "trade_date"),
    ),
    "cyq_perf": TushareExtSpec(
        api="cyq_perf",
        config_id="ext_tushare_cyq_perf",
        label="Tushare 筹码成本分布摘要",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("symbol", "string", "标的代码"),
            ("trade_date", "string", "交易日期"),
            ("his_low", "float", "历史低位"),
            ("his_high", "float", "历史高位"),
            ("cost_5pct", "float", "5%成本"),
            ("cost_15pct", "float", "15%成本"),
            ("cost_50pct", "float", "50%成本"),
            ("cost_85pct", "float", "85%成本"),
            ("cost_95pct", "float", "95%成本"),
            ("weight_avg", "float", "加权平均成本"),
            ("winner_rate", "float", "获利比例"),
        ),
        unique_fields=("symbol", "trade_date"),
    ),
    "forecast": TushareExtSpec(
        api="forecast",
        config_id="ext_tushare_forecast",
        label="Tushare 业绩预告",
        date_col="ann_date",
        query_mode="range",
        fields=(
            ("symbol", "string", "标的代码"),
            ("ann_date", "string", "公告日期"),
            ("end_date", "string", "报告期"),
            ("type", "string", "预告类型"),
            ("p_change_min", "float", "净利变动下限%"),
            ("p_change_max", "float", "净利变动上限%"),
            ("net_profit_min", "float", "净利下限"),
            ("net_profit_max", "float", "净利上限"),
            ("last_parent_net", "float", "上年同期净利"),
            ("summary", "string", "摘要"),
            ("change_reason", "string", "变动原因"),
        ),
        unique_fields=("symbol", "ann_date", "end_date", "type"),
    ),
    "express": TushareExtSpec(
        api="express",
        config_id="ext_tushare_express",
        label="Tushare 业绩快报",
        date_col="ann_date",
        query_mode="range",
        fields=(
            ("symbol", "string", "标的代码"),
            ("ann_date", "string", "公告日期"),
            ("end_date", "string", "报告期"),
            ("revenue", "float", "营业收入"),
            ("operate_profit", "float", "营业利润"),
            ("total_profit", "float", "利润总额"),
            ("n_income", "float", "净利润"),
            ("total_assets", "float", "总资产"),
            ("total_hldr_eqy_exc_min_int", "float", "归母权益"),
            ("diluted_eps", "float", "摊薄EPS"),
            ("diluted_roe", "float", "摊薄ROE"),
            ("yoy_net_profit", "float", "净利同比%"),
            ("bps", "float", "每股净资产"),
            ("perf_summary", "string", "业绩摘要"),
        ),
        unique_fields=("symbol", "ann_date", "end_date"),
    ),
    "block_trade": TushareExtSpec(
        api="block_trade",
        config_id="ext_tushare_block_trade",
        label="Tushare 大宗交易",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("symbol", "string", "标的代码"),
            ("trade_date", "string", "交易日期"),
            ("price", "float", "成交价"),
            ("vol", "float", "成交量"),
            ("amount", "float", "成交额"),
            ("buyer", "string", "买方营业部"),
            ("seller", "string", "卖方营业部"),
        ),
        unique_fields=("symbol", "trade_date", "price", "amount", "buyer", "seller"),
    ),
    "stk_holdernumber": TushareExtSpec(
        api="stk_holdernumber",
        config_id="ext_tushare_holdernumber",
        label="Tushare 股东户数",
        date_col="ann_date",
        query_mode="range",
        fields=(
            ("symbol", "string", "标的代码"),
            ("ann_date", "string", "公告日期"),
            ("end_date", "string", "截止日期"),
            ("holder_num", "int", "股东户数"),
        ),
        unique_fields=("symbol", "ann_date", "end_date"),
    ),
    "margin": TushareExtSpec(
        api="margin",
        config_id="ext_tushare_margin",
        label="Tushare 融资融券汇总",
        date_col="trade_date",
        query_mode="range",
        fields=(
            ("trade_date", "string", "交易日期"),
            ("exchange_id", "string", "交易所"),
            ("rzye", "float", "融资余额"),
            ("rzmre", "float", "融资买入额"),
            ("rzche", "float", "融资偿还额"),
            ("rqye", "float", "融券余额"),
            ("rqmcl", "float", "融券卖出量"),
            ("rzrqye", "float", "融资融券余额"),
            ("rqyl", "float", "融券余量"),
        ),
        unique_fields=("trade_date", "exchange_id"),
    ),
    "hsgt_top10": TushareExtSpec(
        api="hsgt_top10",
        config_id="ext_tushare_hsgt_top10",
        label="Tushare 沪深股通十大成交股",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("symbol", "string", "标的代码"),
            ("trade_date", "string", "交易日期"),
            ("name", "string", "名称"),
            ("close", "float", "收盘价"),
            ("change", "float", "涨跌幅"),
            ("rank", "int", "排名"),
            ("market_type", "int", "市场类型"),
            ("amount", "float", "成交额"),
            ("net_amount", "float", "净买入额"),
            ("buy", "float", "买入额"),
            ("sell", "float", "卖出额"),
        ),
        unique_fields=("symbol", "trade_date", "market_type", "rank"),
    ),
    "ggt_top10": TushareExtSpec(
        api="ggt_top10",
        config_id="ext_tushare_ggt_top10",
        label="Tushare 港股通十大成交股",
        date_col="trade_date",
        query_mode="trade_date",
        fields=(
            ("ts_code", "string", "港股代码"),
            ("trade_date", "string", "交易日期"),
            ("name", "string", "名称"),
            ("close", "float", "收盘价"),
            ("p_change", "float", "涨跌幅"),
            ("rank", "int", "排名"),
            ("market_type", "int", "市场类型"),
            ("amount", "float", "成交额"),
            ("net_amount", "float", "净买入额"),
            ("sh_amount", "float", "沪市成交额"),
            ("sh_net_amount", "float", "沪市净买入额"),
            ("sz_amount", "float", "深市成交额"),
            ("sz_net_amount", "float", "深市净买入额"),
        ),
        unique_fields=("ts_code", "trade_date", "market_type", "rank"),
    ),
}

DEFAULT_AUTO_APIS = (
    "stk_limit",
    "limit_list_d",
    "top_list",
    "top_inst",
    "moneyflow_dc",
    "moneyflow_ths",
    "dc_daily",
    "ths_daily",
    "sw_daily",
    "cyq_perf",
    "forecast",
    "express",
    "block_trade",
    "stk_holdernumber",
    "margin",
)


def sync_tushare_ext_tail(
    data_dir: Path,
    *,
    days: int = 10,
    max_catchup_days: int = 90,
    apis: Iterable[str] = DEFAULT_AUTO_APIS,
    on_progress: Progress | None = None,
) -> dict:
    """Incrementally update selected Tushare extension datasets.

    Re-runs keep a small recent refresh window and also catch up stale imported
    tables from their latest local partition.  This avoids full resyncs while
    still closing gaps after downtime or one-off offline imports.
    """
    emit = on_progress or _noop
    today = date.today()
    provider, provider_name, reason, mode_override = _resolve_tushare_provider()
    if provider is None:
        return {
            "rows": {},
            "total_rows": 0,
            "unavailable": [{"dataset": "tushare_ext", "reason": reason}],
        }
    try:
        with _temporary_env("TUSHARE_MODE", mode_override):
            api_client = provider._ensure_api()  # noqa: SLF001
            api_client, provider_name = _with_datahub_failover(api_client, provider_name)
    except Exception as exc:  # noqa: BLE001
        return {
            "rows": {},
            "total_rows": 0,
            "unavailable": [{"dataset": "tushare_ext", "reason": f"{provider_name} 初始化失败：{exc}"}],
        }

    configured = _env_auto_apis()
    selected = configured or list(apis)
    if not configured:
        for api_name in ("hsgt_top10", "ggt_top10"):
            base = data_dir / "ext_data" / EXT_SPECS[api_name].config_id / "timeseries"
            if api_name not in selected and next(base.glob("date=*/part.parquet"), None) is not None:
                selected.append(api_name)
    wanted = [api for api in selected if api in EXT_SPECS]
    results: dict[str, int] = {}
    unavailable: list[dict[str, str]] = []
    total = max(len(wanted), 1)
    with _temporary_promax_timeout(api_client, _ext_timeout_seconds()):
        for index, api_name in enumerate(wanted, start=1):
            spec = EXT_SPECS[api_name]
            pct = int(100 * (index - 1) / total)
            try:
                ranges = _incremental_ranges_for_ext(
                    data_dir,
                    spec,
                    today,
                    refresh_days=days,
                    max_catchup_days=max_catchup_days,
                )
                emit("sync_tushare_ext", pct, f"{spec.label} 增量计划：{_range_label(ranges)}")
                rows = _sync_one_spec(
                    data_dir,
                    api_client,
                    spec,
                    ranges,
                    on_progress=lambda day: emit("sync_tushare_ext", pct, f"{spec.label} {day}"),
                )
                results[api_name] = rows
                if rows:
                    logger.info("tushare ext %s updated: %d rows", api_name, rows)
                else:
                    unavailable.append({
                        "dataset": spec.config_id,
                        "reason": f"{spec.label} 接口未返回增量数据，可能尚未发布或当前 KEY 无权限。",
                    })
            except Exception as exc:  # noqa: BLE001
                from app.services.pipeline_jobs import JobCancelledError
                if isinstance(exc, JobCancelledError):
                    raise
                logger.warning("tushare ext %s sync failed: %s", api_name, exc)
                results[api_name] = getattr(exc, "rows_written", 0)
                unavailable.append({"dataset": spec.config_id, "reason": f"{spec.label} 同步失败：{exc}"})
            emit("sync_tushare_ext", int(100 * index / total), f"{spec.label} 完成，写入 {results[api_name]} 行")

    return {
        "provider": provider_name,
        "rows": results,
        "total_rows": sum(results.values()),
        "unavailable": _dedupe_unavailable(unavailable),
    }


def _env_auto_apis() -> list[str]:
    from app.config import settings

    raw = os.getenv("TUSHARE_EXT_AUTO_APIS", "").strip() or str(settings.tushare_ext_auto_apis or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _ext_timeout_seconds() -> float:
    from app.config import settings

    raw = os.getenv("TUSHARE_EXT_TIMEOUT_SECONDS", "").strip() or str(settings.tushare_ext_timeout_seconds or 12)
    try:
        return max(3.0, min(60.0, float(raw)))
    except ValueError:
        return 12.0


def _with_datahub_failover(api_client, provider_name: str):
    """Attach datahubco as the extension API backup when configured."""
    from app.plugins.tushare.provider import TushareAPIFailover

    if isinstance(api_client, TushareAPIFailover):
        return api_client, f"{provider_name}->datahub"
    from app.config import settings

    enabled = (
        str(os.getenv("TUSHARE_DATAHUB_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}
        if os.getenv("TUSHARE_DATAHUB_ENABLED") is not None
        else bool(settings.tushare_datahub_enabled)
    )
    raw_chain = (
        os.getenv("TUSHARE_EXT_SOURCE_CHAIN", "").strip()
        or str(settings.tushare_ext_source_chain or "promax,datahub").strip()
    )
    chain = [item.strip().lower() for item in raw_chain.split(",") if item.strip()]
    if not enabled or "datahub" not in chain:
        return api_client, provider_name
    try:
        from app.plugins.tushare.provider import TushareAPIFailover, build_datahub_client

        datahub = build_datahub_client()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tushare datahub backup unavailable: %s", type(exc).__name__)
        return api_client, provider_name
    if chain and chain[0] == "datahub":
        return TushareAPIFailover(datahub, api_client, "datahub", provider_name), f"datahub->{provider_name}"
    return TushareAPIFailover(api_client, datahub, provider_name, "datahub"), f"{provider_name}->datahub"


@contextmanager
def _temporary_promax_timeout(api_client, seconds: float):
    candidates = [api_client, getattr(api_client, "primary", None), getattr(api_client, "backup", None)]
    clients = []
    for candidate in candidates:
        client = getattr(candidate, "_client", None)
        if client is not None and hasattr(client, "timeout") and client not in clients:
            clients.append(client)
    if not clients:
        yield
        return
    old_timeouts = [(client, client.timeout) for client in clients]
    try:
        import httpx

        for client in clients:
            client.timeout = httpx.Timeout(seconds)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        for client, old_timeout in old_timeouts:
            client.timeout = old_timeout


@contextmanager
def _temporary_env(name: str, value: str | None):
    if value is None:
        yield
        return
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _resolve_tushare_provider() -> tuple[object | None, str, str, str | None]:
    from app.config import settings

    ext_mode = (
        os.getenv("TUSHARE_EXT_MODE", "").strip()
        or str(settings.tushare_ext_mode or "current").strip()
    ).lower()
    if ext_mode in {"official", "promax"}:
        try:
            from app.plugins.tushare.provider import TushareProvider
        except Exception as exc:  # noqa: BLE001
            return None, "", f"Tushare 插件不可用：{exc}", None
        return TushareProvider(), f"tushare:{ext_mode}", "", ext_mode

    candidates = []
    daily_provider = preferences.get_daily_data_provider()
    if daily_provider and daily_provider != "tickflow":
        candidates.append(daily_provider)
    if "tushare" not in candidates:
        candidates.append("tushare")

    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            provider = custom_sources.get_provider(name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("resolve provider %s failed: %s", name, exc)
            continue
        if hasattr(provider, "_ensure_api"):
            return provider, name, "", None
    return None, "", "未找到可用 Tushare 数据源；请在设置中配置 TUSHARE_TOKEN 或 Promax 网关。", None


def _incremental_ranges_for_ext(
    data_dir: Path,
    spec: TushareExtSpec,
    today: date,
    *,
    refresh_days: int,
    max_catchup_days: int,
) -> list[tuple[date, date]]:
    latest = _latest_ext_date(data_dir, spec.config_id)
    refresh_start = today - timedelta(days=max(1, refresh_days))
    if latest is None:
        start = refresh_start
    else:
        catchup_start = min(latest + timedelta(days=1), today)
        capped_start = max(catchup_start, today - timedelta(days=max(1, max_catchup_days)))
        start = min(refresh_start, capped_start)
    ranges = [(start, today)]
    if spec.query_mode == "trade_date":
        base = data_dir / "ext_data" / spec.config_id / "timeseries"
        existing = {p.name[5:] for p in base.glob("date=????-??-??") if p.is_dir()}
        if existing:
            floor = max(date.fromisoformat(min(existing)), today - timedelta(days=max(1, max_catchup_days)))
            # Saving a fresh partition must not hide older holes behind max(date).
            ranges.extend((day, day) for day in _trade_dates(data_dir, floor, start - timedelta(days=1))
                          if day.isoformat() not in existing)
    return _merge_ranges(ranges)


def _latest_ext_date(data_dir: Path, config_id: str) -> date | None:
    base = data_dir / "ext_data" / config_id / "timeseries"
    if not base.exists():
        return None
    dates: list[date] = []
    for child in base.iterdir():
        if not child.is_dir() or not child.name.startswith("date="):
            continue
        try:
            dates.append(date.fromisoformat(child.name[5:]))
        except ValueError:
            continue
    return max(dates) if dates else None


def _sync_one_spec(
    data_dir: Path,
    api_client,
    spec: TushareExtSpec,
    ranges: list[tuple[date, date]],
    on_progress=None,
) -> int:
    method = getattr(api_client, spec.api, None)
    if method is None:
        raise RuntimeError(f"Tushare 客户端未实现 {spec.api}")
    total_rows = 0
    if spec.query_mode == "trade_date":
        days = set()
        for start, end in ranges:
            days.update(_trade_dates(data_dir, start, end))
        for day in sorted(days, reverse=True):
            try:
                frames = _fetch_frames(method, spec, day, day, data_dir, on_progress=on_progress)
                for frame in frames:
                    total_rows += _write_timeseries_frame(data_dir, spec, frame)
            except Exception as exc:
                from app.services.pipeline_jobs import JobCancelledError
                if isinstance(exc, JobCancelledError):
                    raise
                raise ExtSyncPartialError(total_rows, exc) from exc
        return total_rows
    for start, end in ranges:
        frames = _fetch_frames(method, spec, start, end, data_dir, on_progress=on_progress)
        if not frames:
            continue
        merged = pl.concat(frames, how="diagonal_relaxed")
        total_rows += _write_timeseries_frame(data_dir, spec, merged)
    return total_rows


def _fetch_frames(method, spec: TushareExtSpec, start: date, end: date, data_dir: Path, on_progress=None) -> list[pl.DataFrame]:
    frames: list[pl.DataFrame] = []
    if spec.query_mode == "trade_date":
        dates = _trade_dates(data_dir, start, end)
        if not dates:
            dates = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
        for day in dates:
            if on_progress:
                on_progress(day)
            frame = _call_tushare(method, spec, {spec.date_col: _yyyymmdd(day)})
            if frame is not None and not frame.is_empty():
                from app.services.market_repair import validate_partition
                frame = validate_partition(frame, day, spec.date_col)
                frames.append(frame)
    else:
        frame = _call_tushare(
            method,
            spec,
            {"start_date": _yyyymmdd(start), "end_date": _yyyymmdd(end)},
        )
        if frame is not None and not frame.is_empty():
            if spec.date_col not in frame.columns or frame.filter(
                ~pl.col(spec.date_col).is_between(pl.lit(start.isoformat()), pl.lit(end.isoformat()))
                | pl.col(spec.date_col).is_null()
            ).height:
                raise ValueError(f"{spec.api} returned records outside requested date range")
            frames.append(frame)
    return frames


def _call_tushare(method, spec: TushareExtSpec, params: dict[str, str]) -> pl.DataFrame | None:
    fields = _request_fields(spec)
    kwargs = dict(params)
    if fields:
        kwargs["fields"] = ",".join(fields)
    try:
        frame = method(**kwargs)
    except TypeError:
        kwargs.pop("fields", None)
        frame = method(**kwargs)
    if frame is None:
        return None
    if isinstance(frame, pl.DataFrame):
        df = frame
    elif isinstance(frame, pd.DataFrame):
        if frame.empty:
            return None
        # Gateways mix empty strings and numbers in the same object column.
        # Normalize only declared fields before Arrow's stricter conversion.
        types = {name: dtype for name, dtype, _ in spec.fields}
        types.update(ts_code="string", **{spec.date_col: "string"})
        normalized = frame.loc[:, [c for c in frame.columns if c in types]].copy()
        for column in normalized.columns:
            normalized[column] = (
                pd.to_numeric(normalized[column], errors="coerce")
                if types[column] in {"float", "int"}
                else normalized[column].astype("string")
            )
        df = pl.from_pandas(normalized, include_index=False)
    else:
        try:
            df = pl.DataFrame(frame)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{spec.api} 返回格式无法解析：{type(frame).__name__}") from exc
    return _normalize_frame(df, spec)


def _request_fields(spec: TushareExtSpec) -> list[str]:
    fields = []
    for name, _, _ in spec.fields:
        fields.append("ts_code" if name == "symbol" else name)
    if spec.date_col not in fields:
        fields.append(spec.date_col)
    if "ts_code" not in fields and "symbol" in {name for name, _, _ in spec.fields}:
        fields.append("ts_code")
    return list(dict.fromkeys(fields))


def _normalize_frame(df: pl.DataFrame, spec: TushareExtSpec) -> pl.DataFrame:
    if df.is_empty():
        return df
    fields = {name for name, _, _ in spec.fields}
    if "symbol" in fields and "symbol" not in df.columns and "ts_code" in df.columns:
        df = df.with_columns(pl.col("ts_code").map_elements(_compact_symbol, return_dtype=pl.Utf8).alias("symbol"))
    for column in (spec.date_col, "trade_date", "ann_date", "end_date"):
        if column in df.columns:
            df = df.with_columns(pl.col(column).map_elements(_normalize_date, return_dtype=pl.Utf8).alias(column))
    keep = [name for name, _, _ in spec.fields if name in df.columns]
    if not keep:
        return pl.DataFrame()
    df = df.select(keep)
    exprs = []
    for name, dtype, _ in spec.fields:
        if name not in df.columns:
            continue
        if dtype == "float":
            exprs.append(pl.col(name).cast(pl.Float64, strict=False).alias(name))
        elif dtype == "int":
            exprs.append(pl.col(name).cast(pl.Int64, strict=False).alias(name))
        else:
            exprs.append(pl.col(name).cast(pl.Utf8, strict=False).alias(name))
    return df.with_columns(exprs)


def _write_timeseries_frame(data_dir: Path, spec: TushareExtSpec, df: pl.DataFrame) -> int:
    if df.is_empty() or spec.date_col not in df.columns:
        return 0
    _upsert_config(data_dir, spec)
    rows = 0
    for day_value, part in df.partition_by(spec.date_col, as_dict=True).items():
        day_text = day_value[0] if isinstance(day_value, tuple) else day_value
        try:
            day = date.fromisoformat(str(day_text))
        except ValueError:
            continue
        part = _normalize_frame(part, spec)
        if part.is_empty():
            continue
        out_dir = data_dir / "ext_data" / spec.config_id / "timeseries" / f"date={day.isoformat()}"
        out_path = out_dir / "part.parquet"
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            existing = pl.read_parquet(out_path)
            part = pl.concat([existing, part], how="diagonal_relaxed")
        subset = [field for field in spec.unique_fields if field in part.columns]
        if subset:
            part = part.unique(subset=subset, keep="last")
        from app.parquet import atomic_write_parquet
        atomic_write_parquet(part, out_path)
        rows += part.height
    return rows


def _upsert_config(data_dir: Path, spec: TushareExtSpec) -> None:
    store = ExtConfigStore(data_dir)
    existing = store.get(spec.config_id)
    field_names = [name for name, _, _ in spec.fields]
    symbol_col = "symbol" if "symbol" in field_names else ("ts_code" if "ts_code" in field_names else field_names[0])
    config = ExtConfig(
        id=spec.config_id,
        label=spec.label,
        mode="timeseries",
        fields=[ExtField(name, dtype, label) for name, dtype, label in spec.fields],
        description=f"从 Tushare 自动增量维护: {spec.api}",
        symbol_map={"type": "mapped", "col": symbol_col},
        code_map={"type": "computed", "from": symbol_col, "method": "strip_exchange"},
        created_at=existing.created_at if existing else None,
    )
    store.upsert(config)


def _trade_dates(data_dir: Path, start: date, end: date) -> list[date]:
    root = data_dir / "kline_daily"
    dates: list[date] = []
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir() or not child.name.startswith("date="):
                continue
            try:
                day = date.fromisoformat(child.name[5:])
            except ValueError:
                continue
            if start <= day <= end:
                dates.append(day)
    return sorted(set(dates))


def _merge_ranges(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    valid = sorted((start, end) for start, end in ranges if start <= end)
    if not valid:
        return []
    merged: list[tuple[date, date]] = []
    for start, end in valid:
        if not merged or start > merged[-1][1] + timedelta(days=1):
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _range_label(ranges: list[tuple[date, date]]) -> str:
    if not ranges:
        return "无需拉取"
    if len(ranges) == 1:
        return f"{ranges[0][0]} 至 {ranges[0][1]}"
    return f"{len(ranges)} 段，{ranges[0][0]} 至 {ranges[-1][1]}"


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


def _compact_symbol(value: object) -> str:
    text = str(value or "").strip().upper()
    match = re.search(r"(\d{6})", text)
    return match.group(1) if match else text


def _normalize_date(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) >= 10 and text[4] == "-":
        return text[:10]
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return text


def _yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")


def _noop(stage: str, pct: int, msg: str) -> None:
    return None
