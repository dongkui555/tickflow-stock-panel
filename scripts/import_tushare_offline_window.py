"""Import useful slices from offline Tushare per-API tar packages.

The offline package stores one ``<api>.tar.gz`` per Tushare API. This importer
streams parquet members directly from tar, filters the requested date window,
and writes only the data TickFlow currently needs. It deliberately avoids
extracting whole archives because local disk may be tight.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import polars as pl


@dataclass(frozen=True)
class ExtSpec:
    api: str
    config_id: str
    label: str
    date_col: str
    fields: tuple[tuple[str, str, str], ...]
    unique_fields: tuple[str, ...] = ()


EXT_SPECS: dict[str, ExtSpec] = {
    "stk_limit": ExtSpec(
        api="stk_limit",
        config_id="ext_tushare_stk_limit",
        label="Tushare 涨跌停价",
        date_col="trade_date",
        fields=(
            ("symbol", "string", "标的代码"),
            ("trade_date", "string", "交易日期"),
            ("up_limit", "float", "涨停价"),
            ("down_limit", "float", "跌停价"),
        ),
        unique_fields=("symbol", "trade_date"),
    ),
    "limit_list_d": ExtSpec(
        api="limit_list_d",
        config_id="ext_tushare_limit_list_d",
        label="Tushare 涨跌停明细",
        date_col="trade_date",
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
    "top_list": ExtSpec(
        api="top_list",
        config_id="ext_tushare_top_list",
        label="Tushare 龙虎榜",
        date_col="trade_date",
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
    "top_inst": ExtSpec(
        api="top_inst",
        config_id="ext_tushare_top_inst",
        label="Tushare 龙虎榜营业部",
        date_col="trade_date",
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
    "moneyflow_dc": ExtSpec(
        api="moneyflow_dc",
        config_id="ext_tushare_moneyflow_dc",
        label="Tushare 东方财富资金流",
        date_col="trade_date",
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
    "moneyflow_ths": ExtSpec(
        api="moneyflow_ths",
        config_id="ext_tushare_moneyflow_ths",
        label="Tushare 同花顺资金流",
        date_col="trade_date",
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
    "dc_daily": ExtSpec(
        api="dc_daily",
        config_id="ext_tushare_dc_daily",
        label="Tushare 东方财富板块日线",
        date_col="trade_date",
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
    "ths_daily": ExtSpec(
        api="ths_daily",
        config_id="ext_tushare_ths_daily",
        label="Tushare 同花顺板块日线",
        date_col="trade_date",
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
    "sw_daily": ExtSpec(
        api="sw_daily",
        config_id="ext_tushare_sw_daily",
        label="Tushare 申万行业日线",
        date_col="trade_date",
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
    "cyq_perf": ExtSpec(
        api="cyq_perf",
        config_id="ext_tushare_cyq_perf",
        label="Tushare 筹码成本分布摘要",
        date_col="trade_date",
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
    "forecast": ExtSpec(
        api="forecast",
        config_id="ext_tushare_forecast",
        label="Tushare 业绩预告",
        date_col="ann_date",
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
    "express": ExtSpec(
        api="express",
        config_id="ext_tushare_express",
        label="Tushare 业绩快报",
        date_col="ann_date",
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
    "block_trade": ExtSpec(
        api="block_trade",
        config_id="ext_tushare_block_trade",
        label="Tushare 大宗交易",
        date_col="trade_date",
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
    "stk_holdernumber": ExtSpec(
        api="stk_holdernumber",
        config_id="ext_tushare_holdernumber",
        label="Tushare 股东户数",
        date_col="ann_date",
        fields=(
            ("symbol", "string", "标的代码"),
            ("ann_date", "string", "公告日期"),
            ("end_date", "string", "截止日期"),
            ("holder_num", "int", "股东户数"),
        ),
        unique_fields=("symbol", "ann_date", "end_date"),
    ),
    "margin": ExtSpec(
        api="margin",
        config_id="ext_tushare_margin",
        label="Tushare 融资融券汇总",
        date_col="trade_date",
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
    "hsgt_top10": ExtSpec(
        api="hsgt_top10",
        config_id="ext_tushare_hsgt_top10",
        label="Tushare 沪深股通十大成交股",
        date_col="trade_date",
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
    "ggt_top10": ExtSpec(
        api="ggt_top10",
        config_id="ext_tushare_ggt_top10",
        label="Tushare 港股通十大成交股",
        date_col="trade_date",
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


def parse_yyyymmdd(value: str) -> date:
    value = value.strip()
    if "-" in value:
        return date.fromisoformat(value)
    return datetime.strptime(value, "%Y%m%d").date()


def compact_symbol(value: object) -> str:
    text = str(value or "").strip().upper()
    match = re.search(r"(\d{6})", text)
    if match:
        return match.group(1)
    return text


def normalize_date_str(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) >= 10 and text[4] == "-":
        return text[:10]
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return text


def date_from_row_value(value: object) -> date | None:
    text = normalize_date_str(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def member_may_intersect(name: str, start: date, end: date) -> bool:
    dates = [parse_yyyymmdd(x) for x in re.findall(r"(?:trade_date|date)=(\d{8})", name)]
    if dates:
        return any(start <= d <= end for d in dates)
    range_match = re.search(r"start_date=(\d{8})_end_date=(\d{8})", name)
    if range_match:
        left = parse_yyyymmdd(range_match.group(1))
        right = parse_yyyymmdd(range_match.group(2))
        return left <= end and right >= start
    return True


def api_tar_path(source_root: Path, api: str) -> Path | None:
    direct = source_root / api / f"{api}.tar.gz"
    if direct.exists():
        return direct
    return None


def read_member_parquet(tf: tarfile.TarFile, member: tarfile.TarInfo) -> pl.DataFrame:
    fp = tf.extractfile(member)
    if fp is None:
        return pl.DataFrame()
    return pl.read_parquet(io.BytesIO(fp.read()))


def iter_parquet_members(tar_path: Path) -> Iterable[tarfile.TarInfo]:
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if member.isfile() and member.name.endswith(".parquet"):
                yield member


def write_config(config_dir: Path, spec: ExtSpec) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    field_names = [name for name, _, _ in spec.fields]
    symbol_col = "symbol" if "symbol" in field_names else ("ts_code" if "ts_code" in field_names else field_names[0])
    payload = {
        "id": spec.config_id,
        "label": spec.label,
        "mode": "timeseries",
        "fields": [
            {"name": name, "dtype": dtype, "label": label}
            for name, dtype, label in spec.fields
        ],
        "description": f"从 Tushare 离线历史包按窗口导入: {spec.api}",
        "symbol_map": {"type": "mapped", "col": symbol_col},
        "code_map": {"type": "computed", "from": symbol_col, "method": "strip_exchange"},
        "created_at": now,
        "updated_at": now,
    }
    (config_dir / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def cast_ext_frame(df: pl.DataFrame, spec: ExtSpec) -> pl.DataFrame:
    if df.is_empty():
        return df
    if "ts_code" in df.columns and "symbol" in {name for name, _, _ in spec.fields}:
        df = df.with_columns(pl.col("ts_code").map_elements(compact_symbol, return_dtype=pl.Utf8).alias("symbol"))
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


def write_timeseries_partition(data_dir: Path, spec: ExtSpec, day: date, df: pl.DataFrame, dry_run: bool) -> int:
    df = cast_ext_frame(df, spec)
    if df.is_empty():
        return 0
    out_dir = data_dir / "ext_data" / spec.config_id / "timeseries" / f"date={day.isoformat()}"
    out_path = out_dir / "part.parquet"
    if not dry_run:
        write_config(data_dir / "ext_data" / spec.config_id, spec)
        out_dir.mkdir(parents=True, exist_ok=True)
        subset = [field for field in spec.unique_fields if field in df.columns]
        if not subset:
            subset = ["symbol"] if "symbol" in df.columns else [df.columns[0]]
        if out_path.exists():
            existing = pl.read_parquet(out_path)
            df = pl.concat([existing, df], how="diagonal_relaxed").unique(subset=subset, keep="last")
        df.write_parquet(out_path, compression="zstd")
    return df.height


def import_ext_api(source_root: Path, data_dir: Path, spec: ExtSpec, start: date, end: date, dry_run: bool) -> dict:
    tar_path = api_tar_path(source_root, spec.api)
    if tar_path is None:
        return {"api": spec.api, "status": "missing", "rows": 0, "files": 0}
    if tar_path.stat().st_size < 1024:
        return {"api": spec.api, "status": "empty_archive", "rows": 0, "files": 0}

    rows = 0
    files = 0
    days: set[str] = set()
    date_groups: dict[str, list[pl.DataFrame]] = defaultdict(list)
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".parquet")):
                continue
            if not member_may_intersect(member.name, start, end):
                continue
            raw = read_member_parquet(tf, member)
            if raw.is_empty() or spec.date_col not in raw.columns:
                continue
            raw = raw.with_columns(
                pl.col(spec.date_col)
                .map_elements(normalize_date_str, return_dtype=pl.Utf8)
                .alias(spec.date_col)
            )
            raw = raw.filter(
                (pl.col(spec.date_col) >= start.isoformat())
                & (pl.col(spec.date_col) <= end.isoformat())
            )
            if raw.is_empty():
                continue
            for day_text, part in raw.partition_by(spec.date_col, as_dict=True).items():
                key = day_text[0] if isinstance(day_text, tuple) else day_text
                try:
                    day = date.fromisoformat(str(key))
                except ValueError:
                    continue
                date_groups[day.isoformat()].append(part)

    for day_text, parts in sorted(date_groups.items()):
        try:
            day = date.fromisoformat(day_text)
        except ValueError:
            continue
        frame = pl.concat(parts, how="diagonal_relaxed") if len(parts) > 1 else parts[0]
        written = write_timeseries_partition(data_dir, spec, day, frame, dry_run)
        if written:
            rows += written
            files += 1
            days.add(day.isoformat())
    return {
        "api": spec.api,
        "status": "ok",
        "rows": rows,
        "files": files,
        "date_range": [min(days), max(days)] if days else None,
        "dry_run": dry_run,
    }


def merge_financial(existing: pl.DataFrame, incoming: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    frames = [df for df in (existing, incoming) if df is not None and not df.is_empty()]
    if not frames:
        return pl.DataFrame()
    merged = pl.concat(frames, how="diagonal_relaxed")
    if "period_end" in merged.columns:
        merged = merged.with_columns(pl.col("period_end").cast(pl.Date, strict=False))
        merged = merged.filter((pl.col("period_end") >= start) & (pl.col("period_end") <= end))
    if "announce_date" in merged.columns:
        merged = merged.with_columns(pl.col("announce_date").cast(pl.Date, strict=False))
    unique_cols = [c for c in ("symbol", "period_end") if c in merged.columns]
    if unique_cols:
        sort_cols = unique_cols + (["announce_date"] if "announce_date" in merged.columns else [])
        merged = merged.sort(sort_cols, nulls_last=True).unique(subset=unique_cols, keep="last")
    return merged.sort(unique_cols) if unique_cols else merged


def import_financial_api(source_root: Path, data_dir: Path, api: str, table: str, start: date, end: date, dry_run: bool) -> dict:
    tar_path = api_tar_path(source_root, api)
    if tar_path is None:
        return {"api": api, "table": table, "status": "missing", "rows": 0}
    if tar_path.stat().st_size < 1024:
        return {"api": api, "table": table, "status": "empty_archive", "rows": 0}

    from app.plugins.tushare.provider import TushareProvider

    provider = TushareProvider()
    frames: list[pl.DataFrame] = []
    read_files = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".parquet")):
                continue
            raw = read_member_parquet(tf, member)
            if raw.is_empty() or "ts_code" not in raw.columns:
                continue
            period_source = "end_date" if "end_date" in raw.columns else "trade_date" if "trade_date" in raw.columns else None
            if not period_source:
                continue
            raw = raw.with_columns(
                pl.col(period_source)
                .map_elements(normalize_date_str, return_dtype=pl.Utf8)
                .alias(period_source)
            )
            raw = raw.filter(
                (pl.col(period_source) >= start.isoformat())
                & (pl.col(period_source) <= end.isoformat())
            )
            if raw.is_empty():
                continue
            symbol = str(raw["ts_code"][0])
            normalized = provider._normalize_financial_frame(raw.to_pandas(), table, symbol)  # noqa: SLF001
            if not normalized.is_empty():
                frames.append(normalized)
                read_files += 1
    incoming = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    out_file = data_dir / "financials" / table / "part.parquet"
    existing = pl.read_parquet(out_file) if out_file.exists() else pl.DataFrame()
    merged = merge_financial(existing, incoming, start, end)
    if not dry_run and not merged.is_empty():
        out_file.parent.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(out_file, compression="zstd")
    return {
        "api": api,
        "table": table,
        "status": "ok",
        "source_files": read_files,
        "incoming_rows": incoming.height,
        "rows": merged.height,
        "date_range": (
            [
                str(merged["period_end"].min()),
                str(merged["period_end"].max()),
            ]
            if not merged.is_empty() and "period_end" in merged.columns
            else None
        ),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Import selected Tushare offline data within a date window.")
    parser.add_argument("--source-root", type=Path, default=Path("/app/offline-data/tushare_15000_history_by_api_packages_20260627"))
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--start-date", default="2024-09-01")
    parser.add_argument("--end-date", default="2026-09-03")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--apis",
        default=(
            "fina_indicator,income,balancesheet,stk_limit,limit_list_d,top_list,top_inst,"
            "moneyflow_dc,moneyflow_ths,dc_daily,ths_daily,sw_daily,cyq_perf,forecast,"
            "express,block_trade,stk_holdernumber,margin,hsgt_top10,ggt_top10"
        ),
        help="Comma-separated APIs to import.",
    )
    args = parser.parse_args()

    start = parse_yyyymmdd(args.start_date)
    end = parse_yyyymmdd(args.end_date)
    wanted = [item.strip() for item in args.apis.split(",") if item.strip()]
    results: list[dict] = []

    for api in wanted:
        if api == "fina_indicator":
            results.append(import_financial_api(args.source_root, args.data_dir, "fina_indicator", "metrics", start, end, args.dry_run))
        elif api == "income":
            results.append(import_financial_api(args.source_root, args.data_dir, "income", "income", start, end, args.dry_run))
        elif api == "balancesheet":
            results.append(import_financial_api(args.source_root, args.data_dir, "balancesheet", "balance_sheet", start, end, args.dry_run))
        elif api in EXT_SPECS:
            results.append(import_ext_api(args.source_root, args.data_dir, EXT_SPECS[api], start, end, args.dry_run))
        else:
            results.append({"api": api, "status": "unsupported"})

    print(json.dumps({"start": start.isoformat(), "end": end.isoformat(), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
