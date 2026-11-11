"""Build daily features from locally stored minute bars.

Minute bars remain the raw source.  This module only creates a compact,
date-level feature table so StockFormer can use intraday behaviour without
loading a multi-year minute matrix into every training run.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

FEATURE_DEFAULTS = {
    "minute_session_coverage": 0.0,
    "minute_bar_count": 0.0,
    "minute_intraday_return": 0.0,
    "minute_early_return": 0.0,
    "minute_open_strength": 0.0,
    "minute_intraday_volatility": 0.0,
    "minute_early_amount_ratio": 0.0,
    "minute_late_amount_ratio": 0.0,
    "minute_volume_concentration": 0.0,
    "minute_vwap_gap": 0.0,
    "minute_close_position": 0.5,
}

# QMT timestamps are end-labelled in some exports (09:31 ... 15:00), while
# other sources use start-labelled bars (09:30 ... 14:59).  Match both legal
# axes so an extra 09:30 auction row cannot hide an internal regular-session
# gap.
EXPECTED_SESSION_BARS = 240
_END_LABELLED_MINUTES = (*range(571, 691), *range(781, 901))
_START_LABELLED_MINUTES = (*range(570, 690), *range(780, 900))


def _compact_stock_symbol(value: object) -> str:
    """Return the canonical six-digit stock key while rejecting bad market tags."""
    text = str(value or "").strip().upper()
    if not text:
        return ""
    exchange = ""
    if "." in text:
        code, suffix = text.split(".", 1)
        text, exchange = code, suffix
    else:
        for prefix in ("SH", "SZ", "BJ"):
            if text.startswith(prefix):
                text, exchange = text[len(prefix):], prefix
                break
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return ""
    code = digits[-6:].zfill(6)
    expected = (
        "SH" if code.startswith(("6", "68"))
        else "BJ" if code.startswith(("43", "83", "87", "88", "92"))
        else "SZ"
    )
    if exchange and exchange != expected:
        return ""
    return code


def _empty() -> pl.DataFrame:
    return pl.DataFrame({"symbol": [], "date": []}, schema={"symbol": pl.Utf8, "date": pl.Date})


def build_daily_features(
    data_dir: Path,
    symbols: list[str],
    start: date,
    end: date,
    *,
    on_progress: Callable[[int, int, date], None] | None = None,
) -> pl.DataFrame:
    """Reduce one partition at a time; retain only compact stock-day results."""
    root = Path(data_dir) / "kline_minute"
    if not symbols or not root.exists():
        return _empty()
    days = []
    for partition in root.glob("date=*"):
        try:
            day = date.fromisoformat(partition.name.split("=", 1)[1])
        except ValueError:
            continue
        if start <= day <= end:
            days.append(day)
    frames = []
    for completed, day in enumerate(sorted(days), 1):
        frame = _build_daily_partition(data_dir, symbols, day, day)
        if not frame.is_empty():
            frames.append(frame)
        # Cancellation must propagate, not be swallowed as missing data.
        if on_progress is not None:
            on_progress(completed, len(days), day)
    return pl.concat(frames, how="vertical_relaxed") if frames else _empty()


def _build_daily_partition(
    data_dir: Path,
    symbols: list[str],
    start: date,
    end: date,
) -> pl.DataFrame:
    """Aggregate valid regular-session minute bars into one row per symbol/day."""
    root = Path(data_dir) / "kline_minute"
    if not symbols or not root.exists():
        return _empty()
    wanted_symbols = {
        _compact_stock_symbol(symbol)
        for symbol in symbols
    }
    wanted_symbols.discard("")
    if not wanted_symbols:
        return _empty()
    raw_wanted_symbols: set[str] = set(wanted_symbols)
    for code in wanted_symbols:
        if code.startswith(("6", "68")):
            exchange = "SH"
        elif code.startswith(("43", "83", "87", "88", "92")):
            exchange = "BJ"
        else:
            exchange = "SZ"
        raw_wanted_symbols.update({f"{code}.{exchange}", f"{exchange}{code}"})
    try:
        required = {"symbol", "datetime", "open", "high", "low", "close", "volume", "amount"}
        partition = root / f"date={start.isoformat()}"
        files = [str(path) for path in partition.rglob("*.parquet")]
        if not files:
            return _empty()
        valid_files: list[str] = []
        for path in files:
            try:
                schema = pl.read_parquet(path, n_rows=0).schema
            except Exception:
                continue
            if required.issubset(set(schema)):
                valid_files.append(path)
        if not valid_files:
            return _empty()
        lf = (
            pl.scan_parquet(valid_files)
            .select(sorted(required))
            .filter(
                pl.col("symbol").cast(pl.Utf8).str.to_uppercase().is_in(
                    sorted(raw_wanted_symbols)
                )
            )
            .with_columns(
                pl.col("symbol")
                .map_elements(_compact_stock_symbol, return_dtype=pl.Utf8)
                .alias("_symbol_key"),
                pl.col("symbol").cast(pl.Utf8).str.to_uppercase()
                .str.contains(r"\.|^(SH|SZ|BJ)")
                .cast(pl.Int8).alias("_identity_rank"),
                pl.col("datetime").dt.date().alias("date"),
                (pl.col("datetime").dt.hour().cast(pl.Int32) * 60
                 + pl.col("datetime").dt.minute().cast(pl.Int32)).alias("_minute"),
            )
            .filter(
                pl.col("_symbol_key").is_in(sorted(wanted_symbols))
                & pl.col("date").is_between(start, end)
                & (pl.col("_minute") >= 570)
                & (pl.col("_minute") <= 900)
                & pl.all_horizontal(
                    [pl.col(c).cast(pl.Float64, strict=False).is_finite() for c in
                     ("open", "high", "low", "close", "volume", "amount")]
                )
                & (pl.col("open") > 0)
                & (pl.col("high") >= pl.col("low"))
                & (pl.col("close") > 0)
                & (pl.col("volume") >= 0)
                & (pl.col("amount") >= 0)
            )
            # Prefer an exchange-qualified row when a legacy bare-code alias
            # and a qualified row describe the same stock-minute. This keeps
            # aliases from doubling the bar count; conflicting qualified rows
            # remain visible to the upstream minute-quality audit.
            .sort(["_symbol_key", "date", "datetime", "_identity_rank"],
                  descending=[False, False, False, True])
            .unique(
                subset=["_symbol_key", "date", "datetime"],
                keep="first",
                maintain_order=True,
            )
        )
        early = (pl.col("_minute") < 600)  # 09:30-10:00
        late = (pl.col("_minute") >= 870)  # 14:30-15:00
        first_open = pl.col("open").sort_by("datetime").first()
        last_close = pl.col("close").sort_by("datetime").last()
        # Sort the whole group before filtering; filtering the value and sort
        # key independently causes a Polars group-length mismatch.
        early_last = pl.col("close").sort_by("datetime").filter(early).last()
        total_amount = pl.col("amount").sum()
        total_volume = pl.col("volume").sum()
        grouped = (
            lf.group_by(["_symbol_key", "date"])
            .agg(
                pl.len().alias("minute_bar_count"),
                pl.col("_minute").is_in(_END_LABELLED_MINUTES).cast(pl.Int32)
                .sum().alias("_end_labelled_bar_count"),
                pl.col("_minute").is_in(_START_LABELLED_MINUTES).cast(pl.Int32)
                .sum().alias("_start_labelled_bar_count"),
                pl.col("datetime").min().alias("_first_time"),
                pl.col("datetime").max().alias("_last_time"),
                first_open.alias("_first_open"),
                last_close.alias("_last_close"),
                early_last.alias("_early_last"),
                pl.col("high").max().alias("_high"),
                pl.col("low").min().alias("_low"),
                pl.col("close").std().alias("_close_std"),
                total_amount.alias("_total_amount"),
                total_volume.alias("_total_volume"),
                pl.col("amount").filter(early).sum().alias("_early_amount"),
                pl.col("amount").filter(late).sum().alias("_late_amount"),
                pl.col("volume").filter(early).sum().alias("_early_volume"),
            )
            .with_columns(
                # Use the best matching legal timestamp axis.  A first/last
                # timestamp span misses internal gaps and falsely reports a
                # sparse session as complete.
                (pl.max_horizontal(
                    "_end_labelled_bar_count", "_start_labelled_bar_count"
                ) / EXPECTED_SESSION_BARS)
                .clip(0.0, 1.0).alias("minute_session_coverage"),
                pl.col("minute_bar_count").cast(pl.Float64),
                ((pl.col("_last_close") / pl.col("_first_open")) - 1.0).alias("minute_intraday_return"),
                ((pl.col("_early_last") / pl.col("_first_open")) - 1.0).alias("minute_early_return"),
                ((pl.col("_early_last") / pl.col("_first_open")) - 1.0).alias("minute_open_strength"),
                (pl.col("_close_std") / pl.col("_first_open")).fill_null(0.0)
                .alias("minute_intraday_volatility"),
                (pl.col("_early_amount") / pl.col("_total_amount").replace(0, None))
                .fill_null(0.0).clip(0.0, 1.0).alias("minute_early_amount_ratio"),
                (pl.col("_late_amount") / pl.col("_total_amount").replace(0, None))
                .fill_null(0.0).clip(0.0, 1.0).alias("minute_late_amount_ratio"),
                (pl.col("_early_volume") / pl.col("_total_volume").replace(0, None))
                .fill_null(0.0).clip(0.0, 1.0).alias("minute_volume_concentration"),
                ((pl.col("_last_close") - pl.col("_low"))
                 / (pl.col("_high") - pl.col("_low")).replace(0, None))
                .fill_null(0.5).clip(0.0, 1.0).alias("minute_close_position"),
            )
            .with_columns(
                # Most Chinese feeds report volume in lots while amount is in
                # currency.  Detect that convention from the price range so
                # the VWAP feature remains comparable across providers.
                (pl.col("_total_amount") / pl.col("_total_volume").replace(0, None))
                .alias("_vwap_raw"),
            )
            .with_columns(
                pl.when(pl.col("_vwap_raw") > pl.col("_high") * 5.0)
                .then(pl.col("_vwap_raw") / 100.0)
                .otherwise(pl.col("_vwap_raw"))
                .truediv(pl.col("_first_open"))
                .sub(1.0).fill_null(0.0).alias("minute_vwap_gap"),
            )
            .select([
                pl.col("_symbol_key").alias("symbol"),
                "date",
                *FEATURE_DEFAULTS.keys(),
            ])
            .collect()
        )
        return grouped
    except Exception as exc:
        # Minute data is an enhancement. A broken or partially written minute
        # partition must not prevent daily model training from running.
        logger.warning("minute feature partition %s failed: %s", start, exc)
        return _empty()
