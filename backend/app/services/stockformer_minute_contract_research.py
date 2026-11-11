"""Independent, information-preserving minute research. Never writes raw data.

Buckets describe stored wall-clock labels, not an inferred tick-time mapping.
A 09:30 row joins the first bucket; its turnover and OHLC are retained. Daily
conservation is necessary evidence, not proof of historical vendor provenance.
"""

from __future__ import annotations

from datetime import datetime, time

import polars as pl

from app.indicators.pipeline import _normalize_raw_volume_unit
from app.services.minute_repair import canonical_minute_aliases, daily_reference, keyed
from app.services.minute_timeframes import aggregate_minutes

GROUPS = {
    "opening": ["m5_early_return", "m5_early_amount_share", "m5_early_volume_share"],
    "tail": ["m5_tail_return", "m5_tail_amount_share"],
    "location": ["m5_close_vwap_gap", "m5_vwap_open_gap", "m5_close_position"],
    "path": ["m5_log_return_std", "m5_intraday_return", "m5_range"],
}
CONTRACT = dict(
    version="preserve_opening_stamped_5m_research_v1",
    period=5,
    opening="retain_all_0930_ohlcv_amount_in_first_bucket_no_raw_timestamp_shift",
    axis="240_unique_end_labels_plus_optional_0930_no_other_clocks",
    unit_evidence="unique_1_or_100_scale_to_existing_normalized_daily_shares_and_vwap_bounds",
    daily_turnover_tolerance=0.002,
    price_tolerance_yuan=0.011,
    early="first_6_stamped_buckets_through_1000",
    tail="last_6_buckets_after_1430",
    volatility="sample_std_47_log_close_changes_including_lunch_gap",
    metadata_not_predictors=True,
    groups=GROUPS,
    exact_vendor_interval_mapping_proven=False,
    production_eligible=False,
)
AXIS = {*range(571, 691), *range(781, 901)}


def build_factors(frame: pl.DataFrame, daily: pl.DataFrame, day, *, data_dir=None):
    """Return accepted research factors and rejection reasons, one date at a time."""
    if frame.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    f = frame
    if f.schema["datetime"].time_zone:
        f = f.with_columns(
            pl.col("datetime").dt.convert_time_zone("Asia/Shanghai").dt.replace_time_zone(None)
        )
    f = keyed(canonical_minute_aliases(f, day)).sort("_code", "datetime")
    refs = daily_reference(_normalize_raw_volume_unit(daily), data_dir=data_dir)
    if not refs:
        q = (
            f.select("_code")
            .unique()
            .with_columns(pl.lit("daily_reference_missing").alias("status"))
        )
        return pl.DataFrame(), q
    ref = pl.DataFrame(
        [
            dict(
                _code=k,
                **{c: float(v[c]) for c in ["open", "high", "low", "close", "volume", "amount"]},
            )
            for k, v in refs.items()
        ]
    )
    clock = pl.col("datetime").dt.hour().cast(pl.Int32) * 60 + pl.col("datetime").dt.minute().cast(
        pl.Int32
    )
    valid = pl.all_horizontal(
        [pl.col(c).is_finite() & (pl.col(c) > 0) for c in ["open", "high", "low", "close"]]
    )
    valid &= (pl.col("high") >= pl.max_horizontal("open", "high", "low", "close")) & (
        pl.col("low") <= pl.min_horizontal("open", "high", "low", "close")
    )
    valid &= pl.all_horizontal(
        [pl.col(c).is_finite() & (pl.col(c) >= 0) for c in ["volume", "amount"]]
    )
    f = f.with_columns(clock.alias("_clock"))
    q = (
        f.group_by("_code")
        .agg(
            pl.len().alias("_rows"),
            pl.col("datetime").n_unique().alias("_unique"),
            pl.col("datetime").filter(pl.col("_clock").is_in(AXIS)).n_unique().alias("_core"),
            (pl.col("_clock") == 570).sum().alias("_opening_rows"),
            (
                (pl.col("datetime").dt.date() == day)
                & pl.col("_clock").is_in(AXIS | {570})
                & (pl.col("datetime").dt.second() == 0)
                & (pl.col("datetime").dt.microsecond() == 0)
            )
            .all()
            .alias("_axis"),
            valid.fill_null(False).all().alias("_valid"),
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("amount").sum(),
        )
        .join(ref, on="_code", how="left", suffix="_daily", validate="1:1")
    )
    q = q.with_columns(
        *[
            (
                (pl.col("volume") * scale - pl.col("volume_daily")).abs()
                <= pl.max_horizontal(pl.lit(100.0), pl.col("volume_daily") * 0.002)
            )
            .fill_null(False)
            .alias(f"_unit_{scale}")
            for scale in [1, 100]
        ]
    )
    q = q.with_columns(
        pl.when(pl.col("_unit_1") & ~pl.col("_unit_100"))
        .then(1.0)
        .when(pl.col("_unit_100") & ~pl.col("_unit_1"))
        .then(100.0)
        .otherwise(None)
        .alias("_unit_scale")
    )
    price_ok = pl.all_horizontal(
        [
            (pl.col(c) - pl.col(c + "_daily")).abs()
            <= pl.max_horizontal(pl.lit(0.011), pl.col(c + "_daily").abs() * 1e-6)
            for c in ["open", "high", "low", "close"]
        ]
    )
    vwap = pl.col("amount") / (pl.col("volume") * pl.col("_unit_scale"))
    tolerance = pl.max_horizontal(pl.lit(0.02), pl.col("high_daily") * 0.002)
    q = q.with_columns(
        pl.when(pl.col("volume_daily").is_null())
        .then(pl.lit("daily_reference_missing"))
        .when(~pl.col("_axis") | (pl.col("_core") != 240) | (pl.col("_rows") != pl.col("_unique")))
        .then(pl.lit("axis_or_duplicate"))
        .when(~pl.col("_valid"))
        .then(pl.lit("invalid_bar"))
        .when(pl.col("_unit_scale").is_null())
        .then(pl.lit("unit_or_volume_conflict"))
        .when(~price_ok.fill_null(False))
        .then(pl.lit("daily_price_conflict"))
        .when(
            (pl.col("amount") - pl.col("amount_daily")).abs()
            > pl.max_horizontal(pl.lit(1.0), pl.col("amount_daily") * 0.002)
        )
        .then(pl.lit("daily_amount_conflict"))
        .when(
            ~vwap.is_between(
                pl.col("low_daily") - tolerance, pl.col("high_daily") + tolerance
            ).fill_null(False)
        )
        .then(pl.lit("vwap_unit_conflict"))
        .otherwise(pl.lit("accepted"))
        .alias("status")
    )
    good = q.filter(pl.col("status") == "accepted").select("_code", "_unit_scale")
    if good.is_empty():
        return pl.DataFrame(), q
    f = f.join(good, on="_code", how="inner", validate="m:1").with_columns(
        pl.col("_code").alias("symbol"), (pl.col("volume") * pl.col("_unit_scale")).alias("volume")
    )
    bars = aggregate_minutes(f, 5, datetime.combine(day, time(15)))
    assert bars["complete"].all()
    bclock = pl.col("datetime").dt.hour().cast(pl.Int32) * 60 + pl.col("datetime").dt.minute().cast(
        pl.Int32
    )
    bars = bars.with_columns(bclock.alias("_clock")).sort("symbol", "datetime")
    a = bars.group_by("symbol").agg(
        pl.len().alias("_buckets"),
        pl.col("open").first().alias("_open"),
        pl.col("high").max().alias("_high"),
        pl.col("low").min().alias("_low"),
        pl.col("close").last().alias("_close"),
        pl.col("close").filter(pl.col("_clock") == 600).first().alias("_early_close"),
        pl.col("close").filter(pl.col("_clock") == 870).first().alias("_tail_open"),
        pl.col("volume").sum().alias("_m5_volume_shares"),
        pl.col("amount").sum().alias("_m5_amount_yuan"),
        pl.col("volume").filter(pl.col("_clock") <= 600).sum().alias("_early_vol"),
        pl.col("amount").filter(pl.col("_clock") <= 600).sum().alias("_early_amt"),
        pl.col("amount").filter(pl.col("_clock") > 870).sum().alias("_tail_amt"),
        pl.col("close").log().diff().std(ddof=1).alias("m5_log_return_std"),
    )
    assert (a["_buckets"] == 48).all()
    a = a.with_columns((pl.col("_m5_amount_yuan") / pl.col("_m5_volume_shares")).alias("_vwap"))
    a = a.with_columns(
        (pl.col("_early_close") / pl.col("_open") - 1).alias("m5_early_return"),
        (pl.col("_early_amt") / pl.col("_m5_amount_yuan")).alias("m5_early_amount_share"),
        (pl.col("_early_vol") / pl.col("_m5_volume_shares")).alias("m5_early_volume_share"),
        (pl.col("_close") / pl.col("_tail_open") - 1).alias("m5_tail_return"),
        (pl.col("_tail_amt") / pl.col("_m5_amount_yuan")).alias("m5_tail_amount_share"),
        (pl.col("_close") / pl.col("_vwap") - 1).alias("m5_close_vwap_gap"),
        (pl.col("_vwap") / pl.col("_open") - 1).alias("m5_vwap_open_gap"),
        pl.when(pl.col("_high") > pl.col("_low"))
        .then((pl.col("_close") - pl.col("_low")) / (pl.col("_high") - pl.col("_low")))
        .otherwise(0.5)
        .alias("m5_close_position"),
        (pl.col("_close") / pl.col("_open") - 1).alias("m5_intraday_return"),
        ((pl.col("_high") - pl.col("_low")) / pl.col("_open")).alias("m5_range"),
        pl.lit(day).alias("date"),
    )
    columns = [column for group in GROUPS.values() for column in group]
    assert a.select(pl.all_horizontal([pl.col(c).is_finite() for c in columns]).all()).item()
    return a.select("symbol", "date", *columns, "_m5_volume_shares", "_m5_amount_yuan"), q
