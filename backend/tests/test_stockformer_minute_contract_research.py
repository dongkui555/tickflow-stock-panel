from datetime import date, datetime, timedelta

import polars as pl
import pytest

from app.services.stockformer_minute_contract_research import GROUPS, build_factors


def sample(opening=True, unit=100):
    day = date(2025, 11, 3)
    stamps = [
        datetime(2025, 11, 3, h, m) + timedelta(minutes=i)
        for h, m in [(9, 30), (13, 0)]
        for i in range(1, 121)
    ]
    if opening:
        stamps = [datetime(2025, 11, 3, 9, 30), *stamps]
    frame = pl.DataFrame(
        dict(
            symbol=["000001.SZ"] * len(stamps),
            datetime=stamps,
            open=[10.0] * len(stamps),
            high=[10.0] * len(stamps),
            low=[10.0] * len(stamps),
            close=[10.0] * len(stamps),
            volume=[100.0 / unit] * len(stamps),
            amount=[1000.0] * len(stamps),
        )
    )
    daily = pl.DataFrame(
        dict(
            symbol=["000001.SZ"],
            date=[day],
            open=[10.0],
            high=[10.0],
            low=[10.0],
            close=[10.0],
            volume=[100.0 * len(stamps)],
            amount=[1000.0 * len(stamps)],
        )
    )
    return frame, daily, day


def test_opening_split_and_embedded_representations_preserve_factors():
    frame, daily, day = sample()
    first = frame.filter(pl.col("datetime").dt.time() == datetime(2025, 11, 3, 9, 31).time())
    merged = pl.concat(
        [first.with_columns(pl.col("volume") * 2, pl.col("amount") * 2), frame.tail(239)]
    ).sort("datetime")
    a, qa = build_factors(frame, daily, day)
    b, qb = build_factors(merged, daily, day)
    assert a.height == b.height == 1
    assert qa["status"].to_list() == qb["status"].to_list() == ["accepted"]
    for col in [column for group in GROUPS.values() for column in group]:
        assert a[col][0] == pytest.approx(b[col][0], abs=1e-12)
    assert a["_m5_volume_shares"][0] == 24100
    assert a["_m5_amount_yuan"][0] == 241000
    assert qa["_opening_rows"][0] == 1 and qb["_opening_rows"][0] == 0


@pytest.mark.parametrize("unit", [1, 100])
def test_unique_unit_evidence_requires_price_and_daily_conservation(unit):
    frame, daily, day = sample(unit=unit)
    a, q = build_factors(frame, daily, day)
    assert a.height == 1 and q["_unit_scale"][0] == unit
    assert a["_m5_volume_shares"][0] == 24100


@pytest.mark.parametrize("fault", ["duplicate", "missing", "lunch", "wrong_amount", "wrong_close"])
def test_faults_fail_closed_without_modifying_input(fault):
    frame, daily, day = sample()
    if fault == "duplicate":
        frame = pl.concat([frame, frame.head(1)])
    if fault == "missing":
        frame = frame.head(240)
    if fault == "lunch":
        frame = pl.concat(
            [frame, frame.head(1).with_columns(pl.lit(datetime(2025, 11, 3, 13)).alias("datetime"))]
        )
    if fault == "wrong_amount":
        daily = daily.with_columns(pl.col("amount") * 2)
    if fault == "wrong_close":
        daily = daily.with_columns(pl.lit(11.0).alias("close"))
    before = frame.clone()
    a, q = build_factors(frame, daily, day)
    assert a.is_empty() and not (q["status"] == "accepted").any()
    assert frame.equals(before)


def test_lots_daily_compatibility_and_timezone():
    frame, daily, day = sample()
    daily = daily.with_columns(pl.col("volume") / 100)
    frame = frame.with_columns(
        pl.col("datetime").dt.replace_time_zone("Asia/Shanghai").dt.convert_time_zone("UTC")
    )
    a, q = build_factors(frame, daily, day)
    assert a.height == 1 and q["status"][0] == "accepted"
