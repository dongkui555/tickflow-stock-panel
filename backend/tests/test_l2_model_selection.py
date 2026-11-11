from copy import deepcopy
from datetime import date, timedelta

import polars as pl

from app.services import l2_model_selection as selection


def _candidate_report(test_value=999.0):
    splits = {}
    for split, gain in (("train", 0.01), ("validation", 0.02), ("test", test_value)):
        splits[split] = {
            "baseline": {
                "rank_ic": 0.01,
                "top_k_after_cost_proxy": 0.001,
            },
            "baseline_plus_l2": {
                "rank_ic": 0.01 + gain,
                "top_k_after_cost_proxy": 0.002 + gain,
            },
            "plus_l2_minus_baseline": {
                "rank_ic": gain,
                "top_k_forward_return": gain,
            },
        }
    return {"horizons": {str(h): {"splits": deepcopy(splits)} for h in (5, 10)}}


def test_candidate_summary_cannot_read_test_split():
    first = selection.summarize_candidate(_candidate_report(test_value=1.0))
    second = selection.summarize_candidate(_candidate_report(test_value=-1.0))
    assert first == second
    assert first["stable"] is True


def test_training_gain_that_reverses_in_validation_is_rejected():
    report = _candidate_report()
    report["horizons"]["5"]["splits"]["validation"]["plus_l2_minus_baseline"]["rank_ic"] = -0.02
    report["horizons"]["10"]["splits"]["validation"]["plus_l2_minus_baseline"]["rank_ic"] = -0.02
    summary = selection.summarize_candidate(report)
    assert summary["stable"] is False
    assert summary["train_positive_validation_negative"]


def test_candidate_selection_falls_back_when_no_stable_l2():
    assert (
        selection.select_candidate({"bad": {"stable": False, "selection_objective": 1.0}})
        == "baseline_only"
    )
    assert (
        selection.select_candidate(
            {
                "weak": {"stable": True, "selection_objective": 0.1},
                "better": {"stable": True, "selection_objective": 0.2},
            }
        )
        == "better"
    )


def test_archive_readiness_requires_sixty_continuous_market_days(tmp_path):
    start = date(2026, 1, 1)
    market = [start + timedelta(days=i) for i in range(70)]
    for day in market[:59]:
        (tmp_path / f"{day:%Y%m%d}.7z").write_bytes(b"x")
    report = selection.audit_2026_archives(tmp_path, market, start=market[0], end=market[-1])
    assert report["ready_for_fresh_oos"] is False
    assert report["longest_continuous_market_day_streak"] == 59
    (tmp_path / f"{market[59]:%Y%m%d}.7z").write_bytes(b"x")
    report = selection.audit_2026_archives(tmp_path, market, start=market[0], end=market[-1])
    assert report["ready_for_fresh_oos"] is True
    assert report["longest_continuous_market_day_streak"] == 60


def test_market_regime_uses_only_current_and_past_closes():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(30)]
    rows = []
    for symbol in ("000001", "000002"):
        for i, day in enumerate(days):
            price = 10 + i * 0.1
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "open": price,
                    "high": price + 0.1,
                    "low": price - 0.1,
                    "close": price,
                    "volume": 100.0,
                    "amount": 1000.0,
                }
            )
    panel = pl.DataFrame(rows)
    first = selection.build_market_regimes(panel, ["000001", "000002"], start=days[0], end=days[-1])
    changed = panel.with_columns(
        pl.when(pl.col("date") == days[-1])
        .then(pl.col("close") * 10)
        .otherwise(pl.col("close"))
        .alias("close")
    ).with_columns(pl.max_horizontal("high", "close").alias("high"))
    second = selection.build_market_regimes(
        changed, ["000001", "000002"], start=days[0], end=days[-1]
    )
    assert first.filter(pl.col("date") < days[-1]).equals(second.filter(pl.col("date") < days[-1]))
