"""Offline tests for masks, causal states, calibration purges and rank invariance."""

import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from test_l2_experiment import sample as sample

from app.services import l2_diagnostics as diagnostic
from app.services import l2_experiment as experiment


@pytest.fixture
def settings():
    return diagnostic.DiagnosticConfig(min_market_stocks=3)


def test_groups_cover_all_features_without_overlap():
    grouped = [name for group in diagnostic.FEATURE_GROUPS.values() for name in group]
    assert len(grouped) == len(set(grouped)) == 13
    assert set(grouped) == set(experiment.L2_FEATURE_COLUMNS)
    variants = diagnostic.feature_variants()
    assert len(variants) == 14
    for name, columns in variants.items():
        assert len(columns) == len(set(columns))
        assert columns[:15] == experiment.BASELINE_FEATURE_COLUMNS
        if name.startswith("without_"):
            assert not set(columns) & set(diagnostic.FEATURE_GROUPS[name[8:]])


def test_all_arms_keep_complete_mask_and_reference(sample, settings):
    panel, l2, config, days = sample
    config = replace(config, horizons=(1,))
    l2 = l2.with_columns(
        pl.when((pl.col("symbol") == "000001.SZ") & (pl.col("date") == days[90]))
        .then(None)
        .otherwise(pl.col("l2_spread_bps_mean"))
        .alias("l2_spread_bps_mean")
    )
    result = diagnostic.evaluate_l2_diagnostics(panel, l2, config, settings)
    ref = result.reference.predictions.select("symbol", "date", "horizon", "split", "target")
    rows = result.predictions.filter(pl.col("calibration") == "raw")
    for arm in diagnostic.feature_variants():
        assert (
            rows.filter(pl.col("arm") == arm)
            .select(ref.columns)
            .sort(["horizon", "date", "symbol"])
            .equals(ref)
        )
    assert rows.filter((pl.col("symbol") == "000001") & (pl.col("date") == days[90])).is_empty()
    assert result.report["reference_arms_exact"]
    assert result.report["test_status"] == "reused_holdout_diagnostic_not_fresh_oos"
    assert result.report["promotion"] is False
    json.dumps(result.report, allow_nan=False)


def test_market_history_is_causal_and_order_invariant(sample, settings):
    panel, _, config, _ = sample
    original = diagnostic.build_market_history(panel, settings)
    changed = panel.with_columns(
        [
            pl.when(pl.col("date") >= config.test_start)
            .then(pl.col(c) * 5)
            .otherwise(pl.col(c))
            .alias(c)
            for c in ("open", "high", "low", "close")
        ]
    )
    new = diagnostic.build_market_history(changed.reverse(), settings)
    assert original.filter(pl.col("date") < config.test_start).equals(
        new.filter(pl.col("date") < config.test_start)
    )
    assert original.head(19)["market_state"].to_list() == ["unknown"] * 19


def test_market_gap_and_zero_trade_are_not_filled(sample):
    panel, _, _, days = sample
    settings = diagnostic.DiagnosticConfig(min_market_stocks=5)
    panel = panel.filter(~((pl.col("symbol") == "000001") & (pl.col("date") == days[50])))
    result = diagnostic.build_market_history(panel, settings)
    assert (
        result.filter(pl.col("date").is_between(days[50], days[69]))["market_state"].to_list()
        == ["unknown"] * 20
    )
    assert result.filter(pl.col("date") == days[70])["market_stocks"].item() == 5
    panel = panel.with_columns(
        pl.when(pl.col("date") == days[75]).then(0.0).otherwise(pl.col("volume")).alias("volume")
    )
    result = diagnostic.build_market_history(panel, settings)
    assert result.filter(pl.col("date").is_between(days[75], days[76]))[
        "market_stocks"
    ].to_list() == [0, 0]


def _validation(sample):
    panel, l2, config, _ = sample
    reference = experiment.evaluate_l2_experiment(panel, l2, replace(config, horizons=(1,)))
    rows = reference.predictions.filter(pl.col("split") == "validation").with_columns(
        pl.lit("weak").alias("market_state"), pl.lit(0.02).alias("prediction")
    )
    return rows, config


def test_calibration_purge_and_fallback(sample, settings):
    rows, config = _validation(sample)
    fitted = diagnostic.fit_calibration(rows, "prediction", config, settings)
    assert fitted["max_label_end"] < config.test_start
    assert fitted["states"]["strong"]["offset"] == fitted["global_offset"]
    assert not fitted["states"]["unknown"]["supported"]
    for mutation in (
        pl.lit("test").alias("split"),
        pl.lit(config.test_start).alias("label_end"),
        pl.lit(config.train_start).alias("date"),
    ):
        with pytest.raises(ValueError, match="purged validation"):
            diagnostic.fit_calibration(rows.with_columns(mutation), "prediction", config, settings)
    with pytest.raises(ValueError, match="finite"):
        diagnostic.fit_calibration(
            rows.with_columns(pl.lit(float("nan")).alias("prediction")),
            "prediction",
            config,
            settings,
        )
    small = rows.filter(pl.col("date").is_in(rows["date"].unique().sort().head(2).to_list()))
    disabled = diagnostic.fit_calibration(small, "prediction", config, settings)
    assert not disabled["enabled"] and disabled["global_offset"] == 0
    assert all(s["offset"] == 0 for s in disabled["states"].values())


def test_calibration_equal_day_weights_not_row_weights(sample, settings):
    rows, config = _validation(sample)
    dates = rows["date"].unique().sort()
    rows = rows.with_columns(
        pl.when(pl.col("date") == dates[0]).then(0.1).otherwise(-0.1).alias("target")
    )
    first = rows.filter(pl.col("date") == dates[0])
    # More stocks on one date must not change that date's calibration weight.
    more = pl.concat([rows, first.with_columns((pl.col("symbol") + "extra").alias("symbol"))])
    a = diagnostic.fit_calibration(rows, "prediction", config, settings)
    b = diagnostic.fit_calibration(more, "prediction", config, settings)
    assert a["global_offset"] == pytest.approx(b["global_offset"])


def test_state_shrinkage_and_rank_invariance(sample, settings):
    rows, config = _validation(sample)
    dates = rows["date"].unique().sort()
    rows = rows.with_columns(
        pl.when(pl.col("date") < dates[7])
        .then(pl.lit("strong"))
        .otherwise(pl.lit("weak"))
        .alias("market_state"),
        pl.col("prediction_baseline").alias("prediction"),
    )
    model = diagnostic.fit_calibration(rows, "prediction", config, settings)
    strong = model["states"]["strong"]
    assert strong["supported"]
    assert strong["shrink_weight"] == pytest.approx(7 / 27)
    assert strong["offset"] == pytest.approx(
        model["global_offset"] + 7 / 27 * (strong["residual_mean"] - model["global_offset"])
    )
    applied = diagnostic.apply_calibration(rows, "prediction", model)
    for mode in ("prediction_global", "prediction_state"):
        rank_before = applied.sort(["date", "prediction", "symbol"])["symbol"].to_list()
        rank_after = applied.sort(["date", mode, "symbol"])["symbol"].to_list()
        assert rank_before == rank_after


def test_test_mutations_do_not_change_fits_or_calibration(sample, settings):
    panel, l2, config, _ = sample
    config = replace(config, horizons=(1,))
    first = diagnostic.evaluate_l2_diagnostics(panel, l2, config, settings)
    changed = panel.with_columns(
        [
            pl.when(pl.col("date") >= config.test_start)
            .then(pl.col(c) * (1.3 + (pl.col("date") - config.test_start).dt.total_days() * 0.003))
            .otherwise(pl.col(c))
            .alias(c)
            for c in ("open", "high", "low", "close")
        ]
    )
    second = diagnostic.evaluate_l2_diagnostics(changed, l2.reverse(), config, settings)
    assert first.report["horizons"]["1"]["models"] == second.report["horizons"]["1"]["models"]
    for arm in ("baseline", "full_l2"):
        a = first.report["horizons"]["1"]["arms"][arm]
        b = second.report["horizons"]["1"]["arms"][arm]
        assert a["calibration_model"] == b["calibration_model"]
    first_test = first.predictions.filter(pl.col("split") == "test")["target"].to_numpy()
    second_test = second.predictions.filter(pl.col("split") == "test")["target"].to_numpy()
    assert not np.array_equal(first_test, second_test)


def test_ablation_rejects_partial_feature_mask(sample, settings):
    panel, l2, config, _ = sample
    with pytest.raises(ValueError, match="complete ordered"):
        diagnostic.evaluate_l2_diagnostics(
            panel, l2, replace(config, l2_features=("l2_trade_count",)), settings
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_state_dates": 0},
        {"min_market_coverage": 1.1},
        {"state_shrink_dates": 0},
        {"min_market_stocks": True},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        diagnostic.DiagnosticConfig(**kwargs)
