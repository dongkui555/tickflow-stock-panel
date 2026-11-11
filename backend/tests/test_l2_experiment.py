"""Deterministic, offline contracts for the paired L2 research experiment."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from app.services import l2_experiment as experiment


@pytest.fixture
def sample():
    days = []
    day = date(2024, 1, 2)
    while len(days) < 110:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    panel_rows, l2_rows = [], []
    for stock in range(5):
        close = 10.0 + stock
        for i, day in enumerate(days):
            close *= 1 + 0.003 * np.sin(i * 0.47 + stock) + 0.0003 * (stock - 2)
            panel_rows.append(
                {
                    "symbol": f"{stock + 1:06d}",
                    "date": day,
                    "open": close * (1 - 0.001 * np.cos(i + stock)),
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 100000.0 + 1000 * i + 300 * stock,
                    "amount": close * (100000.0 + 1000 * i + 300 * stock),
                    "turnover_rate": 1.0 + 0.1 * stock + 0.01 * (i % 5),
                }
            )
            l2_rows.append(
                {
                    "symbol": f"{stock + 1:06d}.SZ",
                    "date": day,
                    "quality_status": "validated",
                    "feature_version": "l2_daily_research_v1",
                    "source_contract": "wind_csv_sh_sz_research_v1",
                    "available_at": datetime.combine(day + timedelta(days=1), time()),
                    **{
                        name: float(np.cos(i * 0.3 + stock + n))
                        for n, name in enumerate(experiment.L2_FEATURE_COLUMNS)
                    },
                    "l2_future_label": 999999.0,
                    "source_archive": "not-a-feature.zip",
                }
            )
    config = experiment.L2ExperimentConfig(
        history_start=days[0],
        train_start=days[35],
        validation_start=days[65],
        test_start=days[85],
        end=days[-1],
        min_history_days=25,
        max_symbols=4,
        horizons=(1, 3),
        top_k=2,
        round_trip_cost_bps=12.0,
    )
    return pl.DataFrame(panel_rows), pl.DataFrame(l2_rows), config, days


def test_deterministic_paired_masks_labels_and_purge(sample):
    panel, l2, config, _ = sample
    result = experiment.evaluate_l2_experiment(panel, l2, config)
    repeat = experiment.evaluate_l2_experiment(panel.reverse(), l2.reverse(), config)
    assert result.report == repeat.report
    assert result.predictions.equals(repeat.predictions)
    json.dumps(result.report, allow_nan=False)
    assert result.report["research_only"] is True
    assert result.report["executable_backtest"] is False
    assert result.report["universe"]["symbols"] == ["000001", "000002", "000003", "000004"]
    assert "l2_future_label" not in result.report["features"]["baseline_plus_l2"]
    assert "source_archive" not in result.report["features"]["baseline_plus_l2"]
    for horizon in config.horizons:
        rows = result.predictions.filter(pl.col("horizon") == horizon)
        for split, upper in [("train", config.validation_start), ("validation", config.test_start)]:
            part = rows.filter(pl.col("split") == split)
            assert part.height > 0
            assert part["label_end"].max() < upper
            metrics = result.report["horizons"][str(horizon)]["splits"][split]
            assert metrics["baseline"]["rows"] == metrics["baseline_plus_l2"]["rows"]
            assert metrics["baseline"]["rows"] == part.height
            assert metrics["samples"]["purged_rows"] == (horizon + 1) * config.max_symbols
        assert rows["label_end"].max() <= config.end
        assert rows.filter(pl.col("available_at") > pl.col("decision_at")).is_empty()
        assert rows.filter(pl.col("l2_date") != pl.col("date")).is_empty()
        assert rows.filter(pl.col("entry_date") <= pl.col("date")).is_empty()
        assert rows.filter(pl.col("label_end") <= pl.col("entry_date")).is_empty()


def test_train_normalization_and_weights_ignore_holdouts(sample):
    panel, l2, config, _ = sample
    first = experiment.evaluate_l2_experiment(panel, l2, config)
    feature = experiment.L2_FEATURE_COLUMNS[0]
    l2 = l2.with_columns(
        pl.when(pl.col("date") >= config.validation_start)
        .then(1e8)
        .otherwise(pl.col(feature))
        .alias(feature)
    )
    panel = panel.with_columns(
        [
            pl.when(pl.col("date") >= config.validation_start)
            .then(pl.col(column) * 2)
            .otherwise(pl.col(column))
            .alias(column)
            for column in ("open", "high", "low", "close")
        ]
    )
    second = experiment.evaluate_l2_experiment(panel, l2, config)
    for horizon in config.horizons:
        a = first.report["horizons"][str(horizon)]["models"]
        b = second.report["horizons"][str(horizon)]["models"]
        assert a == b
        np.testing.assert_allclose(
            a["baseline"]["scaler"]["mean"],
            a["baseline_plus_l2"]["scaler"]["mean"][: len(experiment.BASELINE_FEATURE_COLUMNS)],
        )
    train = first.predictions.filter((pl.col("split") == "train") & (pl.col("horizon") == 1))
    sources = train.select("symbol", pl.col("l2_date").alias("date"))
    values = l2.with_columns(pl.col("symbol").str.replace(r"\.SZ$", ""))
    expected = sources.join(values, on=["symbol", "date"])[feature].mean()
    scaler = first.report["horizons"]["1"]["models"]["baseline_plus_l2"]["scaler"]
    assert scaler["mean"][len(experiment.BASELINE_FEATURE_COLUMNS)] == pytest.approx(expected)


def test_fixed_universe_uses_only_presample_history(sample):
    panel, l2, config, _ = sample
    new_stock = (
        panel.filter(pl.col("symbol") == "000005")
        .with_columns(pl.lit("000000").alias("symbol"), pl.lit(1e16).alias("amount"))
        .filter(pl.col("date") >= config.train_start)
    )
    panel = pl.concat([panel, new_stock]).with_columns(
        pl.when((pl.col("symbol") == "000005") & (pl.col("date") >= config.train_start))
        .then(1e18)
        .otherwise(pl.col("amount"))
        .alias("amount")
    )
    result = experiment.evaluate_l2_experiment(panel, l2, config)
    assert result.report["universe"]["symbols"] == ["000001", "000002", "000003", "000004"]


def test_l2_missing_rejected_delayed_and_nonfinite_share_mask(sample):
    panel, l2, config, days = sample
    missing_day, rejected_day, delayed_day, nan_day = days[45:49]
    first_feature = experiment.L2_FEATURE_COLUMNS[0]
    is_first = pl.col("symbol") == "000001.SZ"
    l2 = l2.filter(~(is_first & (pl.col("date") == missing_day))).with_columns(
        pl.when(is_first & (pl.col("date") == rejected_day))
        .then(pl.lit("quarantined"))
        .otherwise(pl.col("quality_status"))
        .alias("quality_status"),
        pl.when(is_first & (pl.col("date") == delayed_day))
        .then(datetime.combine(days[48], time(16)))
        .otherwise(pl.col("available_at"))
        .alias("available_at"),
        pl.when(is_first & (pl.col("date") == nan_day))
        .then(float("nan"))
        .otherwise(pl.col(first_feature))
        .alias(first_feature),
    )
    result = experiment.evaluate_l2_experiment(panel, l2, config)
    first = result.predictions.filter(pl.col("symbol") == "000001")
    assert first.filter(pl.col("date").is_in(days[45:49])).is_empty()
    assert (
        result.report["coverage"]["common_feature_rows"]
        < result.report["coverage"]["baseline_feature_rows"]
    )


def test_weekend_and_same_day_availability(sample):
    panel, l2, config, days = sample
    result = experiment.evaluate_l2_experiment(panel, l2, config)
    monday = next(day for day in days[40:50] if day.weekday() == 0)
    rows = result.predictions.filter(pl.col("entry_date") == monday)
    assert rows["l2_date"].unique().to_list() == [monday - timedelta(days=3)]
    assert rows["available_at"].unique().to_list() == [
        datetime.combine(monday - timedelta(days=2), time())
    ]
    bad = l2.with_columns(pl.col("date").cast(pl.Datetime).alias("available_at"))
    with pytest.raises(ValueError, match="next-calendar-day"):
        experiment.evaluate_l2_experiment(panel, bad, config)


def test_horizon_uses_market_dates_not_next_observed_stock_bar(sample):
    panel, l2, config, days = sample
    panel = panel.filter(~((pl.col("symbol") == "000001") & (pl.col("date") == days[50])))
    result = experiment.evaluate_l2_experiment(panel, l2, config)
    rows = result.predictions.filter((pl.col("symbol") == "000001") & (pl.col("horizon") == 1))
    assert rows.filter(pl.col("date") == days[49]).is_empty()
    assert rows.filter(pl.col("date") == days[48]).is_empty()
    assert rows.filter(pl.col("date") == days[51]).is_empty()  # no compressed rolling history
    row = rows.filter(pl.col("date") == days[45]).row(0, named=True)
    prices = panel.filter(pl.col("symbol") == "000001")
    start_open = prices.filter(pl.col("date") == days[46])["open"][0]
    end_open = prices.filter(pl.col("date") == days[47])["open"][0]
    assert row["entry_date"] == days[46]
    assert row["label_end"] == days[47]
    assert row["target"] == pytest.approx(end_open / start_open - 1)


def test_daily_metrics_not_pooled_ic_and_fixed_cost():
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)] * 3 + [date(2024, 1, 3)] * 3,
            "symbol": ["000003", "000002", "000001"] * 2,
            "target": [0.1, 0.2, 0.3, 0.8, 0.7, 0.6],
            "prediction": [1.0, 2.0, 3.0, 6.0, 7.0, 8.0],
        }
    )
    metrics = experiment._metrics(frame, "prediction", top_k=1, cost=0.001)
    assert metrics["ic"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["rank_ic"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["ic_days"] == metrics["rank_ic_days"] == 2
    assert metrics["top_k_forward_return"] == pytest.approx(0.45)
    assert metrics["top_k_after_cost_proxy"] == pytest.approx(0.449)
    assert metrics["mse"] == pytest.approx(((frame["prediction"] - frame["target"]) ** 2).mean())
    constant = frame.with_columns(pl.lit(1.0).alias("prediction"))
    metrics = experiment._metrics(constant, "prediction", top_k=1, cost=0)
    assert metrics["ic"] is None and metrics["rank_ic"] is None
    assert metrics["top_k_forward_return"] == pytest.approx(0.45)  # symbol tie-break


@pytest.mark.parametrize(
    "kind", ["duplicate_panel", "duplicate_l2", "missing_feature", "timezone", "dtype"]
)
def test_contract_failures(sample, kind):
    panel, l2, config, _ = sample
    if kind == "duplicate_panel":
        panel = pl.concat([panel, panel.head(1)])
    elif kind == "duplicate_l2":
        l2 = pl.concat([l2, l2.head(1)])
    elif kind == "missing_feature":
        l2 = l2.drop(experiment.L2_FEATURE_COLUMNS[0])
    elif kind == "timezone":
        l2 = l2.with_columns(pl.col("available_at").dt.replace_time_zone("UTC"))
    else:
        l2 = l2.with_columns(pl.col(experiment.L2_FEATURE_COLUMNS[0]).cast(pl.String))
    with pytest.raises(ValueError):
        experiment.evaluate_l2_experiment(panel, l2, config)


def test_empty_splits_and_whitelist_fail_closed(sample):
    panel, l2, config, _ = sample
    with pytest.raises(ValueError, match="whitelist"):
        replace(config, l2_features=("source_archive",))
    with pytest.raises(ValueError, match="chronological"):
        replace(config, validation_start=config.train_start)
    with pytest.raises(ValueError, match="common"):
        experiment.evaluate_l2_experiment(
            panel,
            l2.filter(pl.col("date") < config.test_start - timedelta(days=5)),
            config,
        )


def _write_l2_partition(root, day, part):
    path = root / "l2_research" / "daily" / f"date={day}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    part.write_parquet(path)
    manifest_path = root / "l2_research" / "extraction" / f"{day}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "date": str(day),
        "status": "complete",
        "accepted": part.height,
        "signature": {
            "archive": "fixture.zip",
            "size": 10,
            "mtime_ns": 123,
            "daily_sha256": "a" * 64,
            "symbols": sorted(part["symbol"].to_list()),
            "feature_version": "l2_daily_research_v1",
        },
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "output_bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest_path


def test_streaming_daily_partitions_and_no_production_writes(sample, tmp_path):
    panel, l2, config, _ = sample
    enriched = tmp_path / "kline_daily_enriched" / "year=2024"
    enriched.mkdir(parents=True)
    panel.write_parquet(enriched / "part.parquet")
    for (day,), part in l2.partition_by("date", as_dict=True).items():
        _write_l2_partition(tmp_path, day, part)
    production = tmp_path / "stockformer"
    production.mkdir()
    sentinel = production / "latest_model.txt"
    sentinel.write_text("production-unchanged", encoding="utf-8")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = experiment.run_l2_experiment(tmp_path, config)
    assert result.report["horizons"]["1"]["splits"]["test"]["baseline"]["rows"] > 0
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert result.report == experiment.evaluate_l2_experiment(panel, l2, config).report


def test_partition_date_mismatch_is_rejected(sample, tmp_path):
    _, l2, config, days = sample
    _write_l2_partition(tmp_path, days[40], l2.filter(pl.col("date") == days[41]))
    with pytest.raises(ValueError, match="partition date"):
        experiment.load_l2_daily(tmp_path, config)


def test_quality_metadata_and_raw_prices_are_not_features(sample):
    panel, l2, config, _ = sample
    before = experiment.evaluate_l2_experiment(panel, l2, config)
    panel = panel.with_columns(pl.lit(1e8).alias("raw_open"), pl.lit(1e8).alias("raw_close"))
    l2 = l2.with_columns(
        pl.lit(1e8).alias("l2_side_coverage"),
        pl.lit(None, dtype=pl.Float64).alias("l2_quote_coverage"),
    )
    after = experiment.evaluate_l2_experiment(panel, l2, config)
    assert before.report == after.report
    assert before.predictions.equals(after.predictions)


def test_next_open_and_exit_are_future_labels_not_signal_inputs(sample):
    panel, l2, config, days = sample
    before = experiment.evaluate_l2_experiment(panel, l2, config)
    panel = panel.with_columns(
        pl.when(pl.col("date") == days[91])
        .then(pl.col("open") * 1.1)
        .otherwise(pl.col("open"))
        .alias("open")
    ).with_columns(pl.max_horizontal("high", "open").alias("high"))
    after = experiment.evaluate_l2_experiment(panel, l2, config)
    mask = (pl.col("date") == days[90]) & (pl.col("horizon") == 1)
    a, b = before.predictions.filter(mask), after.predictions.filter(mask)
    assert a["prediction_baseline"].equals(b["prediction_baseline"])
    assert a["prediction_baseline_plus_l2"].equals(b["prediction_baseline_plus_l2"])
    np.testing.assert_allclose(b["target"].to_numpy() + 1, (a["target"].to_numpy() + 1) / 1.1)


def test_explicit_universe_and_two_stock_diagnostics(sample):
    panel, l2, config, _ = sample
    config = replace(config, symbols=("000001.SZ", "000005.SZ"))
    result = experiment.evaluate_l2_experiment(panel, l2, config)
    assert result.report["universe"]["symbols"] == ["000001", "000005"]
    metrics = result.report["horizons"]["1"]["splits"]["test"]["baseline"]
    assert metrics["ic"] is None and metrics["ic_days"] == 0
    assert any("IC/RankIC unavailable" in message for message in result.report["diagnostics"])
    with pytest.raises(ValueError, match="pre-sample history"):
        experiment.evaluate_l2_experiment(panel, l2, replace(config, symbols=("000099",)))


def test_all_horizons_use_the_exact_exit_open(sample):
    panel, l2, config, days = sample
    config = replace(config, horizons=(1, 3, 5, 10))
    result = experiment.evaluate_l2_experiment(panel, l2, config)
    stock = panel.filter(pl.col("symbol") == "000001")
    entry = stock.filter(pl.col("date") == days[91])["open"][0]
    for horizon in config.horizons:
        row = result.predictions.filter(
            (pl.col("symbol") == "000001")
            & (pl.col("date") == days[90])
            & (pl.col("horizon") == horizon)
        ).row(0, named=True)
        exit_open = stock.filter(pl.col("date") == days[91 + horizon])["open"][0]
        assert row["label_end"] == days[91 + horizon]
        assert row["target"] == pytest.approx(exit_open / entry - 1)


def test_same_cost_proxy_never_changes_labels_or_models(sample):
    panel, l2, config, _ = sample
    free = experiment.evaluate_l2_experiment(panel, l2, replace(config, round_trip_cost_bps=0))
    costed = experiment.evaluate_l2_experiment(panel, l2, replace(config, round_trip_cost_bps=20))
    assert free.predictions.equals(costed.predictions)
    for horizon in config.horizons:
        a, b = free.report["horizons"][str(horizon)], costed.report["horizons"][str(horizon)]
        assert a["models"] == b["models"]
        for split in ("train", "validation", "test"):
            for arm in ("baseline", "baseline_plus_l2"):
                gross = a["splits"][split][arm]
                net = b["splits"][split][arm]
                assert gross["mse"] == net["mse"]
                assert gross["top_k_forward_return"] == net["top_k_forward_return"]
                assert net["top_k_after_cost_proxy"] == pytest.approx(
                    gross["top_k_after_cost_proxy"] - 0.002
                )


def test_availability_cutoff_is_next_session_0925(sample):
    panel, l2, config, days = sample
    stamp = datetime.combine(days[45], time(9, 25))
    l2 = l2.with_columns(
        pl.when((pl.col("symbol") == "000001.SZ") & (pl.col("date") == days[44]))
        .then(stamp)
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    accepted = experiment.evaluate_l2_experiment(panel, l2, config)
    mask = (pl.col("symbol") == "000001") & (pl.col("date") == days[44])
    assert accepted.predictions.filter(mask).height == len(config.horizons)
    l2 = l2.with_columns(
        pl.when(pl.col("available_at") == stamp)
        .then(stamp + timedelta(seconds=1))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    rejected = experiment.evaluate_l2_experiment(panel, l2, config)
    assert rejected.predictions.filter(mask).is_empty()


def test_ohlc_shape_checks_both_open_and_close_and_allows_equal_extremes():
    frame = pl.DataFrame(
        {
            "open": [10.0, 9.0, 12.0, 7.0, 9.0, 9.0],
            "high": [10.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            "low": [10.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            "close": [10.0, 10.0, 10.0, 10.0, 12.0, 7.0],
            "volume": [100.0] * 6,
            "amount": [1000.0] * 6,
        }
    )
    assert frame.select(experiment._valid_bar()).to_series().to_list() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize("kind", ["missing", "null", "nan", "high", "low", "volume"])
def test_bad_bar_requires_twenty_one_consecutive_valid_bars(sample, monkeypatch, kind):
    panel, l2, config, days = sample
    bad = (pl.col("symbol") == "000001") & (pl.col("date") == days[45])
    if kind == "missing":
        panel = panel.filter(~bad)
    else:
        column, value = {
            "null": ("low", None),
            "nan": ("high", float("nan")),
            "high": ("high", 0.1),
            "low": ("low", 1000.0),
            "volume": ("volume", -1.0),
        }[kind]
        panel = panel.with_columns(pl.when(bad).then(value).otherwise(pl.col(column)).alias(column))
    original = experiment.stockformer._feature_frame

    def default_filled(frame, include_targets):
        return original(frame, include_targets=include_targets).with_columns(
            [pl.col(name).fill_null(0.0) for name in experiment.BASELINE_FEATURE_COLUMNS]
        )

    monkeypatch.setattr(experiment.stockformer, "_feature_frame", default_filled)
    common, *_ = experiment._common_frame(panel, l2, config)
    stock = common.filter(pl.col("symbol") == "000001")
    assert stock.filter(pl.col("date").is_between(days[45], days[65])).is_empty()
    assert stock.filter(pl.col("date") == days[44]).height == 1
    assert stock.filter(pl.col("date") == days[66]).height == 1


def test_initial_history_gate_does_not_trust_feature_defaults(sample, monkeypatch):
    panel, l2, config, days = sample
    config = replace(config, train_start=days[10], min_history_days=5)

    def all_default_features(frame, include_targets):
        return frame.select("symbol", "date").with_columns(
            [pl.lit(0.0).alias(name) for name in experiment.BASELINE_FEATURE_COLUMNS]
        )

    monkeypatch.setattr(experiment.stockformer, "_feature_frame", all_default_features)
    common, *_ = experiment._common_frame(panel, l2, config)
    assert common["date"].min() == days[20]  # current bar plus 20 previous global sessions


@pytest.mark.parametrize("column,value", [("high", 0.1), ("low", 1000.0), ("close", None)])
def test_invalid_ohlc_entry_and_exit_bars_are_excluded(sample, column, value):
    panel, l2, config, days = sample
    bad = (pl.col("symbol") == "000001") & (pl.col("date") == days[45])
    panel = panel.with_columns(pl.when(bad).then(value).otherwise(pl.col(column)).alias(column))
    result = experiment.evaluate_l2_experiment(panel, l2, config)
    stock = result.predictions.filter((pl.col("symbol") == "000001") & (pl.col("horizon") == 1))
    assert stock.filter(pl.col("date").is_in([days[43], days[44]])).is_empty()
    assert stock.filter(pl.col("date") == days[42]).height == 1
    counts = result.report["horizons"]["1"]["splits"]["train"]["samples"]
    assert counts["missing_endpoint_rows"] == 2


@pytest.mark.parametrize(
    "kind",
    [
        "missing_manifest",
        "incomplete",
        "wrong_date",
        "signature_version",
        "signature_contract",
        "row_version",
        "row_contract",
        "missing_provenance",
        "output_hash",
        "modified_parquet",
        "accepted_count",
        "signature_symbols",
        "malformed_json",
        "non_object",
    ],
)
def test_l2_manifest_and_immutable_versions_fail_closed(sample, tmp_path, kind):
    _, l2, config, days = sample
    part = l2.filter(pl.col("date") == days[40])
    if kind == "row_version":
        part = part.with_columns(pl.lit("unknown").alias("feature_version"))
    elif kind == "row_contract":
        part = part.with_columns(pl.lit("unknown").alias("source_contract"))
    elif kind == "missing_provenance":
        part = part.drop("source_contract")
    path, manifest_path = _write_l2_partition(tmp_path, days[40], part)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if kind == "incomplete":
        manifest["status"] = "running"
    elif kind == "wrong_date":
        manifest["date"] = str(days[41])
    elif kind == "signature_version":
        manifest["signature"]["feature_version"] = "unknown"
    elif kind == "signature_contract":
        manifest["signature"]["source_contract"] = "unknown"
    elif kind == "output_hash":
        manifest["output_sha256"] = "0" * 64
    elif kind == "modified_parquet":
        part.with_columns(pl.lit(123.0).alias(experiment.L2_FEATURE_COLUMNS[0])).write_parquet(path)
    elif kind == "accepted_count":
        manifest["accepted"] += 1
    elif kind == "signature_symbols":
        manifest["signature"]["symbols"] = ["999999.SZ"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if kind == "missing_manifest":
        manifest_path.unlink()
    elif kind == "malformed_json":
        manifest_path.write_text("{", encoding="utf-8")
    elif kind == "non_object":
        manifest_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match=r"manifest|provenance|SHA-256"):
        experiment.load_l2_daily(tmp_path, config)


def test_manifest_versions_are_metadata_not_features(sample, tmp_path):
    _, l2, config, days = sample
    part = l2.filter(pl.col("date") == days[40])
    _, manifest_path = _write_l2_partition(tmp_path, days[40], part)
    legacy = experiment.load_l2_daily(tmp_path, config)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["signature"]["source_contract"] = "wind_csv_sh_sz_research_v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    explicit = experiment.load_l2_daily(tmp_path, config)
    assert legacy.equals(explicit)
    assert set(explicit.columns) == {
        "symbol",
        "date",
        "quality_status",
        "available_at",
        *experiment.L2_FEATURE_COLUMNS,
    }


@pytest.mark.parametrize(
    "alias,code",
    [
        ("000001", "000001"),
        ("000001.SZ", "000001"),
        ("SZ000001", "000001"),
        ("300750", "300750"),
        ("300750.SZ", "300750"),
        ("SZ300750", "300750"),
        ("600519", "600519"),
        ("600519.SH", "600519"),
        ("SH600519", "600519"),
        ("688981", "688981"),
        ("688981.SH", "688981"),
        ("SH688981", "688981"),
    ],
)
def test_keys_infer_only_supported_equity_exchanges(alias, code):
    frame = pl.DataFrame({"symbol": [alias], "date": [date(2026, 1, 5)]})
    assert experiment._keys(frame, "test")["symbol"].to_list() == [code]


@pytest.mark.parametrize(
    "alias",
    [
        "000001.SH",
        "SH000001",
        "300750.SH",
        "600519.SZ",
        "SZ600519",
        "688981.SZ",
        "000001.BJ",
        "920001.BJ",
        "BJ830799",
        "830799",
        "510300.SH",
        "159915.SZ",
    ],
)
def test_keys_reject_wrong_exchange_indices_etfs_and_bj_before_normalizing(alias, monkeypatch):
    def must_not_normalize(frame):
        pytest.fail("Invalid exchange identity reached the lossy normalization helper")

    monkeypatch.setattr(experiment.stockformer, "_normalize_symbol_column", must_not_normalize)
    frame = pl.DataFrame({"symbol": [alias], "date": [date(2026, 1, 5)]})
    with pytest.raises(ValueError, match="exchange identity"):
        experiment._keys(frame, "test")


def test_index_cannot_join_equity_under_a_different_exchange(sample):
    panel, l2, config, _ = sample
    l2 = l2.with_columns(
        pl.when(pl.col("symbol") == "000001.SZ")
        .then(pl.lit("000001.SH"))
        .otherwise(pl.col("symbol"))
        .alias("symbol")
    )
    with pytest.raises(ValueError, match="exchange identity"):
        experiment.evaluate_l2_experiment(panel, l2, config)


def test_cli_writes_only_new_research_directory(sample, tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_l2_experiment.py"
    spec = importlib.util.spec_from_file_location("run_l2_experiment_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    panel, l2, config, _ = sample
    result = experiment.evaluate_l2_experiment(panel, l2, config)
    monkeypatch.setattr(cli, "run_l2_experiment", lambda *args, **kwargs: result)
    args = [
        "--data-dir",
        str(tmp_path),
        "--history-start",
        str(config.history_start),
        "--train-start",
        str(config.train_start),
        "--validation-start",
        str(config.validation_start),
        "--test-start",
        str(config.test_start),
        "--end",
        str(config.end),
    ]
    assert cli.main(args) == 0
    assert not list(tmp_path.iterdir())
    capsys.readouterr()
    assert cli.main([*args, "--run-id", "fixture"]) == 0
    output = tmp_path / "l2_research" / "experiments" / "fixture"
    assert json.loads((output / "report.json").read_text("utf-8")) == result.report
    assert pl.read_parquet(output / "paired_predictions.parquet").equals(result.predictions)
    assert cli.main([*args, "--run-id", "fixture"]) == 2
    assert cli.main([*args, "--run-id", "../../stockformer"]) == 2
    assert not (tmp_path / "stockformer").exists()
    capsys.readouterr()

    def broken_input(*args, **kwargs):
        raise pl.exceptions.ComputeError("invalid parquet")

    monkeypatch.setattr(cli, "run_l2_experiment", broken_input)
    assert cli.main(args) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["research_only"] is True
    assert error["error"] == "invalid parquet"


def test_loader_excludes_bj_from_shared_store_before_identity_validation(
    sample, tmp_path, monkeypatch
):
    panel, l2, config, _ = sample
    bj = panel.head(1).with_columns(pl.lit("920001").alias("symbol"))
    monkeypatch.setattr(
        experiment, "scan_enriched_parquet", lambda *args: pl.concat([panel, bj]).lazy()
    )
    monkeypatch.setattr(experiment, "load_l2_daily", lambda *args, **kwargs: l2)
    result = experiment.run_l2_experiment(tmp_path, config)
    assert result.report["coverage"]["panel_rows"] == panel.height
    assert "920001" not in result.report["universe"]["symbols"]


@pytest.mark.parametrize("custom_root", [False, True])
def test_cli_uses_frozen_extractor_universe(sample, tmp_path, monkeypatch, capsys, custom_root):
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_l2_experiment.py"
    spec = importlib.util.spec_from_file_location("l2_frozen_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    panel, l2, config, _ = sample
    result = experiment.evaluate_l2_experiment(panel, l2, config)
    research_root = tmp_path / ("l2_research_50" if custom_root else "l2_research")
    frozen = research_root / "universe.json"
    frozen.parent.mkdir()
    frozen.write_text(json.dumps({"symbols": ["000001.SZ", "000002.SZ"]}))

    def run(root, actual_config, *, research_dir=None, extra_research_dirs=()):
        assert root == tmp_path
        assert research_dir == research_root.resolve()
        assert extra_research_dirs == ()
        assert actual_config.symbols == ("000001.SZ", "000002.SZ")
        return result

    monkeypatch.setattr(cli, "run_l2_experiment", run)
    args = [
        "--data-dir",
        str(tmp_path),
        "--history-start",
        str(config.history_start),
        "--train-start",
        str(config.train_start),
        "--validation-start",
        str(config.validation_start),
        "--test-start",
        str(config.test_start),
        "--end",
        str(config.end),
    ]
    if custom_root:
        args.extend(["--research-dir", str(research_root), "--run-id", "expanded"])
    assert cli.main(args) == 0
    if custom_root:
        assert (research_root / "experiments/expanded/report.json").is_file()
        assert not (tmp_path / "l2_research").exists()
    capsys.readouterr()
    frozen.write_text('{"symbols": "not-a-list"}')
    assert cli.main(args) == 2
    assert "Invalid frozen universe" in capsys.readouterr().err


def test_isolated_research_partitions_keep_market_data_root(sample, tmp_path, monkeypatch):
    panel, l2, config, _ = sample
    research_dir = tmp_path / "l2_research_50"
    for day in l2["date"].unique():
        _write_l2_partition(tmp_path, day, l2.filter(pl.col("date") == day))
    (tmp_path / "l2_research").rename(research_dir)

    def scan(path):
        assert Path(path).parts[: len(tmp_path.parts)] == tmp_path.parts
        assert "kline_daily_enriched" in Path(path).parts
        assert "l2_research_50" not in Path(path).parts
        return panel.lazy()

    monkeypatch.setattr(experiment, "scan_enriched_parquet", scan)
    with pytest.raises(ValueError, match="No L2 daily partitions"):
        experiment.run_l2_experiment(tmp_path, config)
    result = experiment.run_l2_experiment(tmp_path, config, research_dir=research_dir)
    expected = experiment.evaluate_l2_experiment(panel, l2, config)
    assert result.predictions.equals(expected.predictions)
    assert result.report == expected.report
    assert not (tmp_path / "l2_research").exists()


def test_additional_extraction_batches_are_paired_and_reject_overlap(sample, tmp_path, monkeypatch):
    panel, l2, config, _ = sample
    first_symbol = l2["symbol"][0]
    extra_root = tmp_path / "supplement"
    for day in l2["date"].unique():
        rows = l2.filter(pl.col("date") == day)
        _write_l2_partition(tmp_path, day, rows.filter(pl.col("symbol") != first_symbol))
        _write_l2_partition(extra_root, day, rows.filter(pl.col("symbol") == first_symbol))
    monkeypatch.setattr(experiment, "scan_enriched_parquet", lambda *args: panel.lazy())
    result = experiment.run_l2_experiment(
        tmp_path, config, extra_research_dirs=(extra_root / "l2_research",)
    )
    expected = experiment.evaluate_l2_experiment(panel, l2, config)
    assert result.predictions.equals(expected.predictions)
    assert result.report == expected.report
    with pytest.raises(ValueError, match="duplicate symbol/date"):
        experiment.run_l2_experiment(
            tmp_path, config, extra_research_dirs=(tmp_path / "l2_research",)
        )
