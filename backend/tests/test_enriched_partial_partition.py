"""#223 回归: 盘后管道必须识别并重算覆盖不全的 enriched 分区。

自选实时路径 (merge_live_enriched_asset) 会在全市场 enriched 生成前提前
创建当日分区 (只有几只自选); 仅按日期目录计数比较会把它误判为完整分区
而跳过计算, 造成日K连续缺失与均线错误。_prune_partial_enriched_partitions
按过滤停牌后的标的集合检查覆盖, 避免正确过滤的停牌记录导致反复删除重算。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from app.indicators import pipeline
from app.jobs import daily_pipeline
from app.jobs.daily_pipeline import _prune_partial_enriched_partitions
from app.services import daily_maintenance, data_integrity, maintenance_queue, preferences
from app.tickflow.capabilities import CapabilitySet


def _write_partition(base: Path, day: str, symbols: list[str]) -> None:
    part = base / f"date={day}"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "symbol": symbols,
        "open": [1.0] * len(symbols),
        "high": [1.0] * len(symbols),
        "close": [1.0] * len(symbols),
        "volume": [100.0] * len(symbols),
        "amount": [100.0] * len(symbols),
    }).write_parquet(
        part / "part.parquet"
    )


def test_partial_partition_is_pruned(tmp_path) -> None:
    daily = tmp_path / "kline_daily"
    enriched = tmp_path / "kline_daily_enriched"
    # 2026-08-28: daily 全市场 100 只, enriched 只有实时写入的 5 只 (issue 实测形态)
    _write_partition(daily, "2026-08-28", [f"s{i:06d}" for i in range(100)])
    _write_partition(enriched, "2026-08-28", [f"s{i:06d}" for i in range(5)])
    # 前一日两边都是完整的 → 不动
    _write_partition(daily, "2026-08-27", ["a", "b"])
    _write_partition(enriched, "2026-08-27", ["a", "b"])

    pruned = _prune_partial_enriched_partitions(daily, enriched)

    assert pruned == ["2026-08-28"]
    assert not (enriched / "date=2026-08-28").exists()
    assert (enriched / "date=2026-08-27").exists()  # 完整分区保留


def test_complete_partitions_untouched(tmp_path) -> None:
    daily = tmp_path / "kline_daily"
    enriched = tmp_path / "kline_daily_enriched"
    syms = [f"s{i:06d}" for i in range(50)]
    _write_partition(daily, "2026-09-01", syms)
    _write_partition(enriched, "2026-09-01", syms)

    assert _prune_partial_enriched_partitions(daily, enriched) == []
    assert (enriched / "date=2026-09-01" / "part.parquet").exists()


def test_enriched_date_without_daily_is_left_alone(tmp_path) -> None:
    # 今日日K尚未同步时, 实时创建的当日分区留给当日正常流程处理
    daily = tmp_path / "kline_daily"
    enriched = tmp_path / "kline_daily_enriched"
    day = "2026-09-07"
    _write_partition(enriched, day, ["only_watchlist"])

    assert _prune_partial_enriched_partitions(daily, enriched) == []
    assert (enriched / f"date={day}").exists()


@pytest.mark.parametrize("legacy_halt", [False, True])
def test_computed_partition_with_halt_is_preserved_on_repeated_checks(tmp_path, monkeypatch, legacy_halt):
    monkeypatch.setattr(pipeline, "_custom_signal_exprs", {})
    daily = tmp_path / "kline_daily"
    enriched = tmp_path / "kline_daily_enriched"
    day = "2026-09-07"
    _write_partition(daily, day, ["600001.SH", "600002.SH"])
    raw_path = daily / f"date={day}" / "part.parquet"
    raw = pl.read_parquet(raw_path).with_columns(
        pl.lit(date.fromisoformat(day)).alias("date"),
        pl.lit(1.0).alias("low"),
        *[
            pl.when(pl.col("symbol") == "600002.SH").then(0.0).otherwise(pl.col(c)).alias(c)
            for c in (["volume", "amount"] if legacy_halt else ["open", "high", "volume", "amount"])
        ],
    )
    raw.write_parquet(raw_path)
    computed = pipeline._select_storage_cols(pipeline.compute_enriched(raw))
    assert computed["symbol"].to_list() == ["600001"]
    target = enriched / f"date={day}" / "part.parquet"
    target.parent.mkdir(parents=True)
    computed.write_parquet(target)
    original = target.read_bytes()

    for _ in range(2):
        assert _prune_partial_enriched_partitions(daily, enriched) == []
        assert target.read_bytes() == original


@pytest.mark.parametrize("actual", [["a"], ["a", "a"], ["a", "extra"], ["a", "extra", "other"]])
def test_missing_active_symbol_is_not_hidden_by_counts(tmp_path, actual):
    daily = tmp_path / "daily"
    enriched = tmp_path / "enriched"
    _write_partition(daily, "2026-09-07", ["a", "b"])
    _write_partition(enriched, "2026-09-07", actual)

    assert _prune_partial_enriched_partitions(daily, enriched) == ["2026-09-07"]


def test_duplicate_daily_rows_do_not_require_duplicate_enriched_rows(tmp_path):
    daily = tmp_path / "daily"
    enriched = tmp_path / "enriched"
    _write_partition(daily, "2026-09-07", ["a", "a", "b"])
    _write_partition(enriched, "2026-09-07", ["a", "b"])

    assert _prune_partial_enriched_partitions(daily, enriched) == []


@pytest.mark.parametrize("missing", ["symbol", "open", "high"])
def test_missing_daily_required_column_does_not_delete(tmp_path, missing):
    daily = tmp_path / "daily"
    enriched = tmp_path / "enriched"
    _write_partition(daily, "2026-09-07", ["a", "b"])
    _write_partition(enriched, "2026-09-07", ["a"])
    raw_path = daily / "date=2026-09-07" / "part.parquet"
    pl.read_parquet(raw_path).drop(missing).write_parquet(raw_path)

    assert _prune_partial_enriched_partitions(daily, enriched) == []
    assert (enriched / "date=2026-09-07" / "part.parquet").exists()


@pytest.mark.parametrize("broken_kind", ["daily", "enriched"])
def test_unreadable_partition_does_not_delete(tmp_path, broken_kind):
    daily = tmp_path / "daily"
    enriched = tmp_path / "enriched"
    _write_partition(daily, "2026-09-07", ["a", "b"])
    _write_partition(enriched, "2026-09-07", ["a"])
    (tmp_path / broken_kind / "date=2026-09-07" / "broken.parquet").write_bytes(b"broken")

    assert _prune_partial_enriched_partitions(daily, enriched) == []
    assert (enriched / "date=2026-09-07" / "part.parquet").exists()


def test_partition_files_are_checked_together(tmp_path):
    daily = tmp_path / "daily"
    enriched = tmp_path / "enriched"
    _write_partition(daily, "2026-09-07", ["a", "b"])
    _write_partition(enriched, "2026-09-07", ["a"])
    target = enriched / "date=2026-09-07" / "second.parquet"
    pl.DataFrame({"symbol": ["b"]}).write_parquet(target)

    assert _prune_partial_enriched_partitions(daily, enriched) == []


@pytest.mark.parametrize("missing", ["volume", "amount"])
def test_legacy_daily_without_optional_halt_column(tmp_path, missing):
    daily = tmp_path / "daily"
    enriched = tmp_path / "enriched"
    _write_partition(daily, "2026-09-07", ["a", "b"])
    _write_partition(enriched, "2026-09-07", ["a", "b"])
    path = daily / "date=2026-09-07" / "part.parquet"
    pl.read_parquet(path).drop(missing).write_parquet(path)
    assert _prune_partial_enriched_partitions(daily, enriched) == []


def test_all_halted_partition_does_not_require_active_rows(tmp_path):
    daily = tmp_path / "daily"
    enriched = tmp_path / "enriched"
    _write_partition(daily, "2026-09-07", ["a", "b"])
    _write_partition(enriched, "2026-09-07", ["a"])
    path = daily / "date=2026-09-07" / "part.parquet"
    pl.read_parquet(path).with_columns(pl.lit(0.0).alias("open"), pl.lit(0.0).alias("high")).write_parquet(path)
    assert _prune_partial_enriched_partitions(daily, enriched) == []


@pytest.mark.parametrize("defect", ["missing", "partial", "stale_price"])
def test_interior_date_is_queued_and_rebuilt_without_new_daily_or_factors(tmp_path, monkeypatch, defect):
    daily = tmp_path / "kline_daily"
    enriched = tmp_path / "kline_daily_enriched"
    dates = [f"2026-09-{d:02}" for d in (1, 2, 3, 4, 7, 8, 9, 10)]
    for day in dates:
        _write_partition(daily, day, ["a", "b"])
        if day == dates[1] and defect == "missing":
            continue
        _write_partition(enriched, day, ["a"] if day == dates[1] and defect == "partial" else ["a", "b"])
        target = enriched / f"date={day}" / "part.parquet"
        pl.read_parquet(target).with_columns(
            pl.lit(0.9 if day == dates[1] and defect == "stale_price" else 1.0).alias("raw_close")
        ).write_parquet(target)

    monkeypatch.setattr(daily_pipeline.settings, "data_dir", tmp_path)
    monkeypatch.setattr(daily_pipeline, "cn_today", lambda: date.fromisoformat(dates[-1]))
    monkeypatch.setattr(preferences, "load", lambda: {
        "pipeline_pull_a_share": False, "pipeline_regime_enabled": False,
        "minute_sync_enabled": False, "adj_factor_provider": "tickflow",
    })
    monkeypatch.setattr(daily_pipeline.instrument_sync, "sync_instruments", lambda *_: 0)
    monkeypatch.setattr(daily_pipeline, "_resolve_universe", lambda *_: [])
    monkeypatch.setattr(daily_pipeline, "_invalidate", lambda *_: None)
    monkeypatch.setattr(daily_pipeline, "_refresh_single_view", lambda *_: None)
    monkeypatch.setattr(daily_pipeline, "_refresh_views", lambda *_: None)
    monkeypatch.setattr(data_integrity, "scan_recent_integrity", lambda *_, **__: [])
    calls = []

    def rebuild(root, day):
        assert root == tmp_path
        calls.append(day)
        _write_partition(enriched, dates[1], ["a", "b"])
        return 2

    monkeypatch.setattr(pipeline, "run_date_pipeline", rebuild)
    monkeypatch.setattr(daily_pipeline, "run_pipeline", lambda **_: pytest.fail("historical work must be queued"))
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path),
                           latest_daily_date=lambda: date.fromisoformat(dates[-1]), refresh_cache=lambda: None)
    daily_pipeline.run_now(repo, CapabilitySet(set()))
    assert calls == []
    task = maintenance_queue._read(tmp_path)[0]
    assert task["kind"] == "enriched_dates" and task["items"] == [dates[1]]
    daily_pipeline.run_now(repo, CapabilitySet(set()))
    assert len(maintenance_queue._read(tmp_path)) == 1

    def runner(fn, label, on_started):
        on_started()
        fn()
        return True

    assert maintenance_queue.tick(tmp_path, runner, lambda task, emit: daily_maintenance.execute(
        repo, CapabilitySet(set()), task, emit))
    assert calls == [date.fromisoformat(dates[1])]
    assert pl.read_parquet(enriched / f"date={dates[1]}" / "part.parquet")["symbol"].to_list() == ["a", "b"]
    rows = maintenance_queue._read(tmp_path)
    assert next(r for r in rows if r["kind"] == "enriched_dates")["status"] == "succeeded"
    assert next(r for r in rows if r["kind"] == "regime")["options"]["reason"] == "input_changed"
