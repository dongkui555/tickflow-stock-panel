from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl
import pytest


def history(n=65):
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(n * 2)]
    days = [d for d in days if d.weekday() < 5][:n]
    prices = [10.0 + i * 0.02 for i in range(n)]
    return pl.DataFrame({"symbol": ["000001.SZ"] * n, "date": days,
                         **{c: prices for c in ("open", "high", "low", "close", "raw_close", "raw_high", "raw_low")},
                         "volume": [1000.] * n, "amount": [10000.] * n,
                         "consecutive_limit_ups": [0] * n})


def test_uniform_price_rescaling_does_not_change_market_inputs():
    from app.services.regime_dependencies import changed_dates
    before = history()
    after = before.with_columns(pl.col(c) * 0.8 for c in ("open", "high", "low", "close"))
    inst = pl.DataFrame({"symbol": ["000001.SZ"], "name": ["bank"]})
    assert changed_dates(before, after, inst) == []


def test_price_correction_includes_only_actual_following_indicator_changes():
    from app.services.regime_dependencies import changed_dates
    before = history()
    day = before["date"][25]
    after = before.with_columns(pl.when(pl.col("date") == day).then(100.).otherwise(pl.col("close")).alias("close"))
    inst = pl.DataFrame({"symbol": ["000001.SZ"], "name": ["bank"]})
    changed = changed_dates(before, after, inst)
    assert day in changed and before["date"][26] in changed
    assert before["date"][44] in changed
    assert all(day <= d <= before["date"][44] for d in changed)


def test_added_removed_and_amount_rows_are_changes():
    from app.services.regime_dependencies import changed_dates
    before = history()
    inst = pl.DataFrame({"symbol": ["000001.SZ"], "name": ["bank"]})
    assert changed_dates(before.head(-1), before, inst) == [before["date"][-1]]
    assert changed_dates(before, before.head(-1), inst) == [before["date"][-1]]
    after = before.with_columns(pl.when(pl.col("date") == before["date"][30]).then(20000.).otherwise(pl.col("amount")).alias("amount"))
    assert changed_dates(before, after, inst) == [before["date"][30]]


def test_mainline_explicit_dates_are_batched_without_unrequested_dates(tmp_path, monkeypatch):
    from app.services import market_mainline, regime_builder
    days = history(25)["date"].to_list()
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(regime_builder, "enriched_date_set", lambda _: set(days))
    seen = []
    def compute(repo, root, start, end, **kw):
        seen.append((start, end))
        return pl.DataFrame()
    monkeypatch.setattr(market_mainline, "compute_mainline_range", compute)
    market_mainline.compute_mainline_incremental(repo, tmp_path, today=days[-1], target_dates=days[:10] + days[-2:])
    assert seen == [(days[0], days[9]), (days[-2], days[-1])]


def test_market_history_explicit_manual_task_is_separate(tmp_path):
    from app.api.regime import regime_recompute
    from app.services import maintenance_queue
    path = tmp_path / "kline_daily_enriched/date=2026-01-05/part.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"date": [date(2026, 1, 5)]}).write_parquet(path)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(repo=repo)))
    regime_recompute(request)
    task = maintenance_queue._read(tmp_path)[0]
    assert task["options"] == {"force": True, "reason": "manual_recompute"}


def seed_scope(tmp_path, frame):
    from app.services import regime_builder
    for day, part in frame.partition_by("date", as_dict=True).items():
        path = tmp_path / f"kline_daily_enriched/date={day[0]}/part.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(path)
    state = {"legacy_mtime_ns": 0, "dates": {str(d): regime_builder._source_stamp(tmp_path, d) for d in frame["date"]}}
    regime_builder._save_source_state(tmp_path, state)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path),
        get_instruments=lambda: pl.DataFrame({"symbol": ["000001.SZ"], "name": ["bank"]}),
        get_historical_shares=lambda: pl.DataFrame())
    return repo


def test_scope_does_not_queue_all_dates_and_only_acknowledges_after_commit(tmp_path):
    from app.services import maintenance_queue, regime_builder
    from app.services.regime_dependencies import AdjustmentScope
    frame = history(35)
    repo = seed_scope(tmp_path, frame)
    before = regime_builder._source_state(tmp_path)
    scope = AdjustmentScope(repo, ["000001.SZ"])
    after = frame.with_columns(pl.col(c) * .8 for c in ("open", "high", "low", "close"))
    scope.before_publish(after)
    assert maintenance_queue.status(tmp_path) == []
    assert regime_builder._source_state(tmp_path) == before
    for key, part in after.partition_by("date", as_dict=True).items():
        part.write_parquet(tmp_path / f"kline_daily_enriched/date={key[0]}/part.parquet")
    # An unrelated index change cannot be acknowledged as a harmless adjustment.
    index = tmp_path / f"kline_index_daily/date={frame['date'][-1]}/part.parquet"
    index.parent.mkdir(parents=True)
    pl.DataFrame({"close": [123.]}).write_parquet(index)
    scope.committed()
    state = regime_builder._source_state(tmp_path)
    assert state["dates"][str(frame["date"][0])] == regime_builder._source_stamp(tmp_path, frame["date"][0])
    assert state["dates"][str(frame["date"][-1])] == before["dates"][str(frame["date"][-1])]


def test_actual_changes_queued_before_write_and_survive_interruption(tmp_path):
    from app.services import maintenance_queue
    from app.services.regime_dependencies import AdjustmentScope
    frame = history(35)
    repo = seed_scope(tmp_path, frame)
    scope = AdjustmentScope(repo, ["000001.SZ"])
    day = frame["date"][-1]
    after = frame.with_columns(pl.when(pl.col("date") == day).then(20000.).otherwise(pl.col("amount")).alias("amount"))
    scope.before_publish(after)
    maintenance_queue.recover(tmp_path)
    tasks = maintenance_queue._read(tmp_path)
    assert tasks[0]["items"] == [str(day)] and tasks[0]["total"] == 1
    assert tasks[0]["options"]["reason"] == "input_changed"
    assert pl.read_parquet(tmp_path / f"kline_daily_enriched/date={day}/part.parquet")["amount"][0] == 10000.


@pytest.mark.parametrize("identity_first", [True, False])
@pytest.mark.parametrize("boards_dtype", [pl.Int64, pl.UInt32])
def test_adjustment_scope_accepts_mixed_identity_columns_and_keeps_dependencies(tmp_path, identity_first, boards_dtype):
    from app.services import maintenance_queue
    from app.services.regime_dependencies import AdjustmentScope

    frame = history(35).with_columns(pl.col("consecutive_limit_ups").cast(boards_dtype))
    repo = seed_scope(tmp_path, frame)
    day = frame["date"][0 if identity_first else -1]
    path = tmp_path / f"kline_daily_enriched/date={day}/part.parquet"
    pl.read_parquet(path).with_columns(
        pl.lit("stock").alias("asset_type"), pl.lit("SZ").alias("exchange"),
    ).write_parquet(path)
    scope = AdjustmentScope(repo, ["000001.SZ"])
    assert scope.before.height == 35
    assert scope.before.sort("date")["raw_close"].to_list() == frame["raw_close"].to_list()
    changed = frame["date"][-1]
    scope.before_publish(frame.with_columns(
        pl.when(pl.col("date") == changed).then(20000.).otherwise(pl.col("amount")).alias("amount"),
    ))
    assert maintenance_queue._read(tmp_path)[0]["items"] == [str(changed)]


def test_identical_storage_does_not_change_file_or_generation(tmp_path):
    from app.indicators.pipeline import _publish_storage_if_changed
    frame = history(1)
    path = tmp_path / "part.parquet"
    frame.write_parquet(path)
    stamp = path.stat().st_mtime_ns
    publication = SimpleNamespace(write_parquet=lambda *a: pytest.fail("unchanged publication"))
    assert not _publish_storage_if_changed(publication, frame, path)
    assert path.stat().st_mtime_ns == stamp


def test_explicit_recovery_finishes_even_if_files_already_match(tmp_path, monkeypatch):
    from app.enriched_generation import EnrichedPublication, enriched_publication_incomplete
    from app.indicators import pipeline
    from test_enriched_full_rebuild import _fake_compute_enriched, _write_daily

    _write_daily(tmp_path, "2026-07-15", 15.0)
    monkeypatch.setattr(pipeline, "compute_enriched", _fake_compute_enriched)
    pipeline.run_pipeline(data_dir=tmp_path)
    out = tmp_path / "kline_daily_enriched/date=2026-07-15/part.parquet"
    interrupted = EnrichedPublication(tmp_path)
    interrupted.write_parquet(pl.read_parquet(out), out)
    del interrupted  # Simulate an interrupted owner, not another live writer.
    assert enriched_publication_incomplete(tmp_path, "stock")
    pipeline.run_pipeline(data_dir=tmp_path)
    assert not enriched_publication_incomplete(tmp_path, "stock")


def test_automatic_queue_rechecks_inputs_but_manual_recompute_does_not(tmp_path, monkeypatch):
    from app.services import daily_maintenance, regime_builder
    day = date(2026, 1, 5)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(regime_builder, "load_regime_history", lambda _: pl.DataFrame({"date": [day]}))
    monkeypatch.setattr(regime_builder, "detect_stale_dates", lambda *a: [])
    monkeypatch.setattr(regime_builder, "compute_regime_incremental", lambda *a, **kw: pytest.fail("must skip"))
    task = {"kind": "regime", "items": [str(day)], "label": "regime", "options": {"reason": "missing_result"}}
    result = daily_maintenance.execute(repo, None, task, lambda *a, **kw: None)
    assert result == {"rows": 0, "skipped_dates": 1, "skipped_items": [str(day)]}
    task["options"] = {"force": True, "reason": "manual_recompute"}
    with pytest.raises(pytest.fail.Exception, match="must skip"):
        daily_maintenance.execute(repo, None, task, lambda *a, **kw: None)


def test_scheduler_serves_daily_and_evening_before_background(monkeypatch, tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from app.jobs import daily_pipeline
    from app.services import close_data_sync, maintenance_queue, prediction_workflow, preferences

    class Scheduler:
        def __init__(self, **kw):
            self.jobs = {}
        def add_job(self, fn, **kw):
            self.jobs[kw["id"]] = fn
        def start(self):
            pass

    now = datetime(2026, 9, 9, 19, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(daily_pipeline, "AsyncIOScheduler", Scheduler)
    monkeypatch.setattr(preferences, "load", lambda: {})
    monkeypatch.setattr("app.market_time.cn_now", lambda: now)
    monkeypatch.setattr("app.services.trading_day.is_trading_day", lambda: True)
    monkeypatch.setattr(close_data_sync, "pending_tasks", lambda *a: [])
    monkeypatch.setattr(prediction_workflow, "due", lambda *a: False)
    seen = []
    monkeypatch.setattr(daily_pipeline, "_run_scheduled_incremental_sync", lambda *a, **k: seen.append(a[2]) or {})
    monkeypatch.setattr(maintenance_queue, "tick", lambda *a, **k: seen.append("background"))
    monkeypatch.setattr("app.services.mining_schedule.run_weekly_mining", lambda *a: None)
    def tracked(fn, label, on_started=None):
        if on_started:
            on_started()
        fn()
        return True
    monkeypatch.setattr(daily_pipeline, "_run_tracked", tracked)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    scheduler = daily_pipeline.start_scheduler(repo, None)
    for _ in range(3):
        scheduler.jobs["maintenance_queue"]()
    assert seen == ["after_close", "evening", "background"]


def test_mainline_batch_matches_single_days_and_does_not_read_outside_range(tmp_path, monkeypatch):
    from polars.testing import assert_frame_equal
    from test_market_mainline import TestComputeMainline

    from app.services import market_mainline
    repo, start, end = TestComputeMainline()._setup(tmp_path, monkeypatch)
    corrupt = tmp_path / "kline_daily_enriched/date=2020-01-01/part.parquet"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"unrelated invalid partition")
    cfg = {"filter_cfg": {"min_members": 4, "max_members": 600, "blacklist": []}}
    batch = market_mainline.compute_mainline_range(repo, tmp_path, start, end, **cfg)
    singles = pl.concat([market_mainline.compute_mainline_range(repo, tmp_path, d, d, **cfg) for d in (start, end)])
    assert_frame_equal(batch.sort(["date", "member"]), singles.sort(["date", "member"]))
