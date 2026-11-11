import json
from datetime import date, datetime, timedelta
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest

from app.services import minute_failover as mf

DAY = date(2026, 9, 7)
NOW = datetime(2026, 9, 8, 19)


def bars(code="000701", day=DAY):
    stamps = [datetime.combine(day, datetime.min.time()).replace(hour=9, minute=31) + timedelta(minutes=i) for i in range(120)]
    stamps += [datetime.combine(day, datetime.min.time()).replace(hour=13, minute=1) + timedelta(minutes=i) for i in range(120)]
    return pl.DataFrame({"symbol": [code] * 240, "datetime": stamps,
                         "open": [10.] * 240, "high": [10.] * 240, "low": [10.] * 240,
                         "close": [10.] * 240, "volume": [1.] * 240, "amount": [1000.] * 240})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(mf, "cn_now", lambda: NOW)
    monkeypatch.setattr(mf.settings, "minute_fallback_provider", "tdx")
    monkeypatch.setattr(mf.settings, "data_dir", tmp_path)
    monkeypatch.setattr(mf.settings, "stockdb_enabled", False)
    monkeypatch.setattr(mf.settings, "stockdb_minute_fallback_enabled", False)
    monkeypatch.setattr(mf.settings, "stockdb_after_close_primary_enabled", False)
    monkeypatch.setattr(mf.settings, "stockdb_readiness_sample_size", 10)
    monkeypatch.setattr(mf.settings, "stockdb_readiness_min_coverage", 0.6)
    monkeypatch.setattr(mf.settings, "stockdb_publication_retry_seconds", 600)
    # Keep the legacy route tests deterministic. Individual QMT route tests
    # opt in explicitly below.
    monkeypatch.setattr(mf, "resolve_qmt_backup", lambda: None)
    monkeypatch.setattr(mf, "resolve_promax_backup", lambda: None)
    instruments = tmp_path / "instruments"
    instruments.mkdir()
    pl.DataFrame({"symbol": ["000701.SZ", "000757.SZ"], "code": ["000701", "000757"]}).write_parquet(
        instruments / "instruments.parquet"
    )
    for day in (DAY, DAY + timedelta(days=1)):
        path = tmp_path / f"kline_daily/date={day}/part.parquet"
        path.parent.mkdir(parents=True)
        pl.DataFrame([{"symbol": s, "date": day, "open": 10., "high": 10., "low": 10.,
                       "close": 10., "volume": 240., "amount": 240000.}
                      for s in ("000701", "000757")]).write_parquet(path)
    backup = SimpleNamespace(get_minute=Mock(return_value=bars()))
    monkeypatch.setattr(mf, "resolve_backup", lambda: backup)
    return tmp_path, backup


def fetch(primary, symbols=None, end=None):
    return mf.fetch_minutes(primary, symbols or ["000701"], datetime(2026, 9, 7, 9, 30),
                            end or datetime(2026, 9, 7, 15), asset_type="stock", freq="1m")


def source(frame=None, issues=None):
    return SimpleNamespace(get_minute_diagnostics=Mock(return_value=(
        frame if frame is not None else pl.DataFrame(), issues or [])))


def test_good_primary_never_calls_backup(setup):
    _, backup = setup
    result, _ = fetch(source(bars()))
    assert result.equals(bars())
    backup.get_minute.assert_not_called()


def test_closed_repair_uses_qmt_before_online_fallback(setup, monkeypatch):
    _, backup = setup
    qmt = source(bars())
    monkeypatch.setattr(mf, "resolve_qmt_backup", lambda: qmt)
    result, issues = fetch(source())
    assert result.height == 240
    qmt.get_minute_diagnostics.assert_called_once()
    backup.get_minute.assert_not_called()
    assert not any(item.get("source") == "tdx" for item in issues)


def test_invalid_qmt_rows_continue_to_online_fallback(setup, monkeypatch):
    _, backup = setup
    qmt = source(bars().with_columns(pl.lit(11.).alias("high")))
    monkeypatch.setattr(mf, "resolve_qmt_backup", lambda: qmt)
    monkeypatch.setattr(mf, "resolve_promax_backup", lambda: None)
    result, issues = fetch(source())
    assert result.height == 240
    qmt.get_minute_diagnostics.assert_called_once()
    assert backup.get_minute.call_count == 1
    assert any(item.get("source") == "qmt" and item.get("reason") == "daily_price_conflict" for item in issues)
    assert any(item.get("source") == "tdx" and item.get("reason") == "accepted" for item in issues)


def test_after_close_stockdb_primary_skips_online_sources(setup, monkeypatch):
    _, backup = setup
    monkeypatch.setattr(mf.settings, "stockdb_enabled", True)
    monkeypatch.setattr(mf.settings, "stockdb_minute_fallback_enabled", True)
    monkeypatch.setattr(mf.settings, "stockdb_after_close_primary_enabled", True)
    local = source(bars())
    monkeypatch.setattr(mf, "resolve_stockdb_backup", lambda: local)
    primary = source(bars())
    result, _ = fetch(primary)
    assert result.equals(bars())
    local.get_minute_diagnostics.assert_called_once()
    primary.get_minute_diagnostics.assert_not_called()
    backup.get_minute.assert_not_called()


def test_after_close_stockdb_failure_falls_back_online(setup, monkeypatch):
    _, backup = setup
    monkeypatch.setattr(mf.settings, "stockdb_enabled", True)
    monkeypatch.setattr(mf.settings, "stockdb_minute_fallback_enabled", True)
    monkeypatch.setattr(mf.settings, "stockdb_after_close_primary_enabled", True)
    bad = bars().with_columns(pl.lit(11.).alias("high"))
    monkeypatch.setattr(mf, "resolve_stockdb_backup", lambda: source(bad))
    primary = source(bars())
    result, issues = fetch(primary)
    assert result.equals(bars())
    primary.get_minute_diagnostics.assert_called_once()
    backup.get_minute.assert_not_called()
    assert any(i.get("source") == "stockdb" and i["reason"] == "daily_price_conflict" for i in issues)


def test_after_close_unpublished_snapshot_uses_bounded_online_fallback(setup, monkeypatch):
    root, backup = setup
    monkeypatch.setattr(mf.settings, "stockdb_enabled", True)
    monkeypatch.setattr(mf.settings, "stockdb_minute_fallback_enabled", True)
    monkeypatch.setattr(mf.settings, "stockdb_after_close_primary_enabled", True)
    monkeypatch.setattr(mf.settings, "stockdb_readiness_sample_size", 2)
    monkeypatch.setattr(mf.settings, "stockdb_online_fallback_symbols", 1)
    monkeypatch.setattr(mf, "resolve_stockdb_backup", lambda: source())
    primary = source(bars())
    result, issues = fetch(primary, ["000701", "000757"])
    assert result.height == 240
    primary.get_minute_diagnostics.assert_called_once()
    backup.get_minute.assert_not_called()
    assert all(i["reason"] == "source_pending" for i in issues)
    freshness = json.loads((root / "user_data/stockdb_freshness.json").read_text(encoding="utf-8"))
    assert freshness["latest"]["date"] == str(DAY)
    assert freshness["latest"]["ready"] is False
    journal = max((root / "import_reports/minute_failover").rglob("*.json"), key=lambda p: p.stat().st_mtime_ns)
    stats = json.loads(journal.read_text(encoding="utf-8"))["local_primary"]
    assert stats["publication_pending"] and stats["online_fallback_symbols"] == 1
    assert stats["deferred_online_symbols"] == 1
    mf._save_source(root, "stockdb", {"reason": "source_pending", "cause": "stockdb_snapshot_stale",
                                      "retryable": True, "retry_after_seconds": 600})
    assert mf._deferred(root, "stockdb") is None


def test_after_close_online_fallback_is_bounded_and_rotatable(setup, monkeypatch):
    root, backup = setup
    monkeypatch.setattr(mf.settings, "stockdb_enabled", True)
    monkeypatch.setattr(mf.settings, "stockdb_minute_fallback_enabled", True)
    monkeypatch.setattr(mf.settings, "stockdb_after_close_primary_enabled", True)
    monkeypatch.setattr(mf.settings, "stockdb_online_fallback_symbols", 1)
    monkeypatch.setattr(mf.settings, "stockdb_readiness_min_coverage", 0.3)
    path = root / f"kline_daily/date={DAY}/part.parquet"
    daily = pl.read_parquet(path)
    daily.vstack(pl.DataFrame([{"symbol": "000702", "date": DAY, "open": 10., "high": 10.,
                                "low": 10., "close": 10., "volume": 240., "amount": 240000.}])).write_parquet(path)
    monkeypatch.setattr(mf, "resolve_stockdb_backup", lambda: source(bars("000701")))
    primary = source(bars("000757"))
    result, _ = fetch(primary, ["000701", "000757", "000702"])
    assert result.height == 480
    assert primary.get_minute_diagnostics.call_args.args[0] == ["000757"]
    backup.get_minute.assert_not_called()
    journal = max((root / "import_reports/minute_failover").rglob("*.json"), key=lambda p: p.stat().st_mtime_ns)
    route = json.loads(journal.read_text(encoding="utf-8"))
    assert route["local_primary"]["deferred_online_symbols"] == 1


def test_lunch_retry_after_close_validates_requested_window_not_full_day(setup):
    _, backup = setup
    backup.get_minute.return_value = bars().head(120)
    result, issues = fetch(source(), end=datetime(2026, 9, 7, 11, 30))
    assert result.height == 120
    assert any(i["source"] == "tdx" and i["reason"] == "accepted" for i in issues)
    assert backup.get_minute.call_args.args[2] == datetime(2026, 9, 7, 11, 30)


@pytest.mark.parametrize("reason", ["source_pending", "rate_limited", "request_timeout", "source_empty"])
def test_failed_primary_uses_verified_backup(setup, reason):
    root, backup = setup
    result, issues = fetch(source(issues=[{"reason": reason, "retryable": True}]))
    assert result.height == 240
    assert any(i.get("source") == "tdx" and i["reason"] == "accepted" for i in issues)
    assert backup.get_minute.call_count == 1
    assert list((root / "import_reports/minute_failover").rglob("*.json"))
    assert not (root / "kline_minute").exists()


def test_partial_batch_only_queries_missing_stock(setup):
    _, backup = setup
    backup.get_minute.return_value = bars("000757")
    result, _ = fetch(source(bars()), ["000701", "000757"])
    assert result.height == 480
    assert backup.get_minute.call_args.args[0] == ["000757"]


def test_missing_date_in_range_falls_back_even_with_nonempty_primary(setup):
    _, backup = setup
    result, _ = fetch(source(bars(day=DAY + timedelta(days=1))), end=datetime(2026, 9, 8, 15))
    assert result.height == 480
    assert backup.get_minute.call_args.args[1].date() == DAY
    assert backup.get_minute.call_args.args[2].date() == DAY


@pytest.mark.parametrize("bad", ["price", "volume", "incomplete", "wrong_symbol", "wrong_date"])
def test_bad_backup_is_never_accepted(setup, bad):
    _, backup = setup
    frame = bars()
    if bad == "price":
        frame = frame.with_columns(pl.lit(11.).alias("high"))
    elif bad == "volume":
        frame = frame.with_columns((pl.col("volume") * 2).alias("volume"))
    elif bad == "incomplete":
        frame = frame.head(239)
    elif bad == "wrong_symbol":
        frame = bars("600000")
    else:
        frame = bars(day=DAY - timedelta(days=1))
    backup.get_minute.return_value = frame
    result, issues = fetch(source())
    assert result.is_empty()
    assert not any(i.get("source") == "tdx" and i["reason"] == "accepted" for i in issues)


def test_conflicting_old_records_remain_protected(setup):
    root, _ = setup
    old = bars().head(10).with_columns(pl.lit(9.).alias("close"))
    path = root / f"kline_minute/date={DAY}/part.parquet"
    path.parent.mkdir(parents=True)
    old.write_parquet(path)
    before = path.read_bytes()
    result, issues = fetch(source())
    assert result.is_empty()
    assert any(i["reason"] == "existing_values_conflict" for i in issues)
    assert path.read_bytes() == before


def test_missing_reference_never_trusts_backup(setup):
    _, backup = setup
    result, issues = fetch(source(), ["600000"])
    assert result.is_empty()
    backup.get_minute.assert_not_called()
    assert any(i["reason"] == "daily_reference_missing" for i in issues)


def test_backup_unavailable_preserves_valid_primary(setup, monkeypatch):
    monkeypatch.setattr(mf, "resolve_backup", lambda: None)
    result, issues = fetch(source(bars()), ["000701", "000757"])
    assert result.equals(bars())
    assert any(i["reason"] == "backup_unavailable" for i in issues)


def test_intraday_requires_fresh_complete_requested_minutes(setup, monkeypatch):
    _, backup = setup
    monkeypatch.setattr(mf, "cn_now", lambda: datetime(2026, 9, 7, 10, 1))
    backup.get_minute.return_value = bars().head(30)
    result, issues = fetch(source(), end=datetime(2026, 9, 7, 10))
    assert result.height == 30
    assert any(i.get("source") == "tdx" and i["reason"] == "accepted" for i in issues)
    backup.get_minute.return_value = bars().head(29)
    result, _ = fetch(source(), end=datetime(2026, 9, 7, 10))
    assert result.height == 29


def test_intraday_tail_can_be_saved_without_claiming_completion(setup, monkeypatch):
    _, backup = setup
    monkeypatch.setattr(mf, "cn_now", lambda: datetime(2026, 9, 7, 13, 40))
    backup.get_minute.return_value = bars().head(119)
    result, issues = fetch(source(), end=datetime(2026, 9, 7, 11, 30))
    assert result.height == 119
    issue = next(i for i in issues if i["source"] == "tdx")
    assert issue["reason"] == "source_incomplete"
    assert issue["rows"] == 119
    assert issue["missing_minutes"] == 1
    assert issue["missing_times"] == ["11:30"]


def test_intraday_interior_gap_is_not_hidden_by_later_rows(setup, monkeypatch):
    _, backup = setup
    monkeypatch.setattr(mf, "cn_now", lambda: datetime(2026, 9, 7, 13, 40))
    backup.get_minute.return_value = bars().head(150).filter(
        pl.col("datetime") != datetime(2026, 9, 7, 11, 30))
    result, _ = fetch(source(), end=datetime(2026, 9, 7, 13, 30))
    assert result.height == 119
    assert result["datetime"].max() == datetime(2026, 9, 7, 11, 29)


def test_incomplete_prefix_cannot_replace_conflicting_old_values(setup, monkeypatch):
    root, backup = setup
    monkeypatch.setattr(mf, "cn_now", lambda: datetime(2026, 9, 7, 13, 40))
    backup.get_minute.return_value = bars().head(119)
    path = root / f"kline_minute/date={DAY}/part.parquet"
    path.parent.mkdir(parents=True)
    old = bars().head(2).with_columns(pl.lit(9.).alias("close"))
    old.write_parquet(path)
    result, issues = fetch(source(), end=datetime(2026, 9, 7, 11, 30))
    assert result.is_empty()
    assert issues[-1]["reason"] == "existing_values_conflict"
    assert pl.read_parquet(path).equals(old)


def test_no_expected_minutes_is_not_a_source_failure(setup, monkeypatch):
    _, backup = setup
    monkeypatch.setattr(mf, "cn_now", lambda: datetime(2026, 9, 7, 12))
    result, issues = mf.fetch_minutes(source(), ["000701"], datetime(2026, 9, 7, 11, 31),
                                      datetime(2026, 9, 7, 11, 45))
    assert result.is_empty() and not issues
    backup.get_minute.assert_not_called()


def test_unrelated_providers_and_disabled_config_are_unchanged(setup, monkeypatch):
    provider = SimpleNamespace(_connection_mode=lambda: "promax")
    assert mf.enabled("tushare", provider)
    assert not mf.enabled("tdx", provider)
    assert not mf.enabled("tushare", SimpleNamespace(_connection_mode=lambda: "official"))
    monkeypatch.setattr(mf.settings, "minute_fallback_provider", "")
    assert not mf.enabled("tushare", provider)


def test_primary_incomplete_or_wrong_price_is_replaced_as_a_whole_day(setup):
    for frame in (bars().head(239), bars().with_columns(pl.lit(11.).alias("high")), pl.DataFrame({"bad": [1]})):
        result, _ = fetch(source(frame))
        assert result.equals(bars())


def test_older_history_does_not_trigger_unbounded_tdx_requests(setup):
    _, backup = setup
    result, _ = mf.fetch_minutes(source(), ["000701"], datetime(2025, 1, 2, 9, 30), datetime(2025, 1, 2, 15))
    assert result.is_empty()
    backup.get_minute.assert_not_called()


def test_timezone_is_beijing_and_future_rows_are_excluded(setup):
    from zoneinfo import ZoneInfo
    utc = bars().with_columns(pl.col("datetime").dt.replace_time_zone("Asia/Shanghai").dt.convert_time_zone("UTC"))
    result, _ = mf.fetch_minutes(source(utc), ["000701"], datetime(2026, 9, 7, 1, 30, tzinfo=ZoneInfo("UTC")),
                                 datetime(2026, 9, 7, 7, tzinfo=ZoneInfo("UTC")))
    assert result.equals(bars())


def route(monkeypatch):
    from app.services import kline_sync as ks
    primary = source()
    primary._connection_mode = lambda: "promax"
    monkeypatch.setattr(ks, "_resolve_minute_provider", lambda name: (primary, False, None))
    monkeypatch.setattr(ks.preferences, "get_minute_data_provider", lambda: "tushare")
    return ks, primary


def test_batch_route_uses_backup_and_both_sources_failure_propagates(setup, monkeypatch):
    _, backup = setup
    ks, _ = route(monkeypatch)
    frame, fallback = ks._try_custom_minute_batched(["000701"], datetime(2026, 9, 7, 9, 30),
                                                  datetime(2026, 9, 7, 15), "stock")
    assert fallback is False and frame.height == 240
    backup.get_minute.return_value = pl.DataFrame()
    with pytest.raises(mf.MinuteSourcesUnavailableError):
        ks._try_custom_minute_batched(["000701"], datetime(2026, 9, 7, 9, 30), datetime(2026, 9, 7, 15), "stock")


def test_invalid_first_batch_does_not_starve_good_later_batch(setup, monkeypatch):
    _, backup = setup
    ks, _ = route(monkeypatch)
    backup.get_minute.side_effect = [bars().head(20), bars("000757")]
    frame, fallback = ks._try_custom_minute_batched(
        ["000701", "000757"], datetime(2026, 9, 7, 9, 30), datetime(2026, 9, 7, 15),
        "stock", batch_size=1)
    assert not fallback and frame.height == 240
    assert frame["symbol"].unique().to_list() == ["000757"]
    assert backup.get_minute.call_count == 2


def test_streamed_valid_batch_is_not_reported_as_all_sources_failed(setup, monkeypatch):
    _, backup = setup
    ks, _ = route(monkeypatch)
    backup.get_minute.side_effect = [bars().head(20), bars("000757")]
    persist = Mock()
    frame, fallback = ks._try_custom_minute_batched(
        ["000701", "000757"], datetime(2026, 9, 7, 9, 30), datetime(2026, 9, 7, 15),
        "stock", batch_size=1, on_segment=persist)
    assert not fallback and frame.is_empty()
    persist.assert_called_once()
    assert persist.call_args.args[0].height == 240


def test_empty_closed_group_does_not_block_later_increment_group(setup, monkeypatch):
    root, backup = setup
    ks, _ = route(monkeypatch)
    monkeypatch.setattr(ks, "_cleanup_null_datetime_minute", lambda repo: None)
    monkeypatch.setattr(ks, "_migrate_symbol_to_date_partition", lambda repo: None)
    monkeypatch.setattr(ks, "_latest_minute_datetime_by_symbol", lambda *a, **kw: {})
    monkeypatch.setattr(ks, "_closed_minute_gap_starts", lambda *a, **kw: {})
    monkeypatch.setattr(ks, "_minute_increment_groups", lambda *a, **kw: [
        (["000701"], datetime(2026, 9, 7, 9, 30), "first"),
        (["000757"], datetime(2026, 9, 7, 9, 31), "second")])
    monkeypatch.setattr(ks, "resolve_limit", lambda *a, **kw: SimpleNamespace(batch=10, rpm=None))
    monkeypatch.setattr(ks.preferences, "get_minute_sync_segment_days", lambda: 20)
    backup.get_minute.side_effect = [bars().head(20), bars("000757")]
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(), db=Mock())
    written = ks.sync_and_persist_minute(["000701", "000757"], repo, Mock(), days=1,
                                        as_of=datetime(2026, 9, 7, 15))
    assert written == 240
    frame = pl.read_parquet(root / f"kline_minute/date={DAY}/part.parquet")
    assert frame["symbol"].unique().to_list() == ["000757.SZ"]


def test_failure_message_distinguishes_data_conflict_from_network():
    issues = [{"source": "promax", "symbol": "603215", "date": "2026-09-08",
               "reason": "daily_turnover_conflict"},
              {"source": "tdx", "symbol": "603215", "date": "2026-09-08",
               "reason": "existing_values_conflict"}]
    error = mf.MinuteSourcesUnavailableError(issues)
    assert error.reason == "minute_validation_failed"
    assert not error.retryable
    assert "603215" in str(error) and "2026-09-08" in str(error)
    assert "existing_values_conflict" in str(error)


def test_network_outage_still_fails_fast(setup, monkeypatch):
    ks, _ = route(monkeypatch)
    call = Mock(return_value=(pl.DataFrame(), [
        {"source": "promax", "reason": "rate_limited"},
        {"source": "tdx", "reason": "upstream_unavailable"}]))
    monkeypatch.setattr(mf, "fetch_minutes", call)
    with pytest.raises(mf.MinuteSourcesUnavailableError) as caught:
        ks._try_custom_minute_batched(["000701", "000757"], datetime(2026, 9, 7, 9, 30),
                                     datetime(2026, 9, 7, 15), "stock", batch_size=1)
    assert caught.value.reason == "upstream_unavailable"
    assert call.call_count == 1


def test_closed_failure_evidence_keeps_actual_source_totals(setup):
    _, backup = setup
    backup.get_minute.return_value = bars().with_columns((pl.col("volume") * 1.1).alias("volume"))
    result, issues = fetch(source())
    assert result.is_empty()
    issue = issues[-1]
    assert issue["source_rows"] == 240 and issue["rows"] == 0
    assert issue["missing_minutes"] == 0
    assert issue["source_volume"] == pytest.approx(264)
    assert issue["reference_volume"] == 240


def test_single_route_uses_same_backup(setup, monkeypatch):
    ks, _ = route(monkeypatch)
    frame, fallback = ks._try_custom_minute(["000701"], datetime(2026, 9, 7, 9, 30),
                                          datetime(2026, 9, 7, 15), "stock")
    assert not fallback and frame.height == 240


def test_single_chart_cannot_bypass_failed_backup_validation(setup, monkeypatch):
    _, backup = setup
    ks, _ = route(monkeypatch)
    monkeypatch.setattr(ks, "cn_today", lambda: DAY)
    monkeypatch.setattr(ks.preferences, "get_realtime_data_provider", lambda: "fuyao")
    unchecked = Mock(side_effect=AssertionError("unchecked third-source fallback"))
    monkeypatch.setattr(ks, "get_client", unchecked)
    backup.get_minute.return_value = bars().head(20)
    assert ks.fetch_minute_single("000701", DAY).is_empty()
    assert backup.get_minute.call_count == 1
    unchecked.assert_not_called()


def test_verified_repair_uses_fallback_with_backup_and_idempotency(setup, monkeypatch):
    from app.services.minute_repair import repair_minute_gaps
    root, backup = setup
    _, primary = route(monkeypatch)
    path = root / f"kline_minute/date={DAY}/part.parquet"
    path.parent.mkdir(parents=True)
    old = bars().head(10)
    old.write_parquet(path)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                           rebuild_views=Mock(), clear_cache=Mock())
    result = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="tushare",
                               source_provider=primary, symbols=["000701"])
    assert result["complete"] and result["added_rows"] == 230
    assert pl.read_parquet(path).select(bars().columns).equals(bars("000701.SZ"))
    assert list((root / "import_backups").rglob("part.parquet"))
    assert repo.clear_cache.called and repo.rebuild_views.called
    count = backup.get_minute.call_count
    result = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="tushare",
                               source_provider=primary, symbols=["000701"])
    assert result["added_rows"] == 0 and backup.get_minute.call_count == count


def test_authorized_replacement_still_uses_transaction_backup(setup, monkeypatch):
    from app.services.minute_repair import repair_minute_gaps
    root, _ = setup
    _, primary = route(monkeypatch)
    path = root / f"kline_minute/date={DAY}/part.parquet"
    path.parent.mkdir(parents=True)
    old = bars().head(10).with_columns(pl.lit(9.).alias("close"))
    old.write_parquet(path)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                           rebuild_views=Mock(), clear_cache=Mock())
    result = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="tushare",
                               source_provider=primary, symbols=["000701"], repair_conflicts=True)
    assert result["complete"] and result["replaced_rows"] == 10
    assert pl.read_parquet(path).select(bars().columns).equals(bars("000701.SZ"))
    assert pl.read_parquet(next((root / "import_backups").rglob("part.parquet"))).equals(old)


def test_scheduled_increment_persists_backup_and_skips_complete_day(setup, monkeypatch):
    root, backup = setup
    ks, _ = route(monkeypatch)
    monkeypatch.setattr(ks, "_cleanup_null_datetime_minute", lambda repo: None)
    monkeypatch.setattr(ks, "_migrate_symbol_to_date_partition", lambda repo: None)
    monkeypatch.setattr(ks, "resolve_limit", lambda *a, **kw: SimpleNamespace(batch=10, rpm=None))
    monkeypatch.setattr(ks.preferences, "get_minute_sync_segment_days", lambda: 20)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(), db=Mock())
    written = ks.sync_and_persist_minute(["000701"], repo, Mock(), days=1, as_of=datetime(2026, 9, 7, 15))
    assert written == 240
    assert pl.read_parquet(root / f"kline_minute/date={DAY}/part.parquet").select(bars().columns).equals(bars("000701.SZ"))
    calls = backup.get_minute.call_count
    assert ks.sync_and_persist_minute(["000701"], repo, Mock(), days=1, as_of=datetime(2026, 9, 7, 15)) == 0
    assert backup.get_minute.call_count == calls


def local_source(monkeypatch, frame):
    monkeypatch.setattr(mf.settings, "stockdb_enabled", True)
    monkeypatch.setattr(mf.settings, "stockdb_minute_fallback_enabled", True)
    backup = source(frame)
    monkeypatch.setattr(mf, "resolve_stockdb_backup", lambda: backup)
    return backup


def test_stockdb_only_after_both_sources_fail(setup, monkeypatch):
    _, tdx = setup
    local = local_source(monkeypatch, bars())
    fetch(source(bars()))
    local.get_minute_diagnostics.assert_not_called()
    fetch(source())
    local.get_minute_diagnostics.assert_not_called()
    tdx.get_minute.return_value = pl.DataFrame()
    result, issues = fetch(source())
    assert result.equals(bars())
    assert issues[-1]["source"] == "stockdb" and issues[-1]["reason"] == "accepted"
    assert local.get_minute_diagnostics.call_count == 1


@pytest.mark.parametrize("bad", ["price", "incomplete", "volume", "existing"])
def test_stockdb_obeys_daily_and_old_record_validation(setup, monkeypatch, bad):
    root, tdx = setup
    tdx.get_minute.return_value = pl.DataFrame()
    frame = bars()
    if bad == "price":
        frame = frame.with_columns(pl.lit(11.).alias("high"))
    elif bad == "incomplete":
        frame = frame.head(239)
    elif bad == "volume":
        frame = frame.with_columns((pl.col("volume") * 2).alias("volume"))
    else:
        path = root / f"kline_minute/date={DAY}/part.parquet"
        path.parent.mkdir(parents=True)
        bars().head(3).with_columns(pl.lit(9.).alias("open")).write_parquet(path)
    local_source(monkeypatch, frame)
    result, issues = fetch(source())
    assert result.is_empty() and issues[-1]["source"] == "stockdb"
    assert issues[-1]["reason"] != "accepted"
    if bad == "existing":
        assert issues[-1]["reason"] == "existing_values_conflict"
        assert pl.read_parquet(path)["open"].to_list() == [9.] * 3


def test_stockdb_not_intraday_and_source_cooldowns_are_independent(setup, monkeypatch):
    root, tdx = setup
    tdx.get_minute.return_value = pl.DataFrame()
    local = local_source(monkeypatch, bars())
    fetch(source(), end=datetime(2026, 9, 7, 11, 30))
    local.get_minute_diagnostics.assert_not_called()
    for name in ("promax", "tdx"):
        mf._save_source(root, name, {"reason": "request_timeout"})
    assert mf.retry_state(root)["source_retry"]
    assert not mf.retry_state(root, include_local=True)["source_retry"]
    result, issues = fetch(source())
    assert result.height == 240 and issues[-1]["source"] == "stockdb"


def test_stockdb_network_failure_backs_off(setup, monkeypatch):
    root, tdx = setup
    tdx.get_minute.return_value = pl.DataFrame()
    local = local_source(monkeypatch, pl.DataFrame())
    local.get_minute_diagnostics.return_value = (pl.DataFrame(), [{"reason": "request_timeout"}])
    fetch(source())
    fetch(source())
    assert local.get_minute_diagnostics.call_count == 1
    assert mf._deferred(root, "stockdb")["retry_after_seconds"] > 0
