from datetime import date

import pytest

from app.services import minute_repair_authorization as auth


DAY = date(2026, 9, 17)


def test_grant_survives_reload_and_is_scoped_to_task_source_operation_stock_day(tmp_path):
    grant = auth.grant(tmp_path, "task-1", "qmt", [("000001.SZ", DAY)])
    assert auth.allowed(tmp_path, "task-1", "qmt", "000001.SZ", DAY) == grant
    for task, source, symbol, day in [
        ("task-2", "qmt", "000001.SZ", DAY),
        ("task-1", "tdx", "000001.SZ", DAY),
        ("task-1", "qmt", "000002.SZ", DAY),
        ("task-1", "qmt", "000001.SZ", date(2026, 9, 18)),
    ]:
        assert auth.allowed(tmp_path, task, source, symbol, day) is None
    assert auth.allowed(tmp_path, "task-1", "qmt", "000001.SZ", DAY,
                        operation="replace_daily") is None
    assert auth.grant(tmp_path, "task-1", "qmt", [("000001.SZ", DAY)]) == grant
    assert len(auth.snapshot(tmp_path)["audit"]) == 1


def test_pending_conflict_approval_is_exact_idempotent_and_not_a_wildcard(tmp_path):
    request = auth.request_replacement(tmp_path, "task-1", "qmt", "000001.SZ", DAY)
    assert auth.request_replacement(tmp_path, "task-1", "qmt", "000001.SZ", DAY) == request
    assert len(auth.snapshot(tmp_path)["pending"]) == 1
    auth.approve(tmp_path, [request])
    auth.approve(tmp_path, [request])
    assert auth.allowed(tmp_path, "task-1", "qmt", "000001.SZ", DAY)
    assert auth.snapshot(tmp_path)["pending"] == []
    assert auth.allowed(tmp_path, "task-1", "qmt", "000002.SZ", DAY) is None
    with pytest.raises(ValueError):
        auth.approve(tmp_path, [request, "nonexistent"])


def test_scope_registration_is_per_task_and_operation(tmp_path):
    assert auth.open_scope(tmp_path, "task-1", "qmt") is True
    assert auth.open_scope(tmp_path, "task-1", "qmt") is False
    assert auth.open_scope(tmp_path, "task-1", "tdx") is False
    assert auth.open_scope(tmp_path, "task-1", "qmt", operation="replace_daily") is True


def test_identity_conflicts_cannot_be_authorized(tmp_path):
    for symbol in ("000001.SH", "000001", "unknown", "920001.SZ"):
        with pytest.raises(ValueError):
            auth.grant(tmp_path, "task-1", "qmt", [(symbol, DAY)])


def test_audit_records_reuse_and_backup_without_renewing_grant(tmp_path):
    grant = auth.grant(tmp_path, "task-1", "qmt", [("000001.SZ", DAY)])
    auth.used(tmp_path, grant, "000001.SZ", DAY, backup="import_backups/batch/date=2026-09-17/part.parquet")
    snapshot = auth.snapshot(tmp_path)
    assert {r["event"] for r in snapshot["audit"]} == {"granted", "used"}
    assert auth.allowed(tmp_path, "task-1", "qmt", "000001.SZ", DAY) == grant


def test_retry_reuses_grant_but_new_source_and_new_stock_require_approval(tmp_path):
    from threading import RLock
    from types import SimpleNamespace

    import polars as pl
    from test_minute_repair import DAY as sample_day, bars, reference

    from app.services.minute_repair import repair_minute_gaps

    minute = tmp_path / f"kline_minute/date={sample_day}/part.parquet"
    daily = tmp_path / f"kline_daily/date={sample_day}/part.parquet"
    minute.parent.mkdir(parents=True)
    daily.parent.mkdir(parents=True)
    bad = bars().head(10).with_columns(pl.lit(9.0).alias("close"))
    bad.write_parquet(minute)
    pl.DataFrame([reference()]).write_parquet(daily)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), _write_lock=RLock(),
                           rebuild_views=lambda: None, clear_cache=lambda: None)
    source = SimpleNamespace(get_minute=lambda *a, **k: bars())
    args = dict(start_date=sample_day, end_date=sample_day, symbols=["000001.SZ"],
                task_id="task-1", source_provider=source, bypass_failover=True)
    first = repair_minute_gaps(repo, provider_name="qmt", **args)
    assert first["approval_required"] and pl.read_parquet(minute).equals(bad)
    auth.approve(tmp_path, [first["approval_required"][0]["id"]])
    retried = repair_minute_gaps(repo, provider_name="qmt", **args)
    assert retried["repaired_stock_days"] == 1
    assert pl.read_parquet(minute).select(bars().columns).equals(bars())
    assert pl.read_parquet(minute)["security_key"].unique().to_list() == ["stock|SZ|000001"]
    assert list((tmp_path / "import_backups").glob("*/date=*/part.parquet"))
    assert any(e["event"] == "used" for e in auth.snapshot(tmp_path)["audit"])
    bad.write_parquet(minute)
    changed_source = repair_minute_gaps(repo, provider_name="tdx", **args)
    assert changed_source["repaired_stock_days"] == 0
    assert changed_source["approval_required"][0]["source"] == "tdx"


def test_qmt_sparse_import_is_auto_and_bad_data_still_rejected_with_grant(tmp_path):
    from threading import RLock
    from types import SimpleNamespace

    import polars as pl
    from test_minute_repair import DAY as sample_day, bars, reference

    from app.services.minute_repair import repair_minute_gaps

    daily = tmp_path / f"kline_daily/date={sample_day}/part.parquet"
    daily.parent.mkdir(parents=True)
    pl.DataFrame([reference() | {"volume": 239.0, "amount": 239000.0}]).write_parquet(daily)
    # Omit a no-trade minute inside the session, not the closing timestamp.
    sparse = bars().filter(pl.col("datetime") != bars()["datetime"][50])
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), _write_lock=RLock(),
                           rebuild_views=lambda: None, clear_cache=lambda: None)
    source = SimpleNamespace(get_minute=lambda *a, **k: sparse)
    result = repair_minute_gaps(repo, start_date=sample_day, end_date=sample_day,
                               provider_name="qmt", source_provider=source,
                               task_id="sparse", bypass_failover=True)
    assert result["added_rows"] == 239
    assert result["approval_required"] == []
    assert auth.snapshot(tmp_path)["pending"] == []
