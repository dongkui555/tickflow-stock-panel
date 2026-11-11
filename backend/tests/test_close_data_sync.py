from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl
import pytest

from app.plugins.tushare.provider import TushareProvider
from app.services import close_data_sync as sync

DAY = date(2026, 9, 7)
NOW = datetime(2026, 9, 7, 19, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
SCHEDULE = {"hour": 15, "minute": 30}


def write_day(root, table, day, symbols, actual_day=None):
    folder = root / table / f"date={day}"
    folder.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"symbol": symbols, "date": [actual_day or day] * len(symbols)}).write_parquet(folder / "data.parquet")


def test_date_values_not_folder_or_written_row_count_determine_coverage(tmp_path):
    write_day(tmp_path, "kline_daily", DAY, ["000001.SZ", "600000.SH"])
    write_day(tmp_path, "moneyflow", DAY, ["000001", "600000"], DAY - timedelta(days=3))
    result = sync.coverage(tmp_path, "moneyflow", DAY)
    assert result["actual_symbols"] == 0
    assert not result["complete"]
    write_day(tmp_path, "moneyflow", DAY, ["000001"])
    assert sync.coverage(tmp_path, "moneyflow", DAY)["missing_symbols"] == ["600000"]


def test_publication_calendar_backoff_and_success_watermark(tmp_path):
    write_day(tmp_path, "kline_daily", DAY, ["000001"])
    assert sync.pending_tasks(tmp_path, NOW, SCHEDULE, False) == []
    assert sync.pending_tasks(tmp_path, NOW, SCHEDULE, None) == []
    afternoon = sync.pending_tasks(tmp_path, NOW.replace(hour=16), SCHEDULE, True)
    assert (DAY, "moneyflow") not in afternoon
    assert (DAY, "auction_open") in afternoon
    sync._save(tmp_path, {str(DAY): {
        "index": {"status": "succeeded"},
        "etf": {"retry_at": (NOW + timedelta(minutes=5)).isoformat()},
        "auction_open": {"attempts": sync.MAX_ATTEMPTS},
    }})
    assert sync.pending_tasks(tmp_path, NOW, SCHEDULE, True) == [(DAY, "moneyflow"), (DAY, "auction_open")]
    later = NOW + timedelta(minutes=6)
    assert (DAY, "etf") in sync.pending_tasks(tmp_path, later, SCHEDULE, True)


def test_failed_dataset_does_not_block_other_dataset_or_claim_success(monkeypatch, tmp_path):
    write_day(tmp_path, "kline_daily", DAY, ["000001"])
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), refresh_index_views=lambda: None)
    calls = []

    def fetch(repo, caps, dataset, day, missing, emit):
        calls.append(dataset)
        if dataset == "moneyflow":
            raise TimeoutError("upstream unavailable")
        write_day(tmp_path, "auction_open", day, ["000001"])
        return 1

    monkeypatch.setattr(sync, "_fetch_missing", fetch)
    monkeypatch.setattr(sync, "cn_now", lambda: NOW)
    monkeypatch.setattr("app.api.data.invalidate_storage_cache", lambda: None)
    result = sync.run_close_data_sync(repo, None, [(DAY, "moneyflow"), (DAY, "auction_open")], now=NOW)
    assert calls == ["moneyflow", "auction_open"]
    assert not result["complete"]
    saved = sync.read_status(tmp_path)[str(DAY)]
    assert saved["moneyflow"]["status"] == "retry"
    assert saved["moneyflow"]["error"] == "TimeoutError"
    assert saved["auction_open"]["status"] == "succeeded"
    assert sync.pending_tasks(tmp_path, NOW, SCHEDULE, True) == [(DAY, "index"), (DAY, "etf")]


def asset_frame(code="000001.SH", day="20260907"):
    return pd.DataFrame({"ts_code": [code], "trade_date": [day], "open": [10.0],
                         "high": [11.0], "low": [9.0], "close": [10.5], "vol": [12.0], "amount": [34.0]})


def test_tushare_index_endpoint_identity_and_units(monkeypatch):
    calls = []

    def index_daily(**kwargs):
        calls.append(kwargs)
        return asset_frame()

    provider = TushareProvider()
    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(index_daily=index_daily))
    frame = provider.get_daily(["000001.SH"], NOW, NOW, asset_type="index")
    assert calls[0]["ts_code"] == "000001.SH"
    assert frame["symbol"].to_list() == ["000001.SH"]
    assert frame["volume"].to_list() == [1200.0]
    assert frame["amount"].to_list() == [34000.0]


def test_tushare_etf_uses_market_day_not_stock_daily(monkeypatch):
    calls = []

    def fund_daily(**kwargs):
        calls.append(kwargs)
        return asset_frame("510300.SH")

    provider = TushareProvider()
    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(fund_daily=fund_daily))
    monkeypatch.setattr(provider, "_trade_dates", lambda *args: ["20260907"])
    frame = provider.get_daily(["510300.SH"], NOW, NOW, asset_type="etf")
    assert calls[0]["trade_date"] == "20260907"
    assert "ts_code" not in calls[0]
    assert frame["symbol"].to_list() == ["510300.SH"]


def test_asset_rejects_wrong_date_and_unqualified_index(monkeypatch):
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(index_daily=lambda **kw: asset_frame(day="20260904")))
    with pytest.raises(ValueError, match="another trading date"):
        provider.get_asset_daily(["000001.SH"], NOW, NOW, "index")
    with pytest.raises(ValueError, match="exchange-qualified"):
        provider.get_asset_daily(["000001"], NOW, NOW, "index")


def test_custom_asset_route_does_not_require_tickflow_capability(monkeypatch, tmp_path):
    from app.services import index_sync
    calls = []
    provider = SimpleNamespace(get_asset_daily=lambda *args: pl.DataFrame({"symbol": ["000001.SH"], "date": [DAY]}))
    monkeypatch.setattr(index_sync, "custom_asset_provider", lambda: provider)
    monkeypatch.setattr(index_sync, "compute_enriched", lambda df, **kw: df)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), append_index_daily=calls.append,
                           append_index_enriched=lambda df: None, refresh_index_views=lambda: None)
    caps = SimpleNamespace(has=lambda cap: False)
    assert index_sync.sync_and_persist_index_daily(repo, caps, symbols_override=["000001.SH"], start_date=NOW, end_date=NOW) == 1
    assert calls[0]["symbol"].to_list() == ["000001.SH"]


def test_bulk_enhanced_retry_filters_stale_dates_and_existing_symbols(monkeypatch):
    from app.data_providers import custom
    from app.services import preferences
    calls, saved = [], []

    def fetch(symbols, *, start_time, end_time, trade_dates):
        calls.append((symbols, trade_dates))
        return pl.DataFrame({"symbol": ["000001", "600000", "600000"],
                             "date": [DAY, DAY - timedelta(days=3), DAY]})

    monkeypatch.setattr(preferences, "get_daily_data_provider", lambda: "tushare")
    monkeypatch.setattr(custom, "provider_has_dataset", lambda *args: True)
    monkeypatch.setattr(custom, "get_provider", lambda name: SimpleNamespace(get_moneyflow=fetch))
    repo = SimpleNamespace(append_moneyflow=saved.append)
    assert sync._fetch_missing(repo, None, "moneyflow", DAY, {"600000"}, lambda *args: None) == 1
    assert calls == [(["600000"], [DAY])]
    assert saved[0].to_dicts() == [{"symbol": "600000", "date": DAY}]


def test_after_close_does_not_report_incomplete_dataset_job_as_success(monkeypatch, tmp_path):
    from app.jobs import daily_pipeline
    from app.services import pipeline_jobs
    store = pipeline_jobs.JobStore(store_dir=tmp_path / "jobs")
    monkeypatch.setattr(pipeline_jobs, "job_store", store)
    monkeypatch.setattr(pipeline_jobs, "try_acquire_run_slot", lambda *args: True)
    monkeypatch.setattr(pipeline_jobs, "release_run_slot", lambda *args: None)
    result = {"scheduled_mode": "close_dataset_incremental", "complete": False,
              "datasets": [{"dataset": "moneyflow", "target_date": str(DAY),
                            "missing_symbols": ["600000"], "complete": False}]}
    assert not daily_pipeline._run_tracked(lambda **kwargs: result, "close_dataset_incremental")
    assert store.list_recent(1)[0]["status"] == "failed"


def test_halted_stocks_not_misreported_as_missing_moneyflow(tmp_path):
    folder = tmp_path / "kline_daily" / f"date={DAY}"
    folder.mkdir(parents=True)
    pl.DataFrame({"symbol": ["000001", "600000", "600001"], "date": [DAY] * 3,
                  "volume": [100.0, 0.0, None]}).write_parquet(folder / "data.parquet")
    write_day(tmp_path, "moneyflow", DAY, ["000001"])
    result = sync.coverage(tmp_path, "moneyflow", DAY)
    assert result["expected_symbols"] == 2
    assert result["missing_symbols"] == ["600001"]


def test_core_indices_are_the_daily_completeness_contract(tmp_path):
    from app.services.index_const import CORE_INDEX_SYMBOLS

    symbols = list(CORE_INDEX_SYMBOLS) + ["000300.SH"]
    write_day(tmp_path, "kline_index_daily", DAY, symbols[:-1])
    result = sync.coverage(tmp_path, "index", DAY)
    assert result["complete"]
    assert result["strict_complete"]
    assert result["expected_symbols"] == len(CORE_INDEX_SYMBOLS)


def test_near_complete_enhanced_partition_is_usable_but_keeps_holes(tmp_path):
    symbols = [f"{i:06}" for i in range(100)]
    folder = tmp_path / "kline_daily" / f"date={DAY}"
    folder.mkdir(parents=True)
    pl.DataFrame({"symbol": symbols, "date": [DAY] * len(symbols),
                  "volume": [100.0] * len(symbols)}).write_parquet(folder / "data.parquet")
    write_day(tmp_path, "auction_open", DAY, symbols[:-1])
    result = sync.coverage(tmp_path, "auction_open", DAY)
    assert result["complete"]
    assert not result["strict_complete"]
    assert result["coverage_ratio"] == 0.99
    assert result["missing_symbols"] == [symbols[-1]]

    sync._save(tmp_path, {str(DAY): {"auction_open": {
        "status": "needs_attention", "attempts": sync.MAX_ATTEMPTS,
    }}})
    assert (DAY, "auction_open") in sync.pending_tasks(tmp_path, NOW, SCHEDULE, True)


@pytest.mark.parametrize("empty", [True, False])
def test_auction_no_progress_is_explicit_and_resumes_after_cooldown(monkeypatch, tmp_path, empty):
    write_day(tmp_path, "kline_daily", DAY, ["000001", "600000"])
    if not empty:
        write_day(tmp_path, "auction_open", DAY, ["000001"])
    sync._save(tmp_path, {str(DAY): {"auction_open": {"attempts": sync.MAX_ATTEMPTS - 1}}})
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), refresh_index_views=lambda: None)
    monkeypatch.setattr(sync, "_fetch_missing", lambda *args: 0)
    monkeypatch.setattr(sync, "cn_now", lambda: NOW)
    monkeypatch.setattr("app.api.data.invalidate_storage_cache", lambda: None)
    result = sync.run_close_data_sync(repo, None, [(DAY, "auction_open")], now=NOW)
    row = result["datasets"][0]
    assert not row["complete"]
    assert row["data_state"] == ("empty" if empty else "partial")
    assert row["error_reason"] == "no_matching_rows"
    assert row["error_message"]
    assert (DAY, "auction_open") not in sync.pending_tasks(tmp_path, NOW + timedelta(minutes=31), SCHEDULE, True)
    assert (DAY, "auction_open") in sync.pending_tasks(tmp_path, NOW + timedelta(hours=3), SCHEDULE, True)


def test_auction_retry_requests_strict_errors_without_enabling_different_dataset(monkeypatch):
    from app.data_providers import custom
    from app.services import preferences
    calls = []

    def fetch(symbols, *, start_time, end_time, market, trade_dates, strict=False):
        calls.append((symbols, trade_dates, market, strict))
        return pl.DataFrame()

    monkeypatch.setattr(preferences, "get_daily_data_provider", lambda: "tushare")
    monkeypatch.setattr(custom, "provider_has_dataset", lambda *args: True)
    monkeypatch.setattr(custom, "get_provider", lambda name: SimpleNamespace(get_auction=fetch))
    assert sync._fetch_missing(SimpleNamespace(), None, "auction_open", DAY, {"000001"}, lambda *args: None) == 0
    assert calls == [(["000001"], [DAY], "o", True)]


def test_auction_failover_reasons_are_checkpointed_without_sensitive_details(monkeypatch, tmp_path):
    from app.plugins.tushare.provider import TushareAuctionSourceError

    write_day(tmp_path, 'kline_daily', DAY, ['000001'])
    def fail(*args):
        raise TushareAuctionSourceError('all_sources_failed', source_failures=[
            {'source': 'promax', 'reason': 'network_timeout'},
            {'source': 'datahub', 'reason': 'permission_denied'},
        ])
    monkeypatch.setattr(sync, '_fetch_missing', fail)
    monkeypatch.setattr(sync, 'cn_now', lambda: NOW)
    monkeypatch.setattr('app.api.data.invalidate_storage_cache', lambda: None)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), refresh_index_views=lambda: None)
    item = sync.run_close_data_sync(repo, None, [(DAY, 'auction_open')])['datasets'][0]
    assert item['error_reason'] == 'all_sources_failed'
    assert item['error_sources'] == [
        {'source': 'promax', 'reason': 'network_timeout'},
        {'source': 'datahub', 'reason': 'permission_denied'},
    ]
    assert not item['complete']
    assert item['error_message']
    assert sync.read_status(tmp_path)[str(DAY)]['auction_open']['error_sources'] == item['error_sources']
