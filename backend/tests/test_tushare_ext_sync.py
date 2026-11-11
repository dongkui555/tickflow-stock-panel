from datetime import date

import polars as pl
import pytest

from app.services import tushare_ext_sync


def test_latest_ext_day_is_saved_even_if_older_request_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace

    days = [date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 7)]
    monkeypatch.setattr(tushare_ext_sync, "_trade_dates",
                        lambda root, start, end: [d for d in days if start <= d <= end])
    calls = []
    def method(**kwargs):
        calls.append(kwargs["trade_date"])
        if kwargs["trade_date"] == "20260904":
            raise TimeoutError("old date unavailable")
        return pl.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [kwargs["trade_date"]],
                             "up_limit": [11.0]})

    spec = tushare_ext_sync.EXT_SPECS["stk_limit"]
    with pytest.raises(RuntimeError) as caught:
        tushare_ext_sync._sync_one_spec(tmp_path, SimpleNamespace(stk_limit=method), spec,
                                        [(days[0], days[-1])])
    assert calls == ["20260907", "20260904"]
    assert caught.value.rows_written == 1
    path = tmp_path / "ext_data" / spec.config_id / "timeseries" / "date=2026-09-07" / "part.parquet"
    assert pl.read_parquet(path)["trade_date"].to_list() == ["2026-09-07"]


def test_ext_new_latest_does_not_hide_older_missing_partitions(tmp_path):
    spec = tushare_ext_sync.EXT_SPECS["stk_limit"]
    for day in (date(2026, 7, 15), date(2026, 8, 10), date(2026, 9, 7)):
        (tmp_path / "kline_daily" / f"date={day}").mkdir(parents=True)
    for day in (date(2026, 7, 15), date(2026, 9, 7)):
        (tmp_path / "ext_data" / spec.config_id / "timeseries" / f"date={day}").mkdir(parents=True)
    ranges = tushare_ext_sync._incremental_ranges_for_ext(
        tmp_path, spec, date(2026, 9, 7), refresh_days=10, max_catchup_days=90,
    )
    assert any(start <= date(2026, 8, 10) <= end for start, end in ranges)


def test_incremental_range_catches_import_gap_without_full_resync(tmp_path):
    spec = tushare_ext_sync.EXT_SPECS["stk_limit"]
    latest_dir = tmp_path / "ext_data" / spec.config_id / "timeseries" / "date=2026-07-15"
    latest_dir.mkdir(parents=True)

    ranges = tushare_ext_sync._incremental_ranges_for_ext(
        tmp_path,
        spec,
        date(2026, 9, 3),
        refresh_days=10,
        max_catchup_days=90,
    )

    assert ranges == [(date(2026, 7, 16), date(2026, 9, 3))]


def test_incremental_range_caps_very_stale_import(tmp_path):
    spec = tushare_ext_sync.EXT_SPECS["stk_limit"]
    latest_dir = tmp_path / "ext_data" / spec.config_id / "timeseries" / "date=2024-09-01"
    latest_dir.mkdir(parents=True)

    ranges = tushare_ext_sync._incremental_ranges_for_ext(
        tmp_path,
        spec,
        date(2026, 9, 3),
        refresh_days=10,
        max_catchup_days=90,
    )

    assert ranges == [(date(2026, 6, 5), date(2026, 9, 3))]


def test_write_timeseries_upserts_by_unique_fields(tmp_path):
    spec = tushare_ext_sync.EXT_SPECS["moneyflow_dc"]

    tushare_ext_sync._write_timeseries_frame(
        tmp_path,
        spec,
        pl.DataFrame(
            {
                "symbol": ["000001"],
                "trade_date": ["2026-09-02"],
                "net_amount": [1.0],
            }
        ),
    )
    written = tushare_ext_sync._write_timeseries_frame(
        tmp_path,
        spec,
        pl.DataFrame(
            {
                "symbol": ["000001", "000002"],
                "trade_date": ["2026-09-02", "2026-09-02"],
                "net_amount": [3.0, 2.0],
            }
        ),
    )

    path = tmp_path / "ext_data" / spec.config_id / "timeseries" / "date=2026-09-02" / "part.parquet"
    df = pl.read_parquet(path)
    by_symbol = {row["symbol"]: row["net_amount"] for row in df.to_dicts()}

    assert written == 2
    assert by_symbol == {"000001": 3.0, "000002": 2.0}


def test_default_auto_apis_skip_flaky_cross_border_lists_by_default():
    assert "hsgt_top10" in tushare_ext_sync.EXT_SPECS
    assert "ggt_top10" in tushare_ext_sync.EXT_SPECS
    assert "hsgt_top10" not in tushare_ext_sync.DEFAULT_AUTO_APIS
    assert "ggt_top10" not in tushare_ext_sync.DEFAULT_AUTO_APIS


def test_ext_response_cannot_silently_write_another_day(tmp_path, monkeypatch):
    spec = tushare_ext_sync.EXT_SPECS["stk_limit"]
    monkeypatch.setattr(tushare_ext_sync, "_trade_dates", lambda *args: [date(2026, 9, 3)])
    def method(**kwargs):
        return pl.DataFrame({"symbol": ["000001"], "trade_date": ["20260715"], "up_limit": [10.0]})
    with pytest.raises(ValueError, match="requested trading day"):
        tushare_ext_sync._fetch_frames(method, spec, date(2026, 9, 3), date(2026, 9, 3), tmp_path)


def test_mixed_gateway_columns_are_normalized_before_arrow():
    import pandas as pd

    spec = tushare_ext_sync.EXT_SPECS["top_inst"]
    source = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ"],
        "trade_date": ["20260903", "20260903"],
        "exalter": ["institution", 1.0],
        "buy": [23.5, ""],
        "net_buy": ["-10.25", 12.0],
        "ignored_vendor_field": [1.0, "mixed"],
    })
    result = tushare_ext_sync._call_tushare(lambda **kw: source, spec, {})
    assert result["symbol"].to_list() == ["000001", "000002"]
    assert result["buy"].to_list() == [23.5, None]
    assert result["net_buy"].to_list() == [-10.25, 12.0]
    assert result["exalter"].to_list() == ["institution", "1.0"]
    assert result["trade_date"].to_list() == ["2026-09-03"] * 2
    assert source["buy"][1] == ""


@pytest.mark.parametrize("api", ["forecast", "express"])
def test_range_response_accepts_inclusive_literal_date_boundaries(tmp_path, api):
    spec = tushare_ext_sync.EXT_SPECS[api]

    def method(**kwargs):
        assert kwargs["start_date"] == "20260716"
        assert kwargs["end_date"] == "20260905"
        return pl.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "ann_date": ["20260716", "20260801", "20260905"],
            "end_date": ["20260630"] * 3,
        })

    frames = tushare_ext_sync._fetch_frames(
        method, spec, date(2026, 7, 16), date(2026, 9, 5), tmp_path,
    )
    assert len(frames) == 1
    assert frames[0]["ann_date"].to_list() == ["2026-07-16", "2026-08-01", "2026-09-05"]


@pytest.mark.parametrize("ann_date", ["20260715", "20260906", None, "invalid"])
def test_range_response_rejects_outside_or_unknown_announcement_date(tmp_path, ann_date):
    spec = tushare_ext_sync.EXT_SPECS["forecast"]

    def method(**kwargs):
        return pl.DataFrame({"ts_code": ["000001.SZ"], "ann_date": [ann_date]})

    with pytest.raises(ValueError, match="outside requested date range"):
        tushare_ext_sync._fetch_frames(
            method, spec, date(2026, 7, 16), date(2026, 9, 5), tmp_path,
        )
