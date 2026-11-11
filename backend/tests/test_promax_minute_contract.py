import gzip
import json
from datetime import date, datetime, timedelta

import httpx
import pandas as pd
import pytest

from app.plugins.tushare.minute_contract import normalize_history
from app.plugins.tushare.provider import (
    TushareMinutePendingError,
    TushareProvider,
    _PromaxTushareAPI,
)


def source():
    times = [datetime(2026, 5, 8, 9, 30)]
    times += [datetime(2026, 5, 8, 9, 31) + timedelta(minutes=i) for i in range(120)]
    times += [datetime(2026, 5, 8, 13, 1) + timedelta(minutes=i) for i in range(120)]
    times += [datetime(2026, 5, 8, 15, 1), datetime(2026, 5, 8, 15, 2)]
    frame = pd.DataFrame({"ts_code": "920000.BJ", "trade_time": times,
                          "open": 10., "high": 10., "low": 10., "close": 10.,
                          "vol": 100., "amount": 1000.})
    frame.loc[frame.trade_time.dt.strftime("%H:%M").isin(["15:00", "15:02"]), ["vol", "amount"]] = 0.
    return frame


def daily():
    return pd.DataFrame([{"ts_code": "920000.BJ", "trade_date": "20260508",
                          "open": 10., "high": 10., "low": 10., "close": 10.,
                          "vol": 241., "amount": 241.}])


def test_closing_bucket_is_daily_verified_and_raw_is_preserved(tmp_path):
    raw = source()
    original = raw.copy(deep=True)
    result = normalize_history(raw, pd.concat([daily(), daily()]), "920000", tmp_path)
    assert result.height == 241
    assert result["volume"].sum() == 241.
    assert result["amount"].sum() == 241000.
    assert result["datetime"][-1] == datetime(2026, 5, 8, 15)
    pd.testing.assert_frame_equal(raw, original)
    paths = list(tmp_path.rglob("*.json.gz"))
    assert len(paths) == 1
    event = json.loads(gzip.decompress(paths[0].read_bytes()))
    assert event["closing_1501_mapped_to_1500"] is True
    assert event["raw_minute"][-2]["trade_time"].endswith("15:01:00")
    normalize_history(raw, daily(), "920000", tmp_path)
    normalize_history(raw, daily(), "920000", tmp_path)
    assert len(list(tmp_path.rglob("*.json.gz"))) == 2


@pytest.mark.parametrize("bad", ["price", "volume", "amount", "reference_missing", "reference_ambiguous",
                                      "closing_not_empty", "late_extra", "duplicate", "incomplete", "unit", "daily_unit"])
def test_ambiguous_or_unverified_data_never_passes(tmp_path, bad):
    raw, ref = source(), daily()
    if bad == "price":
        ref.loc[0, "close"] = 11.
    if bad == "volume":
        ref.loc[0, "vol"] *= 2
    if bad == "amount":
        ref.loc[0, "amount"] *= 2
    if bad == "reference_missing":
        ref = ref.iloc[:0]
    if bad == "reference_ambiguous":
        ref = pd.concat([ref, ref.assign(close=11.)])
    if bad == "closing_not_empty":
        raw.loc[raw.trade_time.dt.strftime("%H:%M").eq("15:00"), ["vol", "amount"]] = [100, 1000]
    if bad == "late_extra":
        raw.loc[raw.index[-1], ["vol", "amount"]] = [100, 1000]
    if bad == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]].assign(vol=200.)], ignore_index=True)
    if bad == "incomplete":
        raw = raw.drop(index=10)
    if bad == "unit":
        raw["vol"] *= 100
    if bad == "daily_unit":
        ref["vol"] *= 100
    assert normalize_history(raw, ref, "920000", tmp_path).is_empty()


def test_other_symbol_or_subminute_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="another symbol"):
        normalize_history(source(), daily(), "000001", tmp_path)
    raw = source()
    raw.loc[0, "trade_time"] += timedelta(seconds=1)
    outcomes = []
    with pytest.raises(ValueError, match="sub-minute"):
        normalize_history(raw, daily(), "920000", tmp_path, outcomes=outcomes)
    assert outcomes[0]["reason"] == "source_subminute_timestamps"
    assert outcomes[0]["retryable"] is False
    assert outcomes[0]["evidence"]


def test_normal_gateway_bars_are_not_shifted(tmp_path):
    raw = source().iloc[:-2].copy()
    raw.loc[raw.index[-1], ["vol", "amount"]] = [100, 1000]
    result = normalize_history(raw, daily(), "920000", tmp_path)
    assert result["datetime"].to_list() == raw.trade_time.to_list()
    assert result.height == 241


def snapshot_source():
    raw = source().iloc[:-2].copy()
    raw.loc[raw.index[-1], ["vol", "amount"]] = [100., 1000.]
    closing = raw.iloc[[-1]]
    late = [closing.assign(trade_time=closing.trade_time + timedelta(minutes=i)) for i in range(1, 11)]
    return pd.concat([raw, raw, *late], ignore_index=True)


def test_exact_duplicates_and_closing_snapshots_require_daily_validation(tmp_path):
    raw = snapshot_source()
    original = raw.copy(deep=True)
    outcomes = []
    result = normalize_history(raw, daily(), "920000", tmp_path, outcomes=outcomes)
    assert result.height == 241
    assert result["volume"].sum() == 241.
    assert result["amount"].sum() == 241000.
    assert result["datetime"][-1] == datetime(2026, 5, 8, 15)
    pd.testing.assert_frame_equal(raw, original)
    event = json.loads(gzip.decompress(next(tmp_path.rglob("*.json.gz")).read_bytes()))
    assert event["exact_duplicate_rows_removed"] == 241
    assert event["after_close_snapshot_rows_removed"] == 10
    assert event["closing_1501_mapped_to_1500"] is False
    assert len(event["raw_minute"]) == len(raw)


@pytest.mark.parametrize("bad", ["price", "volume", "amount", "missing", "duplicate_conflict", "late_trade"])
def test_snapshot_cleanup_does_not_hide_bad_data(tmp_path, bad):
    raw, ref = snapshot_source(), daily()
    if bad in {"price", "volume", "amount"}:
        ref.loc[0, {"price": "close", "volume": "vol", "amount": "amount"}[bad]] *= 2
    elif bad == "missing":
        raw = raw[raw.trade_time.ne(datetime(2026, 5, 8, 10))]
    elif bad == "duplicate_conflict":
        raw.loc[0, "amount"] += 100
    else:
        raw.loc[raw.index[-1], "vol"] += 100
    assert normalize_history(raw, ref, "920000", tmp_path).is_empty()


def test_empty_opening_quote_is_not_the_first_transaction(tmp_path):
    raw, ref = source(), daily()
    raw.loc[0, ["open", "high", "low", "close", "vol", "amount"]] = [9., 9., 9., 9., 0., 0.]
    ref.loc[0, ["vol", "amount"]] = [240., 240.]
    result = normalize_history(raw, ref, "920000", tmp_path)
    assert result.height == 241
    assert result["open"][0] == 9.
    assert result["volume"][0] == result["amount"][0] == 0.
    assert result["volume"].sum() == 240.
    assert result["amount"].sum() == 240000.


def test_multi_day_validation_cannot_use_another_days_daily(tmp_path):
    raw = pd.concat([source(), source().assign(trade_time=source().trade_time + timedelta(days=1))], ignore_index=True)
    result = normalize_history(raw, daily(), "920000", tmp_path)
    assert result["datetime"].dt.date().unique().to_list() == [date(2026, 5, 8)]


def test_provider_pads_only_promax_history_and_trims_after_validation(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    calls = []
    def respond(request):
        calls.append((request.url.path, dict(request.url.params)))
        df = source().assign(trade_time=source().trade_time.astype(str)) if request.url.path.endswith("stk_mins") else daily()
        return httpx.Response(200, json={"code": 0, "data": df.to_dict(orient="records")})
    api = _PromaxTushareAPI("https://example.test", "test-key")
    api.close()
    api._client = httpx.Client(transport=httpx.MockTransport(respond))
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_ensure_api", lambda: api)
    try:
        result = provider.get_minute(["920000"], datetime(2026, 5, 8, 9, 30), datetime(2026, 5, 8, 15))
        assert result.height == 241
        assert calls[0][1]["start_date"] == "2026-05-08 09:25:00"
        assert calls[0][1]["end_date"] == "2026-05-08 15:10:00"
        assert calls[1][0] == "/daily"
    finally:
        api.close()


def test_after_close_uses_verified_history_not_realtime(tmp_path, monkeypatch):
    from app.config import settings
    from app.plugins.tushare import provider as module

    class ClosedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 5, 8, 15, 30, tzinfo=tz)

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(module, "datetime", ClosedClock)
    api = _PromaxTushareAPI("https://example.test", "test-key")
    api.close()
    calls = []
    def respond(request):
        calls.append(request.url.path)
        assert not request.url.path.endswith("rt_min_daily")
        df = source().assign(trade_time=source().trade_time.astype(str)) if request.url.path.endswith("stk_mins") else daily()
        return httpx.Response(200, json={"code": 0, "data": df.to_dict(orient="records")})
    api._client = httpx.Client(transport=httpx.MockTransport(respond))
    p = TushareProvider()
    monkeypatch.setattr(p, "_ensure_api", lambda: api)
    try:
        assert p.get_minute(["920000"], datetime(2026, 5, 8, 9, 30), datetime(2026, 5, 8, 15)).height == 241
        assert calls == ["/stk_mins", "/daily"]
    finally:
        api.close()


def test_timeout_falls_back_to_single_stock_local_history_and_validates(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    paths = []
    def respond(request):
        paths.append(request.url.path)
        if request.url.path.endswith("stk_mins"):
            raise httpx.ReadTimeout("timeout", request=request)
        if request.url.path.endswith("a_share_mins"):
            assert request.url.params["ts_code"] == "920000.BJ"
            frame = snapshot_source().assign(trade_time=snapshot_source().trade_time.astype(str))
        else:
            frame = daily()
        return httpx.Response(200, json={"code": 0, "data": frame.to_dict(orient="records")})
    api = _PromaxTushareAPI("https://example.test", "test-key")
    api.close()
    api._client = httpx.Client(transport=httpx.MockTransport(respond))
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_ensure_api", lambda: api)
    try:
        frame = provider.get_minute(["920000"], datetime(2026, 5, 8, 9, 30), datetime(2026, 5, 8, 15))
        assert frame.height == 241
        assert paths == ["/stk_mins", "/a_share_mins", "/daily"]
    finally:
        api.close()


def test_alternate_endpoint_pending_respects_cooldown():
    calls = []
    def respond(request):
        calls.append(request.url.path)
        return httpx.Response(202, headers={"Retry-After": "30"}, json={})
    api = _PromaxTushareAPI("https://example.test", "test-key")
    api.close()
    api._client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        for _ in range(2):
            with pytest.raises(TushareMinutePendingError):
                api.a_share_mins(ts_code="000001.SZ", freq="1MIN")
        assert calls == ["/a_share_mins"]
    finally:
        api.close()
