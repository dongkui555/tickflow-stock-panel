from datetime import date
from pathlib import Path

from app.services.minute_cross_source import (
    fetch_eastmoney_daily,
    fetch_promax_daily,
    _normalized_snapshot,
    persist_verification_report,
    verify_stock_day,
)
from app.services import minute_gap_queue


DAY = date(2026, 9, 14)


def _daily(symbol="920832.BJ", volume=543505.0, amount=3509451.0):
    return {
        "symbol": symbol, "date": DAY.isoformat(),
        "open": 6.47, "high": 6.55, "low": 6.38, "close": 6.50,
        "volume": volume, "amount": amount, "volume_unit": "shares",
    }


def _minute(volume=543500.0, amount=3509451.0):
    return {
        "symbol": "920832.BJ", "date": DAY.isoformat(), "rows": 240,
        "open": 6.47, "high": 6.55, "low": 6.38, "close": 6.50,
        "volume": volume, "amount": amount, "volume_unit": "shares",
    }


def _source(name, volume=5435.0, amount=3509451.0, symbol="920832.BJ"):
    return {
        "source": name, "symbol": symbol, "date": DAY.isoformat(),
        "open": 6.47, "high": 6.55, "low": 6.38, "close": 6.50,
        "volume": volume, "volume_unit": "lots", "amount": amount,
    }


def test_two_independent_sources_can_approve_minute_import():
    result = verify_stock_day(
        symbol="920832.BJ", trade_date=DAY, local_daily=_daily(),
        minute_aggregate=_minute(),
        external_snapshots=[_source("promax"), _source("eastmoney")],
    )

    assert result["status"] == "minute_import_candidate"
    assert result["retryable"] is False
    assert result["matched_sources"] == ["promax", "eastmoney"]


def test_two_external_sources_can_identify_a_stale_local_daily_baseline():
    local = _daily(symbol="920375.BJ", volume=1307199.0, amount=8363523.7)
    minute = {**_minute(volume=1806700.0, amount=11090015.0), "symbol": "920375.BJ"}
    sources = [
        {**_source("promax", volume=18072.0, amount=11090015.0, symbol="920375.BJ")},
        {**_source("eastmoney", volume=18072.0, amount=11090015.0, symbol="920375.BJ")},
    ]

    result = verify_stock_day(
        symbol="920375.BJ", trade_date=DAY, local_daily=local,
        minute_aggregate=minute, external_snapshots=sources,
    )

    assert result["status"] == "daily_baseline_repair_candidate"
    assert result["matched_sources"] == ["promax", "eastmoney"]
    assert result["write_allowed"] is False


def test_local_daily_agreement_rejects_qmt_turnover_conflict():
    result = verify_stock_day(
        symbol="920832.BJ", trade_date=DAY, local_daily=_daily(),
        minute_aggregate=_minute(volume=561800.0, amount=3642661.0),
        external_snapshots=[_source("eastmoney")],
    )

    assert result["status"] == "qmt_turnover_conflict"
    assert result["write_allowed"] is False


def test_two_external_sources_agree_with_daily_but_not_minutes():
    result = verify_stock_day(
        symbol="920832.BJ", trade_date=DAY, local_daily=_daily(),
        minute_aggregate=_minute(amount=3509451.0 * 1.005),
        external_snapshots=[_source("promax"), _source("eastmoney")],
    )
    assert result["status"] == "qmt_turnover_conflict"
    assert result["reason"] == "daily_agreement_minute_conflict"
    assert result["write_allowed"] is False


def test_legacy_daily_unit_requires_two_external_confirmations_without_mutation():
    local = _daily(volume=5435.0)
    local.pop("volume_unit")
    for count in (0, 1, 2):
        result = verify_stock_day(
            symbol="920832.BJ", trade_date=DAY, local_daily=local,
            minute_aggregate=_minute(),
            external_snapshots=[_source("promax"), _source("fuyao")][:count],
        )
        assert result["status"] == ("minute_import_candidate" if count == 2 else "quarantine")
    assert local["volume"] == 5435.0
    assert "volume_unit" not in local
    assert result["local_volume_unit"] == "lots"
    assert result["local_volume_unit_basis"] == "external_daily_consensus"


def test_unknown_local_daily_unit_is_quarantined_even_with_external_consensus():
    local = _daily(volume=100.0) | {"volume_unit": "unknown"}
    result = verify_stock_day(
        symbol="920832.BJ", trade_date=DAY, local_daily=local,
        minute_aggregate=_minute(),
        external_snapshots=[_source("promax"), _source("eastmoney")],
    )
    assert result["status"] == "quarantine"
    assert result["reason"] == "local_daily_unit_unknown"
    assert result["write_allowed"] is False


def test_legacy_daily_unit_with_minute_turnover_conflict_is_not_no_consensus():
    local = _daily(volume=5435.0)
    local.pop("volume_unit")
    result = verify_stock_day(
        symbol="920832.BJ", trade_date=DAY, local_daily=local,
        minute_aggregate=_minute(amount=3509451.0 * 1.005),
        external_snapshots=[_source("promax"), _source("fuyao")],
    )
    assert result["status"] == "qmt_turnover_conflict"
    assert result["reason"] == "daily_agreement_minute_conflict"
    assert result["local_volume_unit"] == "lots"
    assert result["write_allowed"] is False


def test_normalized_evidence_is_idempotent_and_keeps_raw_unit():
    first, _ = _normalized_snapshot(_source("promax"), "920832.BJ", DAY.isoformat())
    second, _ = _normalized_snapshot(first, "920832.BJ", DAY.isoformat())
    assert first["volume_unit"] == "shares"
    assert first["source_volume_unit"] == "lots"
    assert first["volume"] == second["volume"] == 543500.0


def test_bare_external_identity_cannot_grant_permission():
    result = verify_stock_day(
        symbol="920832.BJ", trade_date=DAY, local_daily=_daily(),
        minute_aggregate=_minute(),
        external_snapshots=[_source("promax", symbol="920832"), _source("fuyao")],
    )
    assert result["reason"] == "identity_conflict"
    assert result["write_allowed"] is False


def test_external_consensus_must_be_pairwise_not_first_source_only():
    sources = [_source("a"), _source("b", amount=3509451 * 1.0019),
               _source("c", amount=3509451 * .9981)]
    result = verify_stock_day(
        symbol="920832.BJ", trade_date=DAY, local_daily=_daily(),
        minute_aggregate=_minute(), external_snapshots=sources,
    )
    assert result["status"] == "quarantine"
    assert result["reason"] == "no_external_consensus"


def test_local_evidence_does_not_sum_aliases_or_mix_market_identities(tmp_path):
    from datetime import datetime
    from types import SimpleNamespace
    import polars as pl
    from app.services.daily_maintenance import _cross_source_local_evidence

    dp = tmp_path / "kline_daily" / f"date={DAY}" / "part.parquet"
    mp = tmp_path / "kline_minute" / f"date={DAY}" / "part.parquet"
    dp.parent.mkdir(parents=True)
    mp.parent.mkdir(parents=True)
    daily = _daily(volume=5435.)
    daily.pop("volume_unit")
    pl.DataFrame([daily]).write_parquet(dp)
    bar = {k: daily[k] for k in ("symbol", "open", "high", "low", "close", "volume", "amount")}
    bar["datetime"] = datetime.combine(DAY, datetime.min.time()).replace(hour=9, minute=31)
    frame = pl.DataFrame([bar, {**bar, "symbol": "920832"}, {**bar, "symbol": "000001.SZ"}])
    frame.write_parquet(mp)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    context = _cross_source_local_evidence(repo, "920832.BJ", DAY)
    assert context["local_daily"]["volume_unit"] == "unknown"
    assert not context.get("identity_conflict")
    assert context["minute_aggregate"]["rows"] == 1
    assert context["minute_aggregate"]["volume"] == 5435.
    pl.concat([frame, pl.DataFrame([{**bar, "symbol": "920832.SZ"}])]).write_parquet(mp)
    context = _cross_source_local_evidence(repo, "920832.BJ", DAY)
    assert context["identity_conflict"] == ["920832.SZ"]
    assert context["minute_aggregate"] is None


def test_half_percent_amount_difference_is_not_a_repair_candidate():
    local = _daily()
    minute = _minute(amount=3509451.0 * 1.005)
    sources = [_source("promax"), _source("eastmoney")]
    result = verify_stock_day(
        symbol="920832.BJ", trade_date=DAY, local_daily=local,
        minute_aggregate=minute, external_snapshots=sources,
    )
    assert result["status"] == "qmt_turnover_conflict"
    assert result["reason"] == "daily_agreement_minute_conflict"
    assert result["write_allowed"] is False


def test_transport_error_is_retryable_and_not_no_record():
    result = verify_stock_day(
        symbol="920832.BJ", trade_date=DAY, local_daily=_daily(),
        minute_aggregate=_minute(), external_snapshots=[],
        source_errors=[{"source": "eastmoney", "reason": "transport_error", "retryable": True}],
    )

    assert result["status"] == "source_unavailable"
    assert result["retryable"] is True
    assert result["status"] != "source_no_record"


def test_market_identity_conflict_is_quarantined():
    result = verify_stock_day(
        symbol="920832.BJ", trade_date=DAY, local_daily=_daily(),
        minute_aggregate=_minute(),
        external_snapshots=[_source("eastmoney", symbol="920832.SZ")],
    )

    assert result["status"] == "quarantine"
    assert result["reason"] == "identity_conflict"
    assert result["write_allowed"] is False


def test_source_unavailable_is_backed_off_without_becoming_no_record(tmp_path):
    minute_gap_queue.record_cross_source_results(tmp_path, [{
        "symbol": "920832.BJ", "date": DAY.isoformat(),
        "status": "source_unavailable", "reason": "third_party_source_unavailable",
        "retryable": True, "retry_after_seconds": 600, "actual_minutes": 240,
    }])

    row = minute_gap_queue.summary(tmp_path)["retry"]
    assert row["count"] == 1
    assert minute_gap_queue.pending_batch(tmp_path, DAY, limit=10) == []


def test_baseline_candidate_and_quarantine_are_not_claimed_by_minute_retry(tmp_path):
    minute_gap_queue.record_cross_source_results(tmp_path, [
        {"symbol": "920375.BJ", "date": DAY.isoformat(),
         "status": "daily_baseline_repair_candidate", "reason": "independent_daily_agreement"},
        {"symbol": "920832.BJ", "date": DAY.isoformat(),
         "status": "quarantine", "reason": "no_consensus"},
    ])

    assert minute_gap_queue.pending_batch(tmp_path, DAY, limit=10) == []
    assert minute_gap_queue.summary(tmp_path)["daily_baseline_repair_candidate"]["count"] == 1
    assert minute_gap_queue.summary(tmp_path)["quarantine"]["count"] == 1


def test_qmt_turnover_conflict_reopens_the_next_minute_source(tmp_path):
    minute_gap_queue.record_cross_source_results(tmp_path, [{
        "symbol": "920832.BJ", "date": DAY.isoformat(),
        "status": "qmt_turnover_conflict", "reason": "local_daily_agreement",
        "actual_minutes": 239,
    }])
    assert minute_gap_queue.pending_batch(tmp_path, DAY, limit=10) == ["920832.BJ"]


def test_promax_snapshot_keeps_identity_and_share_units():
    class Provider:
        def _ensure_api(self):
            class API:
                @staticmethod
                def daily(**_kwargs):
                    return [{
                        "ts_code": "920832.BJ", "trade_date": DAY,
                        "open": 6.47, "high": 6.55, "low": 6.38, "close": 6.50,
                        "vol": 5435.05, "amount": 3509.451,
                    }]
            return API()

    snapshot, error = fetch_promax_daily("920832.BJ", DAY, provider=Provider())
    assert error is None
    assert snapshot["symbol"] == "920832.BJ"
    assert snapshot["volume_unit"] == "shares"
    assert snapshot["volume"] == 543505


def test_promax_normalizes_bare_stock_code_before_request():
    from types import SimpleNamespace

    requested = []

    def daily(**kwargs):
        requested.append(kwargs["ts_code"])
        return [{
            "ts_code": "000008.SZ", "trade_date": DAY,
            "open": 2.36, "high": 2.40, "low": 2.36, "close": 2.38,
            "vol": 189408.83, "amount": 45067.174,
        }]

    snapshot, error = fetch_promax_daily(
        "000008", DAY,
        provider=SimpleNamespace(_ensure_api=lambda: SimpleNamespace(daily=daily)),
    )

    assert error is None
    assert requested == ["000008.SZ"]
    assert snapshot["symbol"] == "000008.SZ"


def test_eastmoney_snapshot_normalizes_lots_without_treating_empty_as_transport():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            import json
            return json.dumps({"data": {"klines": [
                "20260914,6.47,6.50,6.55,6.38,5435.05,3509451,0,0,0"
            ]}}).encode()

    snapshot, error = fetch_eastmoney_daily(
        "920832.BJ", DAY, opener=lambda *_args, **_kwargs: Response(),
    )
    assert error is None
    assert snapshot["volume_unit"] == "lots"
    assert snapshot["volume"] == 5435.05


def test_eastmoney_snapshot_normalizes_bare_stock_code_before_request():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            import json
            return json.dumps({"data": {"klines": [
                "20260914,2.36,2.38,2.40,2.36,189408.83,45067174,0,0,0"
            ]}}).encode()

    snapshot, error = fetch_eastmoney_daily(
        "000008", DAY, opener=lambda request, **_args: Response(),
    )

    assert error is None
    assert snapshot["symbol"] == "000008.SZ"


def test_eastmoney_transport_error_is_retryable():
    snapshot, error = fetch_eastmoney_daily(
        "920832.BJ", DAY, opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("down")),
    )
    assert snapshot is None
    assert error["reason"] == "transport_error"
    assert error["retryable"] is True


def test_fuyao_snapshot_normalizes_raw_daily_k_and_identity():
    from app.services.minute_cross_source import fetch_fuyao_daily

    class Client:
        def historical_kline(self, thscode, start_ms, end_ms, adjust):
            assert thscode == "920832.BJ"
            assert adjust == "none"
            return [{
                "thscode": "920832.BJ", "date_ms": 1789315200000,
                "open_price": 6.47, "high_price": 6.55,
                "low_price": 6.38, "close_price": 6.50,
                "volume": 543505.0, "turnover": 3509451.0,
            }]

        def close(self):
            pass

    snapshot, error = fetch_fuyao_daily("920832.BJ", DAY, client=Client())
    assert error is None
    assert snapshot["source"] == "fuyao"
    assert snapshot["volume_unit"] == "shares"


def test_fuyao_normalizes_bare_stock_code_before_request():
    from app.services.minute_cross_source import fetch_fuyao_daily

    class Client:
        def historical_kline(self, thscode, start_ms, end_ms, adjust):
            assert thscode == "000008.SZ"
            assert adjust == "none"
            return [{
                "thscode": "000008.SZ", "date_ms": 1789315200000,
                "open_price": 2.36, "high_price": 2.40,
                "low_price": 2.36, "close_price": 2.38,
                "volume": 18940883.0, "turnover": 45067174.0,
            }]

        def close(self):
            pass

    snapshot, error = fetch_fuyao_daily("000008", DAY, client=Client())
    assert error is None
    assert snapshot["symbol"] == "000008.SZ"


def test_fuyao_wrong_exchange_is_quarantined():
    from app.services.minute_cross_source import fetch_fuyao_daily

    class Client:
        def historical_kline(self, *_args, **_kwargs):
            return [{
                "thscode": "920832.SZ", "date_ms": 1789315200000,
                "open_price": 6.47, "high_price": 6.55,
                "low_price": 6.38, "close_price": 6.50,
                "volume": 543505.0, "turnover": 3509451.0,
            }]

        def close(self):
            pass

    snapshot, error = fetch_fuyao_daily("920832.BJ", DAY, client=Client())
    assert snapshot is None
    assert error["reason"] == "identity_conflict"


def test_promax_compact_date_is_not_misclassified_as_empty():
    from types import SimpleNamespace
    api = SimpleNamespace(daily=lambda **kw: [{
        "ts_code": "920832.BJ", "trade_date": "20260914",
        "open": 6.47, "high": 6.55, "low": 6.38, "close": 6.50,
        "vol": 5435.05, "amount": 3509.451,
    }])
    snapshot, error = fetch_promax_daily(
        "920832.BJ", DAY, provider=SimpleNamespace(_ensure_api=lambda: api),
    )
    assert error is None
    assert snapshot["date"] == DAY.isoformat()


def test_promax_rejects_same_code_in_wrong_exchange():
    from types import SimpleNamespace
    api = SimpleNamespace(daily=lambda **kw: [{
        "ts_code": "920832.SZ", "trade_date": DAY,
        "open": 6.47, "high": 6.55, "low": 6.38, "close": 6.50,
        "vol": 5435.05, "amount": 3509.451,
    }])
    snapshot, error = fetch_promax_daily(
        "920832.BJ", DAY, provider=SimpleNamespace(_ensure_api=lambda: api),
    )
    assert snapshot is None
    assert error["reason"] == "identity_conflict"


def test_eastmoney_iso_dates_and_unadjusted_prices():
    import io
    import json
    from urllib.parse import parse_qs, urlsplit
    requested = []

    def opener(request, **kw):
        requested.append(parse_qs(urlsplit(request.full_url).query))
        return io.BytesIO(json.dumps({"data": {"code": "920832", "klines": [
            "2026-09-14,6.47,6.50,6.55,6.38,5435.05,3509451,0,0,0",
        ]}}).encode())

    snapshot, error = fetch_eastmoney_daily("920832.BJ", DAY, opener=opener)
    assert error is None
    assert snapshot["date"] == DAY.isoformat()
    assert requested[0]["fqt"] == ["0"]


def test_disagreeing_sources_cannot_approve_import_by_order():
    minute = _minute(amount=4000000)
    sources = [_source("promax"), _source("eastmoney", amount=4000000)]
    for ordered in (sources, sources[::-1]):
        result = verify_stock_day(
            symbol="920832.BJ", trade_date=DAY, local_daily=_daily(),
            minute_aggregate=minute, external_snapshots=ordered,
        )
        assert result["write_allowed"] is False
        assert result["status"] == "quarantine"


def test_verification_report_is_append_only_audit_artifact(tmp_path):
    report_dir = persist_verification_report(tmp_path, [{
        "symbol": "920832.BJ", "date": DAY.isoformat(),
        "status": "source_unavailable", "reason": "third_party_source_unavailable",
    }])
    report = Path(report_dir) / "report.json"
    assert report.exists()
    assert '"source_unavailable": 1' in (report.parent / "summary.json").read_text(encoding="utf-8")
    assert (report.parent / "details.csv").exists()
    assert '"status": "source_unavailable"' in (report.parent / "retry_queue.jsonl").read_text(encoding="utf-8")
