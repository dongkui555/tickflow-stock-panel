"""Daily authority must take precedence over a symbol alias's snapshot."""
from datetime import date

import polars as pl
from app.services.minute_repair import daily_reference


def row(symbol="000001.SZ", **extra):
    return {"symbol": symbol, "date": date(2026, 10, 19), "open": 11.48,
            "high": 11.68, "low": 11.41, "close": 11.65,
            "volume": 126779262., "amount": 1466720689.33, **extra}


def test_final_bare_daily_takes_priority_over_qualified_close_snapshot():
    final = row("000001", quote_ts=None)
    snapshot = row(quote_ts=1792393201000, close=11.64, volume=126196362.)
    for records in ([final, snapshot], [snapshot, final]):
        chosen = daily_reference(pl.DataFrame(records))["000001"]
        assert chosen["close"] == 11.65
        assert chosen["volume"] == 126779262.
        assert chosen["quote_ts"] is None


def test_priority_is_per_security_and_keeps_snapshot_when_no_history_exists():
    chosen = daily_reference(pl.DataFrame([
        row("000001", quote_ts=None), row(quote_ts=1792393201000, close=11.64),
        row("600000.SH", quote_ts=1792393201000, close=9.78),
    ]))
    assert chosen["000001"]["close"] == 11.65
    assert chosen["600000"]["close"] == 9.78


def test_missing_snapshot_marker_keeps_existing_canonical_alias_priority():
    chosen = daily_reference(pl.DataFrame([row(), row("000001", close=99.)]))
    assert chosen["000001"]["close"] == 11.65


def test_ambiguous_authoritative_alias_stays_excluded():
    chosen = daily_reference(pl.DataFrame([
        row("000001", quote_ts=None), row("000001", quote_ts=None, close=11.66),
        row(quote_ts=1792393201000, close=11.64),
    ]))
    assert "000001" not in chosen


def test_invalid_authoritative_values_cannot_fall_back_to_stale_snapshot():
    chosen = daily_reference(pl.DataFrame([
        row("000001", quote_ts=None, close=0.), row(quote_ts=1792393201000),
    ]))
    assert "000001" not in chosen
