from datetime import date

from app.services.short_term_ai import _buy_confirmation


def eligible():
    return {"fund_data_complete": True, "daily_fund_signal": "early_layout",
            "kline_quality_label": "正常", "risk_score": 80, "stockformer_score": 80,
            "pred_ret_1d": .01, "sector_score": 80}


def test_live_requires_five_minute_confirmation_but_historical_never_reads_future():
    row = eligible()
    gate = {"gate": "open", "local_score": 80}
    assert "5分钟确认" in _buy_confirmation(row, gate, as_of=None)["required_failed"]
    row["five_minute_confirmation"] = {"passed": True, "label": "5分钟量价确认通过"}
    assert _buy_confirmation(row, gate, as_of=None)["status"] == "confirmed"
    row.pop("five_minute_confirmation")
    assert _buy_confirmation(row, gate, as_of=date(2026, 9, 4))["status"] == "confirmed"


def test_price_confirmation_never_overrides_market_gate():
    row = {**eligible(), "five_minute_confirmation": {"passed": True}}
    for state in ("closed", "observe"):
        assert _buy_confirmation(row, {"gate": state, "local_score": 90}, as_of=None)["status"] == "blocked"
