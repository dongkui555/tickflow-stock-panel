from reconcile_bj_regular_session import classify_row


def base(**overrides):
    row = {
        "market": "BJ",
        "bars": "240",
        "after_hours_rows": "0",
        "price_match": "true",
        "volume_match": "true",
        "amount_match": "false",
        "amount": "1100",
        "minute_amount": "1000",
    }
    row.update(overrides)
    return row


def test_marks_regular_session_when_only_daily_total_includes_extra_turnover():
    assert classify_row(base()) == "regular_session"


def test_keeps_amount_above_daily_as_conflict():
    assert classify_row(base(minute_amount="1200")) == "complete_conflict"


def test_keeps_price_or_session_shape_conflicts():
    assert classify_row(base(price_match="false")) == "complete_conflict"
    assert classify_row(base(bars="239")) == "complete_conflict"
    assert classify_row(base(after_hours_rows="1")) == "complete_conflict"
