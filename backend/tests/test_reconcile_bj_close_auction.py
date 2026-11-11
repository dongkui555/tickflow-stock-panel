from reconcile_bj_close_auction import classify_row


def base(**overrides):
    row = {
        "market": "BJ",
        "bars": "240",
        "after_hours_rows": "0",
        "price_match": "false",
        "volume_match": "true",
        "amount_match": "false",
        "amount": "1100",
        "minute_amount": "1000",
        "alias_count": "1",
        "selected_symbol": "920001.BJ",
    }
    row.update(overrides)
    return row


def test_marks_close_auction_scope_when_ohl_and_volume_are_valid():
    assert classify_row(base(), ohl_match=True) == "regular_session_close_auction"


def test_keeps_non_volume_or_negative_turnover_conflicts_isolated():
    assert classify_row(base(volume_match="false"), ohl_match=True) == "complete_conflict"
    assert classify_row(base(amount="900"), ohl_match=True) == "complete_conflict"


def test_keeps_identity_and_time_axis_conflicts_isolated():
    assert classify_row(base(alias_count="2"), ohl_match=True) == "complete_conflict"
    assert classify_row(base(bars="239"), ohl_match=True) == "complete_conflict"
