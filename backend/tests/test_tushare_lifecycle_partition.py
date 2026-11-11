import pandas as pd

from app.plugins.tushare.provider import TushareProvider


def _frame(code: str, exchange: str, status: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "ts_code": f"{code}.{exchange}",
        "symbol": code,
        "name": f"name-{code}",
        "area": "",
        "industry": "",
        "market": "主板",
        "list_date": "20200101",
        "delist_date": "20250101" if status == "D" else None,
        "list_status": status,
        "exchange": exchange,
    }])


def test_lifecycle_requests_each_status_by_exchange(monkeypatch):
    provider = TushareProvider()
    calls: list[tuple[str, str | None]] = []

    def fake_status(status: str, *, exchange: str | None = None):
        calls.append((status, exchange))
        if exchange == "SSE":
            return _frame("600000", "SH", status)
        if exchange == "SZSE":
            return _frame("000001", "SZ", status)
        if exchange == "BSE":
            return _frame("920001", "BJ", status)
        return None

    monkeypatch.setattr(provider, "_stock_basic_status", fake_status)

    rows = provider.get_instrument_lifecycle()

    assert calls == [
        ("L", "SSE"), ("L", "SZSE"), ("L", "BSE"),
        ("D", "SSE"), ("D", "SZSE"), ("D", "BSE"),
        ("P", "SSE"), ("P", "SZSE"), ("P", "BSE"),
    ]
    assert {row["code"] for row in rows} == {"600000", "000001", "920001"}
    assert len(rows) == 9


def test_lifecycle_deduplicates_partitioned_rows_without_merging_status():
    provider = TushareProvider()
    provider._stock_basic_status = lambda status, *, exchange=None: _frame(
        "000001", "SZ", status,
    )

    rows = provider.get_instrument_lifecycle()

    assert len(rows) == 3
    assert {row["ext"]["list_status"] for row in rows} == {"L", "D", "P"}
