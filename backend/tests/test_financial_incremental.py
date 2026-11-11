import threading
from datetime import date
from types import SimpleNamespace

import polars as pl

from app.services.financial_incremental import sync_announcements


def test_announcement_sync_preserves_history_and_backups(tmp_path, monkeypatch):
    path = tmp_path / "financials" / "income" / "part.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["000001"],
            "period_end": [date(2024, 12, 31)],
            "announce_date": [date(2025, 3, 20)],
            "revenue": [10.0],
        }
    ).write_parquet(path)

    def fetch(table, day):
        if table == "income" and day == date(2026, 9, 4):
            return pl.DataFrame(
                {
                    "symbol": ["000001"],
                    "period_end": [date(2026, 6, 30)],
                    "announce_date": [day],
                    "revenue": [12.0],
                }
            )
        return pl.DataFrame()

    monkeypatch.setattr(
        "app.data_providers.custom.get_provider",
        lambda name: SimpleNamespace(get_financial_updates=fetch),
    )
    monkeypatch.setattr("app.services.financial_sync._financial_provider_name", lambda: "tushare")
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), _write_lock=threading.Lock())
    result = sync_announcements(repo, date(2026, 9, 4), lambda *args: None)
    assert result["rows"] == 1
    assert pl.read_parquet(path)["revenue"].to_list() == [10.0, 12.0]
    assert list((tmp_path / "import_backups").glob("financial_tail_*/income/part.parquet"))
    assert any("cash_flow" in reason for reason in result["unavailable"])
