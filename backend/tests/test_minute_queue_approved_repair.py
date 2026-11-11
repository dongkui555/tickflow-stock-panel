import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from test_minute_failover import bars


def test_offline_repair_preserves_valid_and_unapproved_days_and_backs_up(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/repair_minute_queue_20260909.py"
    spec = importlib.util.spec_from_file_location("approved_minute_repair", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    codes = sorted(set.union(*tool.APPROVED.values()) | {f"0000{i:02}" for i in range(1, 40)})
    assert len(codes) == 55
    audit, originals = [], {}
    for text in tool.APPROVED:
        day = date.fromisoformat(text)
        old, refs = [], []
        for code in codes:
            frame = bars(code, day)
            conflict = code in tool.APPROVED[text]
            reason = "existing_values_conflict" if conflict else "already_complete"
            audit.append({"date": text, "symbol": code, "source_validation": "accepted",
                          "old_validation": "source_incomplete" if conflict else "accepted", "merge_reason": reason})
            if conflict or code == "000001":
                frame = frame.head(2).with_columns(pl.lit(0.5).alias("volume"), pl.lit(500.).alias("amount"))
            elif code == "000002":
                frame = frame.head(2)
            old.append(frame)
            refs.append({"symbol": code, "date": day, **dict.fromkeys(["open", "high", "low", "close"], 10.),
                         "volume": 240., "amount": 240000.})
        for table, frame in [("kline_minute", pl.concat(old)), ("kline_daily", pl.DataFrame(refs))]:
            p = tmp_path / table / f"date={day}" / "part.parquet"
            p.parent.mkdir(parents=True)
            frame.write_parquet(p)
        originals[text] = pl.concat(old)
    report = tmp_path / "import_reports/audit.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"task_id": "be41ce688d", "rows": audit}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["repair", "--data-dir", str(tmp_path), "--audit-report", str(report),
                                     "--base-url", "http://unused", "--apply-approved-replacements"])
    metadata = {"State": {"Running": True}, "Mounts": [{"Source": str(tmp_path), "Destination": "/app/data"}]}
    monkeypatch.setattr(tool.subprocess, "check_output", lambda *a, **kw: json.dumps([metadata]))
    with pytest.raises(RuntimeError, match="Stop the application"):
        tool.main()
    assert not (tmp_path / "import_backups").exists()
    metadata["State"]["Running"] = False
    monkeypatch.setattr(tool, "TdxProvider", lambda: SimpleNamespace(
        get_minute=lambda codes, start, end: bars(codes[0], start.date()), close=lambda: None))
    tool.main()
    result = json.loads(next((tmp_path / "import_reports").glob("minute_queue_repair_*.json")).read_text())
    assert sum(r["result"] == "approved_replacement" for r in result["rows"]) == 16
    assert len(result["backups"]) == 2
    for text, original in originals.items():
        stored = pl.read_parquet(tmp_path / f"kline_minute/date={text}/part.parquet")
        # An unapproved conflict introduced after the audit is never replaced.
        assert stored.filter(pl.col("symbol") == "000001").equals(original.filter(pl.col("symbol") == "000001"))
        for code in tool.APPROVED[text] | {"000002", "000003"}:
            assert stored.filter(pl.col("symbol") == code).equals(bars(code, date.fromisoformat(text)))
        backup = next((tmp_path / "import_backups").glob(f"*/date={text}/part.parquet"))
        assert pl.read_parquet(backup).equals(original)
