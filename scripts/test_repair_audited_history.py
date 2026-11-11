import json
from datetime import datetime, timedelta

import polars as pl

from repair_audited_history import digest, import_minutes


def test_batch_minute_import_backs_up_partition_once_and_is_idempotent(tmp_path):
    root, output = tmp_path / "data", tmp_path / "audit"
    day = datetime(2026, 9, 8)
    times = [day.replace(hour=9, minute=30) + timedelta(minutes=i) for i in range(121)]
    times += [day.replace(hour=13, minute=1) + timedelta(minutes=i) for i in range(120)]
    raw = pl.DataFrame({"symbol": ["000001", "600000"], "date": [day.date()] * 2,
                        **{c: [10., 10.] for c in ("open", "high", "low", "close")},
                        "volume": [24100., 24100.], "amount": [241000., 241000.]})
    daily = root / "kline_daily/date=2026-09-08/part.parquet"
    daily.parent.mkdir(parents=True)
    raw.write_parquet(daily)
    (output / "minutes").mkdir(parents=True)
    frames = []
    for code in ("000001", "600000"):
        frame = pl.DataFrame({"symbol": [code] * 241, "datetime": times,
                              **{c: [10.] * 241 for c in ("open", "high", "low", "close")},
                              "volume": [1.] * 241, "amount": [1000.] * 241})
        frame.write_parquet(output / "minutes" / f"{code}.parquet")
        frames.append(frame.head(1))
    path = root / "kline_minute/date=2026-09-08/part.parquet"
    path.parent.mkdir(parents=True)
    pl.concat(frames).write_parquet(path)
    before = digest(path)
    import_minutes(root, output, apply=False)
    assert digest(path) == before
    assert not (output / "backups").exists()
    import_minutes(root, output)
    assert pl.read_parquet(path).height == 482
    backups = list((output / "backups").rglob("*.parquet"))
    assert len(backups) == 1 and digest(backups[0]) == before
    assert sum(r.get("added", 0) for r in json.loads((output / "minute-import.json").read_text())) == 480
    after = digest(path)
    import_minutes(root, output)
    assert digest(path) == after
    assert digest(backups[0]) == before
