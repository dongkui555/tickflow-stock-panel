from pathlib import Path

import polars as pl

from app.parquet import (
    atomic_write_parquet,
    is_valid_parquet,
    read_parquet_safe,
    repair_parquet_tree,
)


def test_atomic_write_parquet_writes_a_valid_file(tmp_path: Path) -> None:
    target = tmp_path / "part.parquet"

    atomic_write_parquet(pl.DataFrame({"symbol": ["000001.SZ"], "close": [10.5]}), target)

    assert is_valid_parquet(target)
    assert read_parquet_safe(target) is not None


def test_repair_parquet_tree_quarantines_truncated_file(tmp_path: Path) -> None:
    target = tmp_path / "date=2026-08-31" / "part.parquet"
    target.parent.mkdir()
    target.write_bytes(b"")

    repaired = repair_parquet_tree(tmp_path)

    assert len(repaired) == 1
    assert not target.exists()
    assert repaired[0].name.startswith("_part.parquet.corrupt-")


def test_read_parquet_safe_quarantines_parse_failure(tmp_path: Path) -> None:
    target = tmp_path / "broken.parquet"
    target.write_bytes(b"PAR1" + b"not-a-parquet-file" + b"PAR1")

    assert read_parquet_safe(target) is None
    assert not target.exists()
    assert len(list(tmp_path.glob("_broken.parquet.corrupt-*"))) == 1


def test_repair_parquet_tree_hides_legacy_corrupt_marker(tmp_path: Path) -> None:
    target = tmp_path / "date=2026-08-31" / "part.parquet.corrupt-manual"
    target.parent.mkdir()
    target.write_bytes(b"")

    repaired = repair_parquet_tree(tmp_path)

    assert len(repaired) == 1
    assert not target.exists()
    assert (target.parent / "_part.parquet.corrupt-manual").exists()
