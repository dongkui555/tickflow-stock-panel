"""Polars parquet helpers."""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

_PARQUET_MAGIC = b"PAR1"

DAILY_STORAGE_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "amount": pl.Float64,
    "quote_ts": pl.Int64,
}

ENRICHED_STORAGE_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "amount": pl.Float64,
    "raw_close": pl.Float64,
    "raw_high": pl.Float64,
    "raw_low": pl.Float64,
    "turnover_rate": pl.Float64,
    "consecutive_limit_ups": pl.UInt32,
    "consecutive_limit_downs": pl.UInt32,
    "quote_ts": pl.Int64,
}


def scan_parquet_compat(source: Any, **kwargs: Any) -> pl.LazyFrame:
    """Scan partitioned parquet while tolerating additive schema changes."""
    kwargs.setdefault("missing_columns", "insert")
    kwargs.setdefault("extra_columns", "ignore")
    return pl.scan_parquet(source, **kwargs)


def scan_daily_parquet(source: Any, **kwargs: Any) -> pl.LazyFrame:
    kwargs.setdefault("schema", DAILY_STORAGE_SCHEMA)
    kwargs.setdefault("cast_options", pl.ScanCastOptions(integer_cast="allow-float"))
    return scan_parquet_compat(source, **kwargs)


def scan_enriched_parquet(source: Any, **kwargs: Any) -> pl.LazyFrame:
    kwargs.setdefault("schema", ENRICHED_STORAGE_SCHEMA)
    kwargs.setdefault("cast_options", pl.ScanCastOptions(integer_cast="allow-float"))
    return scan_parquet_compat(source, **kwargs)


def is_valid_parquet(path: str | Path) -> bool:
    """快速检查 Parquet 文件的头尾标记和最小长度。"""
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size < 12:
            return False
        with p.open("rb") as fh:
            if fh.read(4) != _PARQUET_MAGIC:
                return False
            fh.seek(-4, 2)
            return fh.read(4) == _PARQUET_MAGIC
    except (OSError, ValueError):
        return False


def quarantine_invalid_parquet(
    path: str | Path,
    *,
    reason: str = "invalid parquet",
) -> Path | None:
    """将损坏文件改名隔离，避免 Polars/DuckDB/PyArrow 数据集扫描继续读到。"""
    p = Path(path)
    if not p.exists():
        return None
    target = _ignored_corrupt_path(p)
    try:
        p.replace(target)
    except OSError as exc:
        logger.warning("无法隔离损坏 Parquet %s: %s", p, exc)
        return None
    logger.warning("已隔离损坏 Parquet (%s): %s -> %s", reason, p, target)
    return target


def _ignored_corrupt_path(path: Path) -> Path:
    """生成以 '_' 开头的隔离文件名，PyArrow dataset 默认会忽略。"""
    name = path.name
    if name.startswith(("_", ".")):
        name = name.lstrip("_.") or "part.parquet"
    target = path.with_name(f"_{name}.corrupt-{time.time_ns()}")
    while target.exists():
        target = path.with_name(f"_{name}.corrupt-{time.time_ns()}-{uuid.uuid4().hex[:8]}")
    return target


def _hide_legacy_corrupt_marker(path: Path) -> Path | None:
    """兼容旧版 part.parquet.corrupt-* 隔离文件，改为 PyArrow 可忽略名称。"""
    if path.name.startswith(("_", ".")):
        return None
    target = path.with_name(f"_{path.name}")
    while target.exists():
        target = path.with_name(f"_{path.name}-{uuid.uuid4().hex[:8]}")
    try:
        path.replace(target)
    except OSError as exc:
        logger.warning("无法迁移旧隔离 Parquet %s: %s", path, exc)
        return None
    logger.warning("已迁移旧隔离 Parquet: %s -> %s", path, target)
    return target


def repair_parquet_tree(root: str | Path) -> list[Path]:
    """扫描目录并隔离头尾不完整的 Parquet 文件。"""
    base = Path(root)
    if not base.exists():
        return []
    quarantined: list[Path] = []
    for path in base.rglob("*.parquet.corrupt-*"):
        moved = _hide_legacy_corrupt_marker(path)
        if moved is not None:
            quarantined.append(moved)
    for path in base.rglob("*.parquet"):
        if not is_valid_parquet(path):
            moved = quarantine_invalid_parquet(path)
            if moved is not None:
                quarantined.append(moved)
    return quarantined


def atomic_write_parquet(df: pl.DataFrame, out: str | Path) -> None:
    """先写临时文件，再校验并原子替换目标路径。"""
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        df.write_parquet(tmp)
        if not is_valid_parquet(tmp):
            raise IOError(f"Parquet 写入后校验失败: {tmp}")
        tmp.replace(target)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def read_parquet_safe(path: str | Path, **kwargs: Any) -> pl.DataFrame | None:
    """读取单个 Parquet；损坏时自动隔离并返回 None。"""
    p = Path(path)
    if not is_valid_parquet(p):
        quarantine_invalid_parquet(p, reason="header/footer 校验失败")
        return None
    try:
        return pl.read_parquet(p, **kwargs)
    except Exception as exc:  # noqa: BLE001
        quarantine_invalid_parquet(p, reason=f"解析失败: {exc}")
        return None
