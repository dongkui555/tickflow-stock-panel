"""Verified QMT 64-byte minute-record layout and exact volume conversion."""

from pathlib import Path

import numpy as np

QMT_DTYPE = np.dtype([
    ("timestamp", "<u4"), ("open", "<u4"), ("high", "<u4"),
    ("low", "<u4"), ("close", "<u4"), ("reserved0", "<u4"),
    ("volume", "<u4"), ("trades", "<u4"), ("amount", "<u8"),
    ("reserved1", "<u4"), ("factor0", "<f4"), ("factor1", "<f4"),
    ("reserved2", "<u4"), ("reserved3", "<u4"), ("reserved4", "<u4"),
])


def qmt_minute_volume_lots(records: np.ndarray) -> np.ndarray:
    # The uint32 at offset 56 stores the remainder in shares (0..99).
    # Keeping only the whole lots loses odd shares and distorts VWAP.
    remainder = records["reserved3"]
    if np.any(remainder >= 100):
        raise ValueError("QMT minute volume remainder outside the verified 0..99 layout")
    return records["volume"].astype(np.float64) + remainder.astype(np.float64) / 100.0


def read_qmt_minute_records(path: Path, start_epoch: int, end_epoch: int) -> np.ndarray:
    size = path.stat().st_size
    if size < 72 or (size - 8) % QMT_DTYPE.itemsize:
        raise ValueError("QMT minute DAT layout is invalid")
    records = np.memmap(path, dtype=QMT_DTYPE, mode="r", offset=8)
    selected = records[(records["timestamp"] >= start_epoch) & (records["timestamp"] <= end_epoch)]
    return selected[np.argsort(selected["timestamp"])]
