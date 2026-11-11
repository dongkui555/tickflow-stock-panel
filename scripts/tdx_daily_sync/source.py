"""Read the Windows LC1 cache and fetch a checksum-bound TDX finance increment."""

from __future__ import annotations

import hashlib
import re
import struct
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


LC1 = struct.Struct("<HHfffffII")
CATALOG = "https://data.tdx.com.cn/tdxfin/gpcw.txt"
PACKAGE_ROOT = "http://data.tdx.com.cn/tdxfin/"
SYMBOL = re.compile(r"([0-9]{6})\.(SH|SZ|BJ)\Z")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def day_records(root: Path, symbol: str, day: date) -> tuple[bytes, str]:
    identity = SYMBOL.fullmatch(symbol)
    if identity is None:
        raise ValueError("Invalid LC1 identity")
    code, market = identity.group(1), identity.group(2).lower()
    path = root / market / "minline" / f"{market}{code}.lc1"
    try:
        before = path.stat()
        if before.st_size % LC1.size:
            return b"", "source_incomplete"
        with path.open("rb") as stream:
            stream.seek(max(0, before.st_size - 512 * LC1.size))
            tail = stream.read()
        after = path.stat()
    except FileNotFoundError:
        return b"", "source_empty"
    except OSError:
        return b"", "source_unavailable"
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        return b"", "source_pending"
    encoded = (day.year - 2004) * 2048 + day.month * 100 + day.day
    if tail and len(tail) < before.st_size and struct.unpack_from("<H", tail)[0] >= encoded:
        return b"", "source_incomplete"
    records = b"".join(
        tail[i:i + LC1.size]
        for i in range(0, len(tail), LC1.size)
        if struct.unpack_from("<H", tail, i)[0] == encoded
    )
    return records, "captured" if records else "source_empty"


def latest_financial_package(catalog: str, day: date) -> dict:
    rows = {}
    for line in catalog.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"gpcw(\d{8})\.zip,([a-f0-9]{32}),(\d+)", line.strip())
        if not match:
            raise ValueError("Invalid TDX financial catalog row")
        period, digest, size = match.groups()
        report_date = date.fromisoformat(period)
        size = int(size)
        if report_date in rows:
            raise ValueError("Duplicate period in TDX catalog")
        rows[report_date] = {"name": f"gpcw{period}.zip", "period": str(report_date), "md5": digest, "size": size}
    return max((row for report_date, row in rows.items() if report_date < day and 100_000 <= row["size"] <= 25_000_000),
               key=lambda row: row["period"])


def verify_package(blob: bytes, package: dict) -> None:
    if len(blob) != package["size"] or hashlib.md5(blob, usedforsecurity=False).hexdigest() != package["md5"]:
        raise ValueError("TDX package did not match the HTTPS catalog")


def fetch_financial(day: date, local: Path, directory: Path) -> dict:
    with urlopen(CATALOG, timeout=20) as response:
        if urlsplit(response.geturl()).hostname != "data.tdx.com.cn" or not response.geturl().startswith("https://"):
            raise ValueError("Unexpected TDX catalog location")
        catalog = response.read(64_001)
    if len(catalog) > 64_000:
        raise ValueError("TDX catalog is unexpectedly large")
    package = latest_financial_package(catalog.decode("ascii"), day)
    cached = local / "cw" / package["name"]
    blob = cached.read_bytes() if cached.is_file() and cached.stat().st_size == package["size"] else b""
    try:
        verify_package(blob, package)
        origin = "tdx_client_cache"
    except ValueError:
        # The client distribution endpoint serves ZIP over HTTP. Validate against
        # the separately retrieved HTTPS catalog before decoding any member.
        with urlopen(PACKAGE_ROOT + package["name"], timeout=35) as response:
            if urlsplit(response.geturl()).hostname != "data.tdx.com.cn":
                raise ValueError("Unexpected TDX package location")
            blob = response.read(package["size"] + 1)
        verify_package(blob, package)
        origin = "tdx_incremental_distribution"
    (directory / package["name"]).write_bytes(blob)
    return {**package, "origin": origin, "sha256": hashlib.sha256(blob).hexdigest(),
            "catalog_sha256": hashlib.sha256(catalog).hexdigest()}
