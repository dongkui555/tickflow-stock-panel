"""Read the audited Wind-style SH/SZ CSV archive contract in bounded batches.

Research-only adapter. No order replay, cancellation attribution or live feed.
Prices are integer CNY/10000; quantities are shares. Source B/S is a supplier
label, not independently reconstructed aggressor identity.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import subprocess
import tempfile
import threading
from pathlib import Path

import polars as pl

CODE = "\u4e07\u5f97\u4ee3\u7801"
DAY = "\u81ea\u7136\u65e5"
TIME = "\u65f6\u95f4"
TRADE_PRICE = "\u6210\u4ea4\u4ef7\u683c"
QUANTITY = "\u6210\u4ea4\u6570\u91cf"
SIDE = "BS\u6807\u5fd7"
TRADE_FLAG = "\u6210\u4ea4\u4ee3\u7801"
KINDS = {"quote": "\u884c\u60c5.csv", "trade": "\u9010\u7b14\u6210\u4ea4.csv"}
CONTRACT_VERSION = "wind_csv_sh_sz_research_v1"


def member_symbol(name: str) -> str:
    parts = name.replace("\\", "/").split("/")
    if ".." in parts or name.startswith(("/", "\\")):
        raise ValueError("Unsafe archive member")
    matches = [p for p in parts if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", p)]
    if len(matches) != 1:
        raise ValueError("Ambiguous instrument in archive member")
    return matches[0]


def archive_catalog(path: Path, tar: str = "tar") -> list[tuple[str, int]]:
    if _is_7zip(tar):
        result = subprocess.run(
            [tar, "l", "-slt", "-sccUTF-8", str(path)], capture_output=True, timeout=60
        )
        if result.returncode:
            raise ValueError("Cannot list L2 archive")
        listing = result.stdout.decode("utf-8").split("----------", 1)
        if len(listing) != 2:
            raise ValueError("Unexpected 7-Zip catalog layout")
        entries = []
        for block in re.split(r"\r?\n\s*\r?\n", listing[1].strip()):
            fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
            if fields.get("Attributes", "").startswith("D") or fields.get("Folder") == "+":
                continue
            name = fields.get("Path", "")
            if name.replace("\\", "/").split("/")[-1] not in KINDS.values():
                continue
            member_symbol(name)
            entries.append((name, int(fields["Size"])))
        return entries
    result = subprocess.run([tar, "-tvf", str(path)], capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError("Cannot list L2 archive")
    entries = []
    for line in result.stdout.decode("gb18030").splitlines():
        fields = line.split(maxsplit=8)
        if len(fields) != 9:
            raise ValueError("Unexpected bsdtar catalog layout")
        if fields[0].startswith("-"):
            if fields[8].replace("\\", "/").split("/")[-1] not in KINDS.values():
                continue
            member_symbol(fields[8])
            entries.append((fields[8], int(fields[4])))
    return entries


def _is_7zip(executable: str) -> bool:
    return Path(executable).stem.lower() in {"7z", "7zr", "7za", "7zz"}


class _HashReader(io.RawIOBase):
    def __init__(self, raw):
        self.raw = raw
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        size = self.raw.readinto(buffer)
        if size:
            self.digest.update(memoryview(buffer)[:size])
            self.bytes_read += size
        return size


def csv_batches(stream, symbols: set[str], batch_rows: int = 8192):
    """Parse concatenated same-kind members emitted by ONE archive traversal."""
    if batch_rows < 1 or batch_rows > 65536:
        raise ValueError("batch_rows must be within 1..65536")
    header, rows, seen = None, [], set()
    reader = csv.reader(stream)
    current_symbol = None
    for row in reader:
        if not row:
            continue
        if row[0].lstrip("\ufeff") == CODE:
            if header is not None and current_symbol is None:
                raise ValueError("Empty L2 member")
            row[0] = CODE
            if header is not None and row != header:
                raise ValueError("Mixed CSV schemas")
            if len(set(row)) != len(row) or not {CODE, DAY, TIME} <= set(row):
                raise ValueError("Invalid L2 header")
            header, current_symbol = row, None
            continue
        if header is None or len(row) != len(header):
            raise ValueError("Malformed L2 CSV row")
        symbol = row[header.index(CODE)]
        if symbol not in symbols:
            raise ValueError("Unexpected L2 instrument")
        if current_symbol is None:
            if symbol in seen:
                raise ValueError("Duplicate L2 member")
            current_symbol = symbol
            seen.add(symbol)
        if symbol != current_symbol:
            raise ValueError("Mixed instruments inside a CSV member")
        rows.append(row)
        if len(rows) >= batch_rows:
            yield pl.DataFrame(rows, schema={c: pl.String for c in header}, orient="row")
            rows = []
    if seen != symbols or current_symbol is None:
        raise ValueError(
            f"Incomplete L2 member stream: missing={sorted(symbols - seen)}, "
            f"empty_last_member={current_symbol is None}"
        )
    if rows:
        yield pl.DataFrame(rows, schema={c: pl.String for c in header}, orient="row")


def stream_archive(
    path: Path,
    entries: list[tuple[str, int]],
    symbols: set[str],
    kind: str,
    evidence: dict,
    *,
    batch_rows=8192,
    timeout=600,
    tar="tar",
):
    """Yield batches without extracting raw files; kill subprocess on cancellation."""
    names = [
        n
        for n, _ in entries
        if n.replace("\\", "/").split("/")[-1] == KINDS[kind] and member_symbol(n) in symbols
    ]
    if len(names) != len(symbols) or {member_symbol(n) for n in names} != symbols:
        raise ValueError("Missing or duplicate L2 source members")
    command = (
        [tar, "x", "-so", "-bso0", "-bsp0", str(path), *names]
        if _is_7zip(tar)
        else [tar, "-xOqf", str(path), *names]
    )
    if sum(len(p) + 3 for p in command) > 28000:
        raise ValueError("Too many members for one bounded extraction; reduce universe")
    expected_bytes = sum(size for n, size in entries if n in set(names))
    timed_out = threading.Event()
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)

        def abort():
            timed_out.set()
            proc.kill()

        timer = threading.Timer(timeout, abort)
        timer.daemon = True
        timer.start()
        hashed = _HashReader(proc.stdout)
        text = io.TextIOWrapper(io.BufferedReader(hashed, 65536), encoding="gb18030", newline="")
        try:
            yield from csv_batches(text, symbols, batch_rows)
            result = proc.wait(timeout=10)
            if timed_out.is_set() or result:
                raise ValueError("L2 archive extraction timed out or failed")
            if hashed.bytes_read != expected_bytes:
                raise ValueError("Archive output byte count differs from catalog")
            evidence.update(
                kind=kind,
                members=names,
                bytes_read=hashed.bytes_read,
                stream_sha256=hashed.digest.hexdigest(),
            )
        except ValueError as exc:
            # EOF can precede process exit by a few milliseconds. Preserve the
            # reader's actual CRC/I/O failure instead of reporting only missing CSVs.
            try:
                result = proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                result = None
            if result not in (None, 0):
                errors.seek(0)
                payload = errors.read(2048)
                try:
                    detail = payload.decode("utf-8").strip()
                except UnicodeDecodeError:
                    detail = payload.decode("gb18030", errors="replace").strip()
                raise ValueError(f"L2 archive reader failed ({proc.returncode}): {detail}") from exc
            raise
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
            text.close()
            proc.stdout.close()
