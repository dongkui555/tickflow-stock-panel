"""Download and import QMT historical 1-minute bars through its kernel.

The installed QMT 3.6.x client may return from ``supply_history_data`` without
populating the standalone ``xtdata`` cache.  This runner uses the authenticated
research kernel, submits a non-blocking supply request, polls the same kernel
cache, and pushes only rows that pass the bridge validation.  It deliberately
does not synthesize missing minutes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qmt_bridge.qmt_http_api import (  # noqa: E402
    DEFAULT_CONFIG,
    QmtKernel,
    _KERNEL_HELPERS,
    _kline_code,
)


DEFAULT_BRIDGE = "http://127.0.0.1:8188/api/push/minutes"


def submit_code(symbol: str, start: str, end: str) -> str:
    return _KERNEL_HELPERS + f'''\n
from xtquant import xtdata as _qxtdata
_qclient = _qxtdata.get_client()
_qclient.supply_history_data({symbol!r}, "1m", {start!r}, {end!r})
_qemit({{"submitted": True, "connected": bool(_qclient.is_connected())}})
'''


def parse_day(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y%m%d").date()


def load_symbols(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    if args.symbols:
        values.extend(args.symbols.replace(";", ",").split(","))
    if args.symbols_file:
        for line in Path(args.symbols_file).read_text(encoding="utf-8").splitlines():
            values.extend(line.replace(";", ",").split(","))
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        symbol = value.strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    if not result:
        raise ValueError("请通过 --symbols 或 --symbols-file 提供股票代码")
    return result


def push_rows(rows: list[dict], url: str) -> None:
    api_key = os.getenv("QMT_BRIDGE_API_KEY", "").strip()
    for offset in range(0, len(rows), 5000):
        chunk = rows[offset:offset + 5000]
        payload = json.dumps(
            {"source": "iquant_kernel_history", "rows": chunk},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        if api_key:
            request.add_header("X-API-Key", api_key)
        with urlopen(request, timeout=60) as response:
            if not 200 <= int(response.getcode()) < 300:
                raise RuntimeError("bridge HTTP %s" % response.getcode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--symbols", help="逗号分隔的 QMT 代码，例如 000001.SZ")
    parser.add_argument("--symbols-file", help="每行一个 QMT 代码")
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--poll-rounds", type=int, default=6)
    parser.add_argument("--kernel-url", default="http://127.0.0.1:8888")
    parser.add_argument("--bridge-url", default=DEFAULT_BRIDGE)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    start, end = parse_day(args.start), parse_day(args.end)
    if end < start:
        raise ValueError("end 不能早于 start")
    if args.poll_seconds < 1 or args.poll_rounds < 1:
        raise ValueError("poll-seconds 和 poll-rounds 必须为正数")
    symbols = load_symbols(args)
    kernel = QmtKernel(args.kernel_url, DEFAULT_CONFIG, timeout=15)
    try:
        day = start
        while day <= end:
            if day.weekday() >= 5:
                day += dt.timedelta(days=1)
                continue
            start_time = day.strftime("%Y%m%d093000")
            end_time = day.strftime("%Y%m%d150000")
            for index, symbol in enumerate(symbols, 1):
                print("%s %s/%s submit %s" % (day, index, len(symbols), symbol), flush=True)
                result = kernel.execute_json(submit_code(symbol, start_time, end_time))
                print("  %s" % result, flush=True)
                rows: list[dict] = []
                for attempt in range(1, args.poll_rounds + 1):
                    time.sleep(args.poll_seconds)
                    rows = list(kernel.execute_json(
                        _kline_code(symbol, "1m", start_time, end_time, -1)
                    ))
                    print("  poll %s/%s rows=%s" % (attempt, args.poll_rounds, len(rows)), flush=True)
                    if rows:
                        break
                full = len(rows) == 240
                if full and not args.no_push:
                    push_rows(rows, args.bridge_url)
                    print("  pushed=240", flush=True)
                elif rows:
                    print("  retained locally only; incomplete=%s" % (not full), flush=True)
                else:
                    print("  unavailable from QMT", flush=True)
            day += dt.timedelta(days=1)
    finally:
        kernel.stop()


if __name__ == "__main__":
    main()
