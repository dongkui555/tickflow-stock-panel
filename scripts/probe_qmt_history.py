"""Probe QMT's non-blocking history supply path through the research kernel."""

from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.qmt_bridge.qmt_http_api import DEFAULT_CONFIG, QmtKernel, _KERNEL_HELPERS, _kline_code


def supply_code(symbol: str, start: str, end: str) -> str:
    return _KERNEL_HELPERS + f'''\n
from xtquant import xtdata as _qxtdata
_qclient = _qxtdata.get_client()
_qclient.supply_history_data({symbol!r}, "1m", {start!r}, {end!r})
_qemit({{"submitted": True, "connected": bool(_qclient.is_connected())}})
'''


def main() -> None:
    symbol = "000001.SZ"
    day = "20260909"
    start, end = day + "093000", day + "150000"
    kernel = QmtKernel("http://127.0.0.1:8888", DEFAULT_CONFIG, timeout=15)
    try:
        print("submit", kernel.execute_json(supply_code(symbol, start, end)))
        for attempt in range(1, 7):
            time.sleep(5)
            rows = kernel.execute_json(_kline_code(symbol, "1m", start, end, -1))
            print("poll", attempt, "rows", len(rows))
            if rows:
                print(rows[0], rows[-1])
                break
    finally:
        kernel.stop()


if __name__ == "__main__":
    main()
