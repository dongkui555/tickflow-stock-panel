"""Read-only probe for a locally running QMT/iQuant quote service."""

from __future__ import print_function

import datetime
import json
import os
import traceback

from xtquant import xtdata


def normalize(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def frame_summary(frame):
    if frame is None or getattr(frame, "empty", True):
        return {"rows": 0}
    tail = frame.tail(3)
    return {
        "rows": int(len(frame)),
        "first_time": str(frame.index[0]),
        "last_time": str(frame.index[-1]),
        "columns": list(frame.columns),
        "tail": normalize(tail.reset_index().to_dict("records")),
    }


def main():
    symbol = "000001.SZ"
    today = datetime.datetime.now().strftime("%Y%m%d")
    start = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y%m%d")
    client = xtdata.get_client()
    result = {
        "ok": bool(client.is_connected()),
        "symbol": symbol,
    }
    try:
        result["quote"] = normalize(xtdata.get_full_tick([symbol]).get(symbol, {}))
    except Exception as exc:
        result["quote_error"] = str(exc)
    for period, key in (("1m", "minute_1m"), ("1d", "daily_1d")):
        try:
            frame = xtdata.get_market_data_ex(
                [], [symbol], period=period, start_time=start, end_time=today,
                count=-1, dividend_type="none", fill_data=False,
            ).get(symbol)
            result[key] = frame_summary(frame)
        except Exception as exc:
            result[key + "_error"] = str(exc)
    emit(result)


def emit(result):
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    output_path = os.environ.get("QMT_PROBE_OUTPUT")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit({
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
