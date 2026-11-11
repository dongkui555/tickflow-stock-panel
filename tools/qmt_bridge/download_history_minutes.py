"""Download historical 1-minute bars inside the QMT Python environment.

This file must run where ``xtquant`` is available, usually QMT/iQuant's
strategy Python environment. It is intentionally independent from Docker.
QMT downloads into its own cache first; validated rows are then pushed to the
local TickFlow bridge for the application-side checks and import.
"""

from __future__ import print_function

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import time

try:
    from urllib.request import Request, urlopen
except ImportError:  # pragma: no cover - Python 2 compatibility for old QMT
    from urllib2 import Request, urlopen

FIELDS = ["open", "high", "low", "close", "volume", "amount"]
DEFAULT_BRIDGE = "http://127.0.0.1:8188/api/push/minutes"
DEFAULT_SECTORS = [u"沪深A股", u"北证A股"]
LOG_FILE = None
DOWNLOAD_WAIT = 5


def log(message):
    line = "[%s] %s" % (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
    print(line)
    if LOG_FILE:
        try:
            with open(LOG_FILE, "a") as handle:
                handle.write(line + "\n")
        except Exception:
            pass


def clean(value):
    if value is None:
        return None
    try:
        value = value.item()
    except Exception:
        pass
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
    except Exception:
        pass
    return value


def number(value):
    value = clean(value)
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def normalize_symbol(value):
    text = str(value or "").strip().upper()
    match = re.match(r"^(SH|SZ|BJ)(\d{6})$", text)
    if match:
        return "%s.%s" % (match.group(2), match.group(1))
    if re.match(r"^\d{6}\.(SH|SZ|BJ)$", text):
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) != 6:
        return None
    if digits.startswith(("4", "8", "92")):
        market = "BJ"
    elif digits.startswith(("5", "6", "9")):
        market = "SH"
    else:
        market = "SZ"
    return "%s.%s" % (digits, market)


def parse_datetime(value):
    if value is None:
        return None
    try:
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if isinstance(value, dt.datetime):
            return value.replace(tzinfo=None)
        if isinstance(value, dt.date):
            return dt.datetime.combine(value, dt.time())
    except Exception:
        pass
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    try:
        if len(digits) >= 14:
            return dt.datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
        if len(digits) == 12:
            return dt.datetime.strptime(digits, "%Y%m%d%H%M")
        if len(digits) == 8:
            return dt.datetime.strptime(digits, "%Y%m%d")
        value = float(text)
        if value > 10000000000:
            value /= 1000.0
        if value > 1000000000:
            return dt.datetime.fromtimestamp(value)
    except (TypeError, ValueError, OSError):
        return None
    return None


def minute_label(value):
    value = value.replace(second=0, microsecond=0)
    if value.strftime("%H%M") in ("0930", "1300"):
        return value + dt.timedelta(minutes=1)
    return value


def in_session(value):
    hm = value.hour * 60 + value.minute
    return 570 <= hm <= 690 or 780 <= hm <= 900


def valid_row(row):
    prices = [row.get(key) for key in ("open", "high", "low", "close")]
    if any(value is None or value <= 0 for value in prices):
        return False
    if row.get("volume") is None or row.get("volume") < 0:
        return False
    if row.get("amount") is None or row.get("amount") < 0:
        return False
    return (
        row["high"] >= max(row["open"], row["close"])
        and row["low"] <= min(row["open"], row["close"])
    )


def frame_rows(symbol, frame, day):
    symbol = normalize_symbol(symbol)
    if not symbol or frame is None or not hasattr(frame, "iterrows"):
        return []
    rows = {}
    try:
        iterator = frame.iterrows()
        for index, raw in iterator:
            when = parse_datetime(index)
            if when is None or when.date() != day or not in_session(when):
                continue
            when = minute_label(when)
            row = {"symbol": symbol, "datetime": when.strftime("%Y-%m-%d %H:%M:%S")}
            for key in FIELDS:
                try:
                    value = raw.get(key)
                except Exception:
                    value = None
                row[key] = number(value)
            if valid_row(row):
                rows[row["datetime"]] = row
    except Exception as exc:
        log("%s parse failed: %s %s" % (symbol, type(exc).__name__, exc))
    return list(rows.values())


def load_symbols(args):
    values = []
    if args.symbols:
        values.extend(re.split(r"[ ,;\r\n]+", args.symbols))
    if args.symbols_file:
        with open(args.symbols_file, "r") as handle:
            for line in handle:
                values.extend(re.split(r"[ ,;\r\n]+", line.strip()))
    if not values:
        getter = getattr(xtdata, "get_stock_list_in_sector", None)
        if getter:
            for sector in DEFAULT_SECTORS:
                try:
                    values.extend(getter(sector) or [])
                except Exception as exc:
                    log("load sector %s failed: %s" % (sector, exc))
    result = []
    seen = set()
    for value in values:
        symbol = normalize_symbol(value)
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    if not result:
        raise RuntimeError("no symbols; pass --symbols or --symbols-file")
    return result


def load_target_days(path):
    """Read a durable gap list with date,symbol columns."""
    targets = {}
    with open(path, "r", newline="") as handle:
        for row in csv.DictReader(handle):
            day = str(row.get("date") or "").replace("-", "")[:8]
            symbol = normalize_symbol(row.get("symbol") or row.get("code"))
            if len(day) == 8 and symbol:
                targets.setdefault(day, set()).add(symbol)
    return targets


def download_batch(symbols, day, incremental):
    start = day.strftime("%Y%m%d093000")
    end = day.strftime("%Y%m%d150000")
    # QMT 3.6.8's batch API waits indefinitely when its progress callback
    # never arrives. The single-symbol API is the safer compatibility path.
    function = getattr(xtdata, "download_history_data")
    for symbol in symbols:
        log("%s download request %s %s-%s" % (day, symbol, start, end))
        function(symbol, "1m", start, end)
        if DOWNLOAD_WAIT:
            time.sleep(DOWNLOAD_WAIT)
        log("%s download returned %s" % (day, symbol))


def _field_value(frame, row_index, column):
    try:
        return frame.loc[row_index, column]
    except Exception:
        try:
            return frame.iloc[row_index, frame.columns.get_loc(column)]
        except Exception:
            return None


def _rows_from_field_frames(data, symbols, day):
    """Handle xtdata's field -> DataFrame return shape."""
    field_frames = {
        field: data.get(field) for field in FIELDS
        if hasattr(data.get(field), "columns") and hasattr(data.get(field), "index")
    }
    close_frame = field_frames.get("close")
    if close_frame is None:
        return [], {}
    rows = []
    counts = {}
    for position, raw_symbol in enumerate(symbols):
        symbol = normalize_symbol(raw_symbol)
        if not symbol:
            continue
        try:
            row_index = raw_symbol if raw_symbol in close_frame.index else position
        except Exception:
            row_index = position
        symbol_rows = {}
        for column in close_frame.columns:
            when = parse_datetime(column)
            if when is None or when.date() != day or not in_session(when):
                continue
            when = minute_label(when)
            item = {"symbol": symbol, "datetime": when.strftime("%Y-%m-%d %H:%M:%S")}
            for field in FIELDS:
                frame = field_frames.get(field)
                item[field] = number(_field_value(frame, row_index, column)) if frame is not None else None
            if valid_row(item):
                symbol_rows[item["datetime"]] = item
        if symbol_rows:
            rows.extend(symbol_rows.values())
            counts[symbol] = len(symbol_rows)
    return rows, counts


def read_batch(symbols, day):
    start = day.strftime("%Y%m%d093000")
    end = day.strftime("%Y%m%d150000")
    # xtdata 3.x does not accept subscribe; the QMT cache has already been
    # populated by download_history_data2 before this read.
    data = xtdata.get_market_data_ex(
        FIELDS, symbols, period="1m", start_time=start, end_time=end,
        count=-1, dividend_type="none", fill_data=False,
    )
    if not isinstance(data, dict):
        return [], {}
    # ContextInfo-style runtimes return symbol -> DataFrame. Standalone
    # xtdata returns field -> DataFrame. Support both without guessing.
    if any(normalize_symbol(key) in [normalize_symbol(item) for item in symbols]
           for key in data.keys()):
        rows = []
        counts = {}
        for raw_symbol, frame in data.items():
            symbol_rows = frame_rows(raw_symbol, frame, day)
            if symbol_rows:
                rows.extend(symbol_rows)
                counts[normalize_symbol(raw_symbol)] = len(symbol_rows)
        return rows, counts
    return _rows_from_field_frames(data, symbols, day)


def push_rows(rows, url):
    api_key = os.environ.get("QMT_BRIDGE_API_KEY", "").strip()
    for offset in range(0, len(rows), 5000):
        chunk = rows[offset:offset + 5000]
        payload = json.dumps(
            {"source": "iquant_history", "rows": chunk},
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if api_key:
            headers["X-API-Key"] = api_key
        response = urlopen(Request(url, data=payload, headers=headers), timeout=60)
        try:
            if not 200 <= int(response.getcode()) < 300:
                raise RuntimeError("bridge HTTP %s" % response.getcode())
        finally:
            response.close()


def parse_day(value):
    return dt.datetime.strptime(value, "%Y%m%d").date()


def main():
    global xtdata
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--symbols", help="comma-separated QMT symbols")
    parser.add_argument("--symbols-file", help="one symbol per line")
    parser.add_argument("--targets-file", help="CSV with date,symbol; only request listed stock-days")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=int, default=5)
    parser.add_argument("--download-wait", type=int, default=5,
                        help="seconds to wait after QMT schedules a download")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--bridge-url", default=DEFAULT_BRIDGE)
    parser.add_argument("--log-file", help="append progress to this file")
    parser.add_argument("--read-only", action="store_true",
                        help="read QMT local cache without requesting a download")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--incremental", action="store_true",
                        help="use QMT incremental download; omit for first full pull")
    args = parser.parse_args()
    if args.log_file:
        global LOG_FILE
        LOG_FILE = args.log_file
    global DOWNLOAD_WAIT
    DOWNLOAD_WAIT = max(0, args.download_wait)
    try:
        from xtquant import xtdata as qmt_xtdata
    except ImportError:
        raise RuntimeError(
            "xtquant is unavailable; run this script in the QMT/iQuant Python environment"
        )
    xtdata = qmt_xtdata
    if args.batch_size < 1 or args.retries < 1:
        raise ValueError("batch-size and retries must be positive")
    start, end = parse_day(args.start), parse_day(args.end)
    if end < start:
        raise ValueError("end must not be before start")
    symbols = load_symbols(args)
    target_days = load_target_days(args.targets_file) if args.targets_file else None
    log("QMT history start: %s to %s, symbols=%s, batch=%s, incremental=%s" %
        (start, end, len(symbols), args.batch_size, args.incremental))
    day = start
    while day <= end:
        if day.weekday() >= 5:
            day += dt.timedelta(days=1)
            continue
        day_symbols = symbols
        if target_days is not None:
            day_symbols = sorted(target_days.get(day.strftime("%Y%m%d"), set()))
            if not day_symbols:
                day += dt.timedelta(days=1)
                continue
        day_total = day_full = 0
        for offset in range(0, len(day_symbols), args.batch_size):
            batch = day_symbols[offset:offset + args.batch_size]
            rows, counts = [], {}
            for attempt in range(1, args.retries + 1):
                try:
                    if not args.read_only:
                        download_batch(batch, day, args.incremental)
                    rows, counts = read_batch(batch, day)
                    if rows:
                        break
                except Exception as exc:
                    log("%s batch %s-%s attempt %s failed: %s %s" %
                        (day, offset + 1, offset + len(batch), attempt,
                         type(exc).__name__, exc))
                if attempt < args.retries:
                    time.sleep(max(0, args.retry_wait))
            full = sum(1 for value in counts.values() if value >= 240)
            day_total += len(counts)
            day_full += full
            if rows and not args.no_push and not args.read_only:
                push_rows(rows, args.bridge_url)
            log("%s batch %s-%s: rows=%s symbols=%s full240=%s%s" %
                (day, offset + 1, offset + len(batch), len(rows), len(counts), full,
                 " pushed" if rows and not args.no_push and not args.read_only else ""))
            time.sleep(max(0.0, args.sleep))
        status = "complete" if day_full == len(day_symbols) else "partial"
        log("%s finished: %s, symbols=%s/%s, full240=%s/%s" %
            (day, status, day_total, len(day_symbols), day_full, len(day_symbols)))
        day += dt.timedelta(days=1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("FAILED: %s %s" % (type(exc).__name__, exc))
        raise
