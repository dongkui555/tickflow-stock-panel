# coding: gbk
"""Read-only TickFlow collector for Guosen iQuant strategy runtime.

Import this file as a Python strategy, set the default period to 1 minute, and
run it in simulation mode. It never calls a trading or account API.
"""
from __future__ import print_function

import json
import os
import re
import threading
import time
import datetime as dt
try:
    from urllib.request import Request, urlopen
except ImportError:
    from urllib2 import Request, urlopen


BRIDGE_URL = "http://127.0.0.1:8188/api/push/quotes"
MINUTE_BRIDGE_URL = "http://127.0.0.1:8188/api/push/minutes"
INDEX_UNIVERSE_URL = "http://127.0.0.1:3018/api/index/universe"
INDEX_DAILY_BRIDGE_URL = "http://127.0.0.1:8188/api/push/daily"
CORE_SYMBOLS = ["000001.SH", "399001.SZ", "399006.SZ", "000680.SH",
                "000001.SZ", "600000.SH"]
MARKETS = ["SH", "SZ", "BJ"]
FLUSH_SECONDS = 2.0
CHUNK_SIZE = 500
MINUTE_POST_CHUNK = 5000
HISTORY_BATCH_SIZE = 25
HISTORY_BATCHES_PER_RUN = 1
HISTORY_LOOKBACK_DAYS = 2
HISTORY_INTERVAL_SECONDS = 60
INDEX_BATCH_SIZE = 20
INDEX_BATCHES_PER_RUN = 1
INDEX_LOOKBACK_DAYS = 30
INDEX_INTERVAL_SECONDS = 60

_pending = {}
_lock = threading.Lock()
_sender_running = False
_last_send_started = 0.0
_subscription_ids = []
_last_runtime_error = 0.0
_history_universe = []
_history_cursor = 0
_history_window_key = ""
_history_attempts = {}
_history_complete_logged = ""
_history_busy = False
_history_started = False
_index_universe = []
_index_cursor = 0
_index_window_key = ""
_index_attempts = {}
_index_complete_logged = ""
_index_busy = False


def _log(message):
    line = "%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), message)
    print(line.rstrip())
    try:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "TickFlow_QMT_Bridge.runtime.log")
        with open(log_path, "a") as log_file:
            log_file.write(line)
    except Exception:
        pass


def _clean(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _clean(value.item())
        except Exception:
            pass
    return str(value)


def _rows_from_callback(data):
    rows = []
    if not isinstance(data, dict):
        return rows
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        row = _clean(value)
        row["symbol"] = str(row.get("stockCode") or row.get("symbol") or key).upper()
        rows.append(row)
    return rows


def _post_rows(rows):
    payload = json.dumps({"source": "iquant_context", "rows": rows},
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = Request(BRIDGE_URL, data=payload,
                      headers={"Content-Type": "application/json; charset=utf-8"})
    response = urlopen(request, timeout=8)
    try:
        return 200 <= int(response.getcode()) < 300
    finally:
        response.close()


def _post_minute_rows(rows):
    for offset in range(0, len(rows), MINUTE_POST_CHUNK):
        chunk = rows[offset:offset + MINUTE_POST_CHUNK]
        payload = json.dumps({"source": "iquant_history", "rows": chunk},
                             ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(MINUTE_BRIDGE_URL, data=payload,
                          headers={"Content-Type": "application/json; charset=utf-8"})
        response = urlopen(request, timeout=30)
        try:
            if not 200 <= int(response.getcode()) < 300:
                raise RuntimeError("minute bridge returned HTTP %s" % response.getcode())
        finally:
            response.close()


def _post_daily_rows(rows):
    for offset in range(0, len(rows), MINUTE_POST_CHUNK):
        chunk = rows[offset:offset + MINUTE_POST_CHUNK]
        payload = json.dumps({"source": "iquant_index_daily", "rows": chunk},
                             ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(INDEX_DAILY_BRIDGE_URL, data=payload,
                          headers={"Content-Type": "application/json; charset=utf-8"})
        response = urlopen(request, timeout=30)
        try:
            if not 200 <= int(response.getcode()) < 300:
                raise RuntimeError("daily bridge returned HTTP %s" % response.getcode())
        finally:
            response.close()


def _normalize_symbol(value):
    text = str(value or "").strip().upper()
    match = re.match(r"^(SH|SZ|BJ)(\d{6})$", text)
    if match:
        return "%s.%s" % (match.group(2), match.group(1))
    if re.match(r"^\d{6}\.(SH|SZ|BJ)$", text):
        return text
    return None


def _datetime_from_value(value):
    if value is None:
        return None
    try:
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if isinstance(value, dt.datetime):
            return value.replace(tzinfo=None)
    except Exception:
        pass
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    try:
        if len(digits) == 8:
            return dt.datetime.strptime(digits, "%Y%m%d")
        if len(digits) >= 14:
            return dt.datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
        number = float(value)
        if number > 10000000000:
            number /= 1000.0
        if number > 1000000000:
            return dt.datetime.fromtimestamp(number)
    except Exception:
        pass
    return None


def _minute_rows_from_frames(frames):
    rows = []
    if not isinstance(frames, dict):
        return rows
    for raw_symbol, frame in frames.items():
        symbol = _normalize_symbol(raw_symbol)
        if not symbol or not hasattr(frame, "iterrows"):
            continue
        try:
            iterator = frame.iterrows()
            for index, row in iterator:
                bar_time = _datetime_from_value(index)
                if bar_time is None:
                    continue
                item = {"symbol": symbol,
                        "datetime": bar_time.strftime("%Y-%m-%d %H:%M:%S")}
                for field in ("open", "high", "low", "close", "volume", "amount"):
                    try:
                        value = row.get(field)
                    except Exception:
                        value = None
                    item[field] = _clean(value)
                rows.append(item)
        except Exception as exc:
            _log("minute frame parse failed %s: %s %s" %
                 (symbol, type(exc).__name__, str(exc)))
    return rows


def _daily_rows_from_frames(frames):
    rows = []
    if not isinstance(frames, dict):
        return rows
    for raw_symbol, frame in frames.items():
        symbol = _normalize_symbol(raw_symbol)
        if not symbol or not hasattr(frame, "iterrows"):
            continue
        try:
            iterator = frame.iterrows()
            for index, row in iterator:
                day = _datetime_from_value(index)
                if day is None:
                    day = _datetime_from_value(row.get("date"))
                if day is None:
                    continue
                item = {"symbol": symbol, "date": day.strftime("%Y-%m-%d")}
                for field in ("open", "high", "low", "close", "volume", "amount"):
                    try:
                        value = row.get(field)
                    except Exception:
                        value = None
                    item[field] = _clean(value)
                rows.append(item)
        except Exception as exc:
            _log("daily frame parse failed %s: %s %s" %
                 (symbol, type(exc).__name__, str(exc)))
    return rows


def _history_window_open(now):
    if now.weekday() >= 5:
        # No trading-session contention on weekends; run the recent-history job all day.
        return True
    hm = now.hour * 100 + now.minute
    return hm >= 1505 or hm <= 30


def _load_history_universe(ContextInfo):
    result = []
    # Sector names are escaped so this source remains compatible with the GBK
    # Python strategy editor used by iQuant.
    sector_names = [u"\u6caa\u6df1A\u80a1", u"\u5317\u8bc1A\u80a1"]
    for sector_name in sector_names:
        try:
            values = ContextInfo.get_stock_list_in_sector(sector_name) or []
        except Exception as exc:
            _log("load sector failed %s: %s %s" %
                 (sector_name.encode("utf-8"), type(exc).__name__, str(exc)))
            values = []
        for value in values:
            symbol = _normalize_symbol(value)
            if symbol and symbol not in result:
                result.append(symbol)
    if not result:
        try:
            values = ContextInfo.get_universe() or []
        except Exception:
            values = []
        for value in values:
            symbol = _normalize_symbol(value)
            if symbol and symbol not in result:
                result.append(symbol)
    if not result:
        result = list(CORE_SYMBOLS)
        _log("full-market stock list unavailable; history fallback symbols=%s" % len(result))
    return result


def _history_tick(ContextInfo):
    global _history_universe, _history_cursor, _history_window_key
    global _history_attempts, _history_complete_logged, _history_busy, _history_started
    if _history_busy or not _history_window_open(dt.datetime.now()):
        return
    _history_started = True
    today = dt.datetime.now().date()
    window_key = today.strftime("%Y%m%d")
    if _history_window_key != window_key:
        _history_universe = _load_history_universe(ContextInfo)
        _history_cursor = 0
        _history_attempts = {}
        _history_window_key = window_key
        _history_complete_logged = ""
        _log("minute history window %s: symbols=%s lookback=%s days" %
             (window_key, len(_history_universe), HISTORY_LOOKBACK_DAYS))
    if _history_cursor >= len(_history_universe):
        if _history_complete_logged != window_key:
            _history_complete_logged = window_key
            _log("minute history window %s complete: symbols=%s" %
                 (window_key, len(_history_universe)))
        return
    _history_busy = True
    try:
        start_time = (today - dt.timedelta(days=HISTORY_LOOKBACK_DAYS)).strftime("%Y%m%d")
        end_time = today.strftime("%Y%m%d")
        for _ in range(HISTORY_BATCHES_PER_RUN):
            if _history_cursor >= len(_history_universe):
                break
            batch_start = _history_cursor
            batch = _history_universe[batch_start:batch_start + HISTORY_BATCH_SIZE]
            try:
                _log("minute history batch start %s-%s/%s" %
                     (batch_start + 1, min(batch_start + len(batch), len(_history_universe)),
                      len(_history_universe)))
                frames = ContextInfo.get_market_data_ex(
                    fields=["open", "high", "low", "close", "volume", "amount"],
                    stock_code=batch, period="1m", start_time=start_time,
                    end_time=end_time, count=-1, dividend_type="none",
                    fill_data=False, subscribe=False,
                )
                rows = _minute_rows_from_frames(frames)
                if not rows:
                    _log("minute history local cache empty for batch %s-%s; retry with QMT subscription" %
                         (batch_start + 1, min(batch_start + len(batch), len(_history_universe))))
                    frames = ContextInfo.get_market_data_ex(
                        fields=["open", "high", "low", "close", "volume", "amount"],
                        stock_code=batch, period="1m", start_time=start_time,
                        end_time=end_time, count=-1, dividend_type="none",
                        fill_data=False, subscribe=True,
                    )
                    rows = _minute_rows_from_frames(frames)
                if rows:
                    _post_minute_rows(rows)
                    _history_attempts.pop(batch_start, None)
                    _log("minute history batch %s-%s/%s rows=%s" %
                         (batch_start + 1, min(batch_start + len(batch), len(_history_universe)),
                          len(_history_universe), len(rows)))
                    _history_cursor += len(batch)
                else:
                    attempts = _history_attempts.get(batch_start, 0) + 1
                    _history_attempts[batch_start] = attempts
                    if attempts >= 3:
                        _log("minute history batch skipped after %s empty attempts: %s-%s" %
                             (attempts, batch_start + 1, batch_start + len(batch)))
                        _history_cursor += len(batch)
                    else:
                        _log("minute history batch empty, retry %s/3: %s-%s" %
                             (attempts, batch_start + 1, batch_start + len(batch)))
                        break
            except Exception as exc:
                attempts = _history_attempts.get(batch_start, 0) + 1
                _history_attempts[batch_start] = attempts
                _log("minute history batch failed retry %s/3: %s-%s %s %s" %
                     (attempts, batch_start + 1, batch_start + len(batch),
                      type(exc).__name__, str(exc)))
                if attempts >= 3:
                    _history_cursor += len(batch)
                else:
                    break
    finally:
        _history_busy = False


def _schedule_history(ContextInfo):
    try:
        now = dt.datetime.now()
        first = now + dt.timedelta(seconds=5)
        if now.weekday() < 5 and now.hour < 15:
            first = now.replace(hour=15, minute=5, second=0, microsecond=0)
        ContextInfo.schedule_run(
            _history_timer, first, -1,
            dt.timedelta(seconds=HISTORY_INTERVAL_SECONDS),
            "tickflow_qmt_minute_history",
        )
        _log("minute history scheduler started at %s" % first.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as exc:
        _log("minute history scheduler unavailable: %s %s" %
             (type(exc).__name__, str(exc)))


def _history_timer(ContextInfo):
    _history_tick(ContextInfo)


def _load_index_universe(ContextInfo):
    """Read the app's exact index universe; use QMT sectors only as fallback."""
    result = []
    try:
        request = Request(INDEX_UNIVERSE_URL)
        response = urlopen(request, timeout=8)
        try:
            payload = json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
        values = payload.get("symbols", []) if isinstance(payload, dict) else []
        for value in values:
            symbol = _normalize_symbol(value)
            if symbol and symbol not in result:
                result.append(symbol)
        if result:
            _log("index universe loaded from app: %s symbols" % len(result))
            return result
    except Exception as exc:
        _log("index universe app request failed: %s %s" %
             (type(exc).__name__, str(exc)))
    for sector_name in [u"\u6caa\u6df1\u6307\u6570", u"\u4e2d\u8bc1\u6307\u6570", u"\u6307\u6570"]:
        try:
            values = ContextInfo.get_stock_list_in_sector(sector_name) or []
        except Exception:
            values = []
        for value in values:
            symbol = _normalize_symbol(value)
            if symbol and symbol not in result:
                result.append(symbol)
    if not result:
        result = list(CORE_SYMBOLS[:4])
        _log("index universe fallback symbols=%s" % len(result))
    return result


def _index_tick(ContextInfo):
    global _index_universe, _index_cursor, _index_window_key
    global _index_attempts, _index_complete_logged, _index_busy
    if _index_busy or not _history_window_open(dt.datetime.now()):
        return
    today = dt.datetime.now().date()
    window_key = today.strftime("%Y%m%d")
    if _index_window_key != window_key:
        _index_universe = _load_index_universe(ContextInfo)
        _index_cursor = 0
        _index_attempts = {}
        _index_window_key = window_key
        _index_complete_logged = ""
        _log("index daily window %s: symbols=%s lookback=%s days" %
             (window_key, len(_index_universe), INDEX_LOOKBACK_DAYS))
    if _index_cursor >= len(_index_universe):
        if _index_complete_logged != window_key:
            _index_complete_logged = window_key
            _log("index daily window %s complete: symbols=%s" %
                 (window_key, len(_index_universe)))
        return
    _index_busy = True
    try:
        start_time = (today - dt.timedelta(days=INDEX_LOOKBACK_DAYS)).strftime("%Y%m%d")
        end_time = today.strftime("%Y%m%d")
        for _ in range(INDEX_BATCHES_PER_RUN):
            if _index_cursor >= len(_index_universe):
                break
            batch_start = _index_cursor
            batch = _index_universe[batch_start:batch_start + INDEX_BATCH_SIZE]
            try:
                _log("index daily batch start %s-%s/%s" %
                     (batch_start + 1, min(batch_start + len(batch), len(_index_universe)),
                      len(_index_universe)))
                frames = ContextInfo.get_market_data_ex(
                    fields=["open", "high", "low", "close", "volume", "amount"],
                    stock_code=batch, period="1d", start_time=start_time,
                    end_time=end_time, count=-1, dividend_type="none",
                    fill_data=False, subscribe=False,
                )
                rows = _daily_rows_from_frames(frames)
                if not rows:
                    _log("index daily local cache empty for batch %s-%s; retry with QMT subscription" %
                         (batch_start + 1, min(batch_start + len(batch), len(_index_universe))))
                    frames = ContextInfo.get_market_data_ex(
                        fields=["open", "high", "low", "close", "volume", "amount"],
                        stock_code=batch, period="1d", start_time=start_time,
                        end_time=end_time, count=-1, dividend_type="none",
                        fill_data=False, subscribe=True,
                    )
                    rows = _daily_rows_from_frames(frames)
                if rows:
                    _post_daily_rows(rows)
                    _index_attempts.pop(batch_start, None)
                    _log("index daily batch %s-%s/%s rows=%s" %
                         (batch_start + 1, min(batch_start + len(batch), len(_index_universe)),
                          len(_index_universe), len(rows)))
                    _index_cursor += len(batch)
                else:
                    attempts = _index_attempts.get(batch_start, 0) + 1
                    _index_attempts[batch_start] = attempts
                    if attempts >= 3:
                        _log("index daily batch skipped after %s empty attempts: %s-%s" %
                             (attempts, batch_start + 1, batch_start + len(batch)))
                        _index_cursor += len(batch)
                    else:
                        _log("index daily batch empty, retry %s/3: %s-%s" %
                             (attempts, batch_start + 1, batch_start + len(batch)))
                        break
            except Exception as exc:
                attempts = _index_attempts.get(batch_start, 0) + 1
                _index_attempts[batch_start] = attempts
                _log("index daily batch failed retry %s/3: %s-%s %s %s" %
                     (attempts, batch_start + 1, batch_start + len(batch),
                      type(exc).__name__, str(exc)))
                if attempts >= 3:
                    _index_cursor += len(batch)
                else:
                    break
    finally:
        _index_busy = False


def _schedule_index(ContextInfo):
    try:
        first = dt.datetime.now() + dt.timedelta(seconds=15)
        now = dt.datetime.now()
        if now.weekday() < 5 and now.hour < 15:
            first = now.replace(hour=15, minute=6, second=0, microsecond=0)
        ContextInfo.schedule_run(
            _index_timer, first, -1,
            dt.timedelta(seconds=INDEX_INTERVAL_SECONDS),
            "tickflow_qmt_index_daily",
        )
        _log("index daily scheduler started at %s" % first.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as exc:
        _log("index daily scheduler unavailable: %s %s" %
             (type(exc).__name__, str(exc)))


def _index_timer(ContextInfo):
    _index_tick(ContextInfo)


def _sender(rows):
    global _sender_running
    failed = []
    try:
        for offset in range(0, len(rows), CHUNK_SIZE):
            chunk = rows[offset:offset + CHUNK_SIZE]
            try:
                if not _post_rows(chunk):
                    failed.extend(chunk)
            except Exception as exc:
                _log("bridge send failed: %s %s" %
                     (type(exc).__name__, str(exc)))
                failed.extend(chunk)
    finally:
        with _lock:
            for row in failed:
                symbol = row.get("symbol")
                if symbol:
                    _pending[symbol] = row
            _sender_running = False


def _flush(force=False):
    global _sender_running, _last_send_started
    now = time.time()
    with _lock:
        if _sender_running or not _pending:
            return
        if not force and now - _last_send_started < FLUSH_SECONDS:
            return
        rows = list(_pending.values())
        _pending.clear()
        _sender_running = True
        _last_send_started = now
    thread = threading.Thread(target=_sender, args=(rows,))
    thread.daemon = True
    thread.start()


def on_whole_quote(data):
    rows = _rows_from_callback(data)
    if rows:
        with _lock:
            for row in rows:
                _pending[row["symbol"]] = row
        _flush()


def _push_snapshot(ContextInfo, force=False):
    data = ContextInfo.get_full_tick(CORE_SYMBOLS)
    rows = _rows_from_callback(data)
    if rows:
        with _lock:
            for row in rows:
                _pending[row["symbol"]] = row
        _flush(force=force)
    return len(rows)


def init(ContextInfo):
    _log("collector init started")
    ContextInfo.set_universe(CORE_SYMBOLS)
    for market in MARKETS:
        try:
            sub_id = ContextInfo.subscribe_whole_quote([market], on_whole_quote)
            if sub_id and sub_id > 0:
                _subscription_ids.append(sub_id)
                _log("subscribed %s id=%s" % (market, sub_id))
            else:
                _log("subscribe rejected %s id=%s" % (market, sub_id))
        except Exception as exc:
            _log("subscribe failed %s: %s %s" %
                 (market, type(exc).__name__, str(exc)))
    try:
        count = _push_snapshot(ContextInfo, force=True)
        _log("initial snapshot rows=%s subscriptions=%s" %
             (count, len(_subscription_ids)))
    except Exception as exc:
        _log("initial snapshot failed: %s %s" %
             (type(exc).__name__, str(exc)))
    _schedule_history(ContextInfo)
    _schedule_index(ContextInfo)


def after_init(ContextInfo):
    try:
        count = _push_snapshot(ContextInfo, force=True)
        _log("after_init snapshot rows=%s" % count)
    except Exception as exc:
        _log("after_init snapshot failed: %s %s" %
             (type(exc).__name__, str(exc)))


def handlebar(ContextInfo):
    global _last_runtime_error
    # Whole-market callbacks do the main work. This keeps core indices alive if
    # the vendor's whole-quote stream only emits securities whose price changed.
    try:
        _push_snapshot(ContextInfo)
    except Exception as exc:
        now = time.time()
        if now - _last_runtime_error >= 60:
            _last_runtime_error = now
            _log("handlebar snapshot failed: %s %s" %
                 (type(exc).__name__, str(exc)))
    _flush()
