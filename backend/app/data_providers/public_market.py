"""Bounded public L1 backups. Prices/yuan, stock volume/shares, rates/decimals.

Undocumented endpoints are best-effort, never a replacement for historical
minute, auction, financial or L2 feeds. Provider time is kept, not fetch time.
"""
from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter
from datetime import datetime
from datetime import time as dt_time

import requests

from app.config import settings
from app.market_time import CN_TZ, cn_now
from app.services.security_identity import exchange_for_code

SOURCES = ("tencent", "sina", "eastmoney", "xueqiu")
LABELS = {"tencent": "腾讯", "sina": "新浪", "eastmoney": "东方财富", "xueqiu": "雪球"}
_SYMBOL = re.compile(r"^(\d{6})\.(SH|SZ|BJ)$")


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def wire_symbol(symbol):
    match = _SYMBOL.fullmatch(str(symbol).upper())
    return match[2].lower() + match[1] if match else None


def qualified_symbol(symbol):
    raw = str(symbol or "").strip().upper()
    match = _SYMBOL.fullmatch(raw)
    if match:
        # Legacy pools inferred SH from every 9-prefix, including BSE 92xxxx.
        return f"{match[1]}.BJ" if match[1].startswith("92") else raw
    prefixed = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", raw)
    if prefixed:
        return f"{prefixed[2]}.{prefixed[1]}"
    if not re.fullmatch(r"\d{6}", raw):
        return raw
    market = exchange_for_code(raw, "stock")
    return f"{raw}.{market}" if market else raw


def timestamp_ms(value):
    try:
        if isinstance(value, datetime):
            dt = value.replace(tzinfo=CN_TZ) if value.tzinfo is None else value
        elif isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(value / 1000 if value > 1e11 else value, CN_TZ)
        else:
            text = str(value)
            dt = (datetime.strptime(text, "%Y%m%d%H%M%S") if len(text) == 14 and text.isdigit()
                  else datetime.fromisoformat(text))
            dt = dt.replace(tzinfo=CN_TZ) if dt.tzinfo is None else dt
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def fresh_timestamp(value, now=None):
    now = now or cn_now()
    stamp = timestamp_ms(value)
    if stamp is None:
        return False
    dt = datetime.fromtimestamp(stamp / 1000, CN_TZ)
    if dt.date() != now.date() or (dt - now).total_seconds() > 30:
        return False
    # Lunch/close accept only the latest session's terminal snapshot, not any
    # same-day row. Pre-open yesterday's close cannot become today's OHLC.
    t = now.time()
    reference = now
    if dt_time(11, 30) < t < dt_time(13):
        reference = now.replace(hour=11, minute=30, second=0, microsecond=0)
    elif t > dt_time(15):
        reference = now.replace(hour=15, minute=0, second=0, microsecond=0)
    return (reference - dt).total_seconds() <= settings.public_market_fallback_max_age_s


def valid_quote(row, now=None):
    if not wire_symbol(row.get("symbol")) or not fresh_timestamp(row.get("timestamp"), now):
        return False
    prices = [number(row.get(k)) for k in ("open", "high", "low", "last_price", "prev_close")]
    if any(v is None or v <= 0 for v in prices):
        return False
    op, high, low, last, _ = prices
    if not low <= min(op, last) <= max(op, last) <= high:
        return False
    # Index volume conventions differ. Do not fabricate equity volume units.
    if row.get("asset_type") == "index":
        return True
    volume, amount = number(row.get("volume")), number(row.get("amount"))
    if volume is None or amount is None or volume < 0 or amount < 0:
        return False
    return amount == 0 if volume == 0 else low * 0.98 <= amount / volume <= high * 1.02


def valid_depth(row, now=None):
    if not fresh_timestamp(row.get("timestamp"), now) or row.get("asset_type") == "index":
        return False
    for side in ("bid", "ask"):
        prices = [number(v) for v in row.get(f"{side}_prices") or []]
        volumes = [number(v) for v in row.get(f"{side}_volumes") or []]
        if len(prices) != 5 or len(volumes) != 5:
            return False
        if any(number(p) is None or number(v) is None or p < 0 or v < 0
               or (p == 0) != (v == 0) for p, v in zip(prices, volumes, strict=True)):
            return False
        active = [p for p in prices if p > 0]
        if active != sorted(active, reverse=side == "bid"):
            return False
    bid, ask = number(row["bid_prices"][0]), number(row["ask_prices"][0])
    return (bid > 0 or ask > 0) and (not bid or not ask or bid <= ask)


def _finish(row, symbol, source, asset_type):
    row.update(symbol=symbol, source=source, asset_type=asset_type)
    row["timestamp"] = timestamp_ms(row.get("timestamp"))
    row["change_amount"] = row["last_price"] - row["prev_close"]
    row["change_pct"] = row["change_amount"] / row["prev_close"] if row["prev_close"] else None
    row["amplitude"] = (row["high"] - row["low"]) / row["prev_close"] if row["prev_close"] else None
    if asset_type == "index":
        row["volume"] = row["amount"] = None
        for key in ("bid_prices", "bid_volumes", "ask_prices", "ask_volumes"):
            row.pop(key, None)
    return row


def parse_tencent(text, assets):
    result = []
    for code, payload in re.findall(r'v_([a-z]+\d+)="([^"]*)"', text):
        symbol = code[2:] + "." + code[:2].upper()
        if symbol not in assets:
            continue
        fields = payload.split("~")
        try:
            if len(fields) < 58 or fields[2] != code[2:]:
                continue
            amount_fields = fields[35].split("/")
            amount = number(amount_fields[2]) if len(amount_fields) >= 3 else None
            if amount is None:
                exact = number(fields[57])
                amount = exact * 10000 if exact is not None else None
            row = dict(name=fields[1], last_price=float(fields[3]), prev_close=float(fields[4]),
                       open=float(fields[5]), high=float(fields[33]), low=float(fields[34]),
                       volume=float(fields[6]) * 100, amount=amount, timestamp=fields[30],
                       turnover_rate=number(fields[38]) / 100 if number(fields[38]) is not None else None)
            for side, offset in (("bid", 9), ("ask", 19)):
                row[f"{side}_prices"] = [float(fields[offset + i * 2]) for i in range(5)]
                row[f"{side}_volumes"] = [int(float(fields[offset + i * 2 + 1]) * 100) for i in range(5)]
            result.append(_finish(row, symbol, "tencent", assets[symbol]))
        except (ValueError, TypeError, IndexError):
            continue
    return result


def parse_sina(text, assets):
    result = []
    for code, payload in re.findall(r'hq_str_([a-z]+\d+)="([^"]*)"', text):
        symbol = code[2:] + "." + code[:2].upper()
        if symbol not in assets:
            continue
        fields = payload.split(",")
        try:
            if len(fields) < 32:
                continue
            row = dict(name=fields[0], open=float(fields[1]), prev_close=float(fields[2]),
                       last_price=float(fields[3]), high=float(fields[4]), low=float(fields[5]),
                       volume=float(fields[8]), amount=float(fields[9]),
                       timestamp=fields[30] + "T" + fields[31])
            for side, offset in (("bid", 10), ("ask", 20)):
                row[f"{side}_prices"] = [float(fields[offset + i * 2 + 1]) for i in range(5)]
                row[f"{side}_volumes"] = [int(float(fields[offset + i * 2])) for i in range(5)]
            result.append(_finish(row, symbol, "sina", assets[symbol]))
        except (ValueError, TypeError, IndexError):
            continue
    return result


def parse_eastmoney(payload, symbol, asset_type):
    data = payload.get("data") or {}
    if str(data.get("f57")) != symbol.split(".")[0]:
        return []
    try:
        row = dict(name=data.get("f58"), last_price=float(data["f43"]), prev_close=float(data["f60"]),
                   open=float(data["f46"]), high=float(data["f44"]), low=float(data["f45"]),
                   volume=float(data["f47"]) * 100, amount=float(data["f48"]), timestamp=data.get("f86"))
        for side, offset in (("bid", 19), ("ask", 39)):
            row[f"{side}_prices"] = [number(data.get(f"f{offset - i * 2}")) for i in range(5)]
            row[f"{side}_volumes"] = [int(number(data[f"f{offset - i * 2 + 1}"]) * 100)
                                       if number(data.get(f"f{offset - i * 2 + 1}")) is not None else None
                                       for i in range(5)]
        # Net-only, not gross buy/sell moneyflow. Never write these into that table.
        flow = {key: number(data.get(field)) for key, field in (
            ("main_net", "f137"), ("super_large_net", "f140"), ("large_net", "f143"))}
        if all(v is not None for v in flow.values()) and abs(
            flow["main_net"] - flow["super_large_net"] - flow["large_net"]
        ) <= max(1, abs(flow["main_net"]) * 1e-5):
            row["fund_flow_snapshot"] = {**flow, "unit": "CNY", "source": "eastmoney", "kind": "vendor_net_only"}
        return [_finish(row, symbol, "eastmoney", asset_type)]
    except (TypeError, ValueError, KeyError):
        return []


class PublicMarketBackup:
    """Shared cache + circuit breakers; bounded work even on a total outage."""

    def __init__(self):
        self._lock = threading.RLock()
        self._fetch_lock = threading.Lock()
        self._cache = {}
        self._empty_until = {}
        self._health = {}
        self._last = {}

    def available(self, source):
        with self._lock:
            return time.monotonic() >= self._health.get(source, {}).get("retry_at", 0)

    def record(self, source, ok, reason=""):
        with self._lock:
            old = self._health.get(source, {})
            failures = 0 if ok else min(6, old.get("failures", 0) + 1)
            self._health[source] = dict(failures=failures, error=reason[:120],
                                        retry_at=time.monotonic() + min(900, 30 * 2 ** (failures - 1)) if failures else 0,
                                        checked_at=cn_now().isoformat())

    def status(self):
        with self._lock:
            return {
                "enabled": settings.public_market_fallback_enabled,
                "chain": [dict(source=s, label=LABELS[s]) for s in SOURCES],
                "max_symbols": settings.public_market_fallback_max_symbols,
                "health": {s: {k: v for k, v in h.items() if k != "retry_at"} | {
                    "retry_in_s": round(max(0, h["retry_at"] - time.monotonic()), 1)
                } for s, h in self._health.items()},
                "last": dict(self._last),
            }

    @staticmethod
    def _request(source, assets):
        if source == "xueqiu":
            # Reuse the optional provider so its identity checks, token/cookie
            # handling and unit conversion stay in one place.
            from app.data_providers import custom as custom_sources

            if not custom_sources.provider_has_dataset("xueqiu", "realtime"):
                return []
            provider = custom_sources.get_provider("xueqiu")
            frame = provider.get_realtime(symbols=list(assets))
            return frame.to_dicts() if frame is not None and not frame.is_empty() else []
        with requests.Session() as session:
            session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"})
            if source in ("tencent", "sina"):
                codes = ",".join(wire_symbol(s) for s in assets)
                url = "https://qt.gtimg.cn/q=" if source == "tencent" else "https://hq.sinajs.cn/list="
                response = session.get(url + codes, timeout=(3, 5))
                response.raise_for_status()
                parser = parse_tencent if source == "tencent" else parse_sina
                return parser(response.content.decode("gb18030", errors="replace"), assets)
            symbol = next(iter(assets))
            code, market = symbol.split(".")
            # Default fields include depth; requesting a fields mask can silently remove it.
            response = session.get("https://push2.eastmoney.com/api/qt/stock/get", params={
                "secid": f"{1 if market == 'SH' else 0}.{code}", "fltt": 2, "invt": 2,
            }, timeout=(3, 5))
            response.raise_for_status()
            return parse_eastmoney(response.json(), symbol, assets[symbol])

    def fetch(self, assets, *, capability="realtime", now=None):
        if not settings.public_market_fallback_enabled:
            return []
        now = now or cn_now()
        assets = {s: a for s, a in assets.items() if wire_symbol(s) and
                  (capability != "depth5" or a != "index")}
        requested = len(assets)
        assets = dict(list(assets.items())[:settings.public_market_fallback_max_symbols])
        validator = valid_depth if capability == "depth5" else valid_quote
        result = {}
        # The same lock provides single-flight across quote and depth polling.
        with self._fetch_lock:
            deadline = time.monotonic() + 20
            for source in SOURCES:
                missing = {s: a for s, a in assets.items() if s not in result}
                if not missing:
                    break
                # Xueqiu's endpoint is a quote snapshot only. It must never
                # be advertised as a five-level order-book fallback.
                if source == "xueqiu" and capability == "depth5":
                    continue
                for symbol in list(missing):
                    cached = self._cache.get((source, symbol))
                    if cached and time.monotonic() - cached[0] < 15 and validator(cached[1], now):
                        result[symbol] = dict(cached[1])
                        missing.pop(symbol)
                missing = {s: a for s, a in missing.items()
                           if time.monotonic() >= self._empty_until.get((source, capability, s), 0)}
                if not missing or not self.available(source) or time.monotonic() >= deadline:
                    continue
                # Tencent/Sina are batched. Eastmoney is per-symbol: at most 3
                # calls per pass, not N requests during a full-market outage.
                items = list(missing.items())
                groups = [dict(items[i:i + 50]) for i in range(0, len(items), 50)] if source != "eastmoney" else [dict([i]) for i in items[:3]]
                good = 0
                invalid = False
                for group in groups:
                    if time.monotonic() >= deadline:
                        break
                    try:
                        rows = self._request(source, group)
                    except Exception as exc:
                        # Never log URLs/headers or primary API keys.
                        self.record(source, False, type(exc).__name__)
                        invalid = True
                        break
                    for row in rows:
                        if row.get("symbol") in group and validator(row, now):
                            result[row["symbol"]] = row
                            self._cache[(source, row["symbol"])] = (time.monotonic(), row)
                            good += 1
                    for symbol in group:
                        if symbol not in result:
                            self._empty_until[(source, capability, symbol)] = time.monotonic() + 30
                if not invalid:
                    # A suspended/unsupported symbol is not a provider-wide outage.
                    self.record(source, True, "" if good else "empty_or_stale_or_invalid_for_batch")
            self._last[capability] = dict(requested=requested, returned=len(result),
                                         missing=requested - len(result), updated_at=now.isoformat(),
                                         sources=dict(Counter(r["source"] for r in result.values())))
            if len(self._cache) > 1200:
                self._cache = {k: v for k, v in self._cache.items() if time.monotonic() - v[0] < 60}
            if len(self._empty_until) > 1200:
                self._empty_until = {k: v for k, v in self._empty_until.items() if v > time.monotonic()}
        return list(result.values())


public_market_backup = PublicMarketBackup()
