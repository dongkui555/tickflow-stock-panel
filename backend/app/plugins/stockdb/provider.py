"""Read-only local StockDB adapter: raw yuan/share input, yuan/lot output."""
from __future__ import annotations

import ipaddress
import json
import math
import re
import time as clock
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from urllib.parse import urlsplit

import polars as pl
import requests

from app.config import settings
from app.market_time import cn_now

SCHEMA = {"symbol": pl.String, "datetime": pl.Datetime("us"),
          **dict.fromkeys(("open", "high", "low", "close", "volume", "amount"), pl.Float64)}
TABLE = "分钟k"


def local_url(value: str) -> str:
    url = urlsplit(value.strip())
    host = url.hostname or ""
    allowed = host in {"localhost", "host.docker.internal", "stockdb-api"}
    try:
        address = ipaddress.ip_address(host)
        allowed |= address.is_loopback or any(address in ipaddress.ip_network(net) for net in (
            "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7"))
    except ValueError:
        pass
    if (not allowed or url.scheme not in {"http", "https"} or url.username or url.password
            or url.query or url.fragment or url.path not in {"", "/"}):
        raise ValueError("STOCKDB_BASE_URL must be a local/private HTTP origin, not a public demo")
    _ = url.port
    return value.strip().rstrip("/")


def availability() -> tuple[bool, str]:
    if not settings.stockdb_enabled:
        return False, "未启用 STOCKDB_ENABLED"
    try:
        local_url(settings.stockdb_base_url)
    except ValueError:
        return False, "STOCKDB_BASE_URL 必须是本地或内网地址, 禁止公网体验源"
    return True, "本地地址已配置; 实际补缺仍须返回合格分钟数据"


@dataclass
class _Config:
    name: str = "stockdb"
    display_name: str = "本地 StockDB (历史分钟补缺)"
    datasets: dict = field(default_factory=lambda: {"minute": None})
    path: None = None
    builtin: bool = True


class StockDBError(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def symbol_identity(value: str) -> tuple[str, str]:
    match = re.fullmatch(r"(\d{6})(?:\.(SH|SZ|BJ))?", str(value).strip().upper())
    if not match:
        raise StockDBError("source_symbol_conflict")
    code, suffix = match.groups()
    market = ("SH" if code.startswith(("60", "68")) else "SZ" if code.startswith(("00", "30"))
              else "BJ" if code.startswith(("43", "83", "87", "88", "92")) else None)
    if not market or suffix not in {None, market}:
        raise StockDBError("source_symbol_conflict")
    return code, f"{code}.{market}"


def _parse_payload(payload, symbol: str, day, hour: int | None, limit: int) -> list[dict]:
    code, normalized = symbol_identity(symbol)
    if isinstance(payload, dict) and payload.get("status") == "not_found":
        return []
    if not isinstance(payload, list) or len(payload) > limit:
        raise StockDBError("source_schema_invalid")
    records = []
    for item in payload:
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], dict):
            raise StockDBError("source_schema_invalid")
        key, raw = item
        stamp = str(raw.get("date", ""))
        if raw.get("code") != code or key != f"{TABLE}:{code}:{stamp}":
            raise StockDBError("source_symbol_conflict")
        if not re.fullmatch(r"\d{14}", stamp):
            raise StockDBError("source_schema_invalid")
        try:
            moment = datetime.strptime(stamp, "%Y%m%d%H%M%S")
            values = {name: float(raw[name]) for name in ("open", "high", "low", "close", "volume", "amount")}
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise StockDBError("source_schema_invalid") from exc
        if moment.date() != day or (hour is not None and moment.hour != hour) or moment.second:
            raise StockDBError("source_time_conflict")
        if (not all(math.isfinite(v) for v in values.values())
                or min(values[n] for n in ("open", "high", "low", "close")) <= 0
                or min(values["volume"], values["amount"]) < 0
                or values["high"] < max(values["open"], values["close"], values["low"])
                or values["low"] > min(values["open"], values["close"], values["high"])):
            raise StockDBError("source_invalid")
        if moment.time() == time(13) and values["volume"] == values["amount"] == 0:
            continue
        beijing_after_hours = normalized.endswith(".BJ") and time(15, 1) <= moment.time() <= time(15, 30)
        if not (time(9, 30) <= moment.time() <= time(11, 30)
                or time(13, 1) <= moment.time() <= time(15)
                or beijing_after_hours):
            raise StockDBError("source_time_conflict")
        # StockDB volume is shares; the existing verified minute store uses lots.
        values["volume"] /= 100.0
        records.append({"symbol": normalized, "datetime": moment, **values})
    return records


def parse_hour(payload, symbol: str, day, hour: int) -> list[dict]:
    return _parse_payload(payload, symbol, day, hour, 61)


def parse_day(payload, symbol: str, day) -> list[dict]:
    return _parse_payload(payload, symbol, day, None, 272)


class StockDBProvider:
    name = "stockdb"
    builtin = True

    def __init__(self):
        self.config = _Config()
        self.base_url = local_url(settings.stockdb_base_url)
        self.timeout = settings.stockdb_timeout_seconds
        self._session = requests.Session()
        self._session.trust_env = False

    def close(self):
        self._session.close()

    def _request(self, query):
        started = clock.monotonic()
        # Small hourly prefixes also avoid the demo server's hanging day responses.
        with self._session.get(self.base_url + "/", params={"cmd": "get", "t": query, "json": 1},
                               timeout=(self.timeout, self.timeout), stream=True, allow_redirects=False) as response:
            if response.status_code in {401, 403}:
                raise StockDBError("permission_denied")
            if response.status_code == 429:
                raise StockDBError("rate_limited")
            if response.status_code != 200:
                raise StockDBError("upstream_unavailable")
            body = bytearray()
            for chunk in response.iter_content(16384):
                body.extend(chunk)
                if len(body) > 1024 * 1024:
                    raise StockDBError("source_schema_invalid")
                if clock.monotonic() - started > self.timeout * 3:
                    raise StockDBError("request_timeout")
        try:
            return json.loads(body)
        except (ValueError, UnicodeError) as exc:
            raise StockDBError("source_schema_invalid") from exc

    def get_minute_diagnostics(self, symbols, start_time, end_time, asset_type="stock", freq="1m", on_chunk_done=None):
        from app.services.kline_sync import CN_TZ
        from app.services.minute_failover import _intraday_reason

        now = cn_now().astimezone(CN_TZ).replace(tzinfo=None)
        def wall(value):
            return value.astimezone(CN_TZ).replace(tzinfo=None) if value.tzinfo else value
        if asset_type != "stock" or freq != "1m" or start_time is None or end_time is None:
            return pl.DataFrame(schema=SCHEMA), [{"reason": "unsupported_minute_scope", "retryable": False}]
        start, end = wall(start_time), min(wall(end_time), now)
        if start > end or (end.date() - start.date()).days > settings.stockdb_max_stock_days * 4:
            return pl.DataFrame(schema=SCHEMA), [{"reason": "request_scope_too_large", "retryable": False}]
        days = [start.date() + timedelta(days=i) for i in range((end.date() - start.date()).days + 1)]
        days = [day for day in days if day.weekday() < 5 and (day < now.date() or now.time() >= time(15, 35))
                and end >= datetime.combine(day, time(15))]
        if not days:
            return pl.DataFrame(schema=SCHEMA), [{"reason": "closed_history_only", "retryable": False}]
        if len(days) * len(set(symbols)) > settings.stockdb_max_stock_days:
            return pl.DataFrame(schema=SCHEMA), [{"reason": "request_scope_too_large", "retryable": False}]
        frames, issues = [], []
        for index, (symbol, day) in enumerate(((s, d) for s in dict.fromkeys(symbols) for d in days), 1):
            stop = False
            try:
                code, _ = symbol_identity(symbol)
                payload = self._request(f"{TABLE}:{code}:{day:%Y%m%d}*")
                records = parse_day(payload, symbol, day)
                frame = pl.DataFrame(records, schema=SCHEMA).sort("datetime")
                reason = _intraday_reason(frame, day, datetime.combine(day, time(9, 31)), datetime.combine(day, time(15)))
                if reason == "accepted":
                    frames.append(frame)
            except requests.Timeout:
                reason, stop = "request_timeout", True
            except requests.RequestException:
                reason, stop = "transport_error", True
            except StockDBError as exc:
                reason = exc.reason
                stop = reason in {"permission_denied", "upstream_unavailable", "rate_limited", "request_timeout"}
            issues.append({"source": self.name, "symbol": symbol, "date": str(day), "reason": reason,
                           "retryable": reason not in {"accepted", "source_symbol_conflict"}})
            if on_chunk_done:
                on_chunk_done(index, len(days) * len(set(symbols)))
            if stop:
                issues[-1]["batch_aborted"] = True
                break
        result = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)
        return result.filter(pl.col("datetime").is_between(start, end)), issues

    def get_minute(self, symbols, start_time, end_time, asset_type="stock", freq="1m", on_chunk_done=None):
        return self.get_minute_diagnostics(symbols, start_time, end_time, asset_type, freq, on_chunk_done)[0]
