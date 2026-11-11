from __future__ import annotations

import logging
import math
import os
import threading
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pytdx.hq import TdxHq_API

logger = logging.getLogger("tdx_api")

app = FastAPI(title="Local TDX HTTP API", version="0.1.0")

CN_TZ = ZoneInfo("Asia/Shanghai")
_minute_node_lock = threading.Lock()
_minute_node_retry: dict[str, float] = {}

DEFAULT_HOSTS = (
    "180.153.18.170:7709",
    "119.147.212.81:7709",
    "119.97.185.5:7709",
    "101.227.73.20:7709",
    "180.153.39.51:7709",
    "14.215.128.18:7709",
)

CATEGORY_MAP = {
    "5m": 0,
    "minute5": 0,
    "15m": 1,
    "minute15": 1,
    "30m": 2,
    "minute30": 2,
    "60m": 3,
    "hour": 3,
    "1h": 3,
    "day": 4,
    "daily": 4,
    "week": 5,
    "weekly": 5,
    "month": 6,
    "monthly": 6,
    "1m": 7,
    "minute1": 7,
    "min1": 7,
}


def _parse_hosts() -> list[tuple[str, int]]:
    raw = os.getenv("TDX_HOSTS", ",".join(DEFAULT_HOSTS))
    hosts: list[tuple[str, int]] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        if ":" in text:
            host, port_text = text.rsplit(":", 1)
            try:
                port = int(port_text)
            except ValueError:
                port = 7709
        else:
            host, port = text, 7709
        hosts.append((host.strip(), port))
    return hosts or [(host.split(":", 1)[0], 7709) for host in DEFAULT_HOSTS]


def _timeout() -> float:
    try:
        return max(1.0, float(os.getenv("TDX_CONNECT_TIMEOUT", "3")))
    except ValueError:
        return 3.0


def _usable_minute_sample(rows) -> bool:
    if not rows:
        return False
    for row in rows:
        try:
            prices = [float(row[k]) for k in ("open", "high", "low", "close")]
            if not all(math.isfinite(p) and p > 0 for p in prices):
                return False
            if prices[1] < max(prices) or prices[2] > min(prices):
                return False
            datetime.fromisoformat(str(row["datetime"]))
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _connect(require_minute: bool = False) -> tuple[TdxHq_API, str]:
    errors: list[str] = []
    deadline = time.monotonic() + 4.0 if require_minute else None
    for host, port in _parse_hosts():
        label = f"{host}:{port}"
        if require_minute:
            with _minute_node_lock:
                deferred = _minute_node_retry.get(label, 0) > time.monotonic()
            if deferred:
                continue
            if time.monotonic() >= deadline:
                break
        api = TdxHq_API()
        try:
            timeout = min(_timeout(), max(.1, deadline - time.monotonic())) if require_minute else _timeout()
            if api.connect(host, port, time_out=timeout):
                if require_minute:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError()
                    client = getattr(api, "client", None)
                    if client is not None:
                        client.settimeout(remaining)
                if not require_minute or _usable_minute_sample(api.get_security_bars(7, 0, "000001", 0, 5)):
                    return api, label
                errors.append(f"{label}: no valid minute sample")
            else:
                errors.append(f"{label}: connect returned false")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {type(exc).__name__}")
        try:
            api.disconnect()
        except Exception:  # noqa: BLE001
            pass
        if require_minute:
            with _minute_node_lock:
                _minute_node_retry[label] = time.monotonic() + 60
    if require_minute:
        raise RuntimeError("TDX minute nodes unavailable; retry after cooldown")
    raise RuntimeError("; ".join(errors) if errors else "no TDX hosts configured")


def _compact_code(code: str) -> str:
    return "".join(ch for ch in str(code or "") if ch.isdigit())[-6:].zfill(6)


def _market(code: str) -> int:
    raw = str(code or "").strip().upper()
    compact = _compact_code(raw)
    if raw.startswith("SH") or raw.endswith(".SH"):
        return 1
    if raw.startswith("SZ") or raw.endswith(".SZ"):
        return 0
    if raw.startswith("BJ") or raw.endswith(".BJ") or compact.startswith(("4", "8", "92")):
        return 2
    if compact.startswith(("5", "6", "9")):
        return 1
    return 0


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _price(value: Any) -> float | None:
    number = _num(value)
    return round(number * 1000, 4) if number is not None else None


def _amount(value: Any) -> float | None:
    number = _num(value)
    return round(number * 1000, 4) if number is not None else None


def _volume_hands(value: Any) -> float | None:
    number = _num(value)
    return round(number / 100, 4) if number is not None else None


def _format_dt(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value or "")


def _quote_payload(raw: dict[str, Any], compact: str, market: int, host: str) -> dict[str, Any]:
    now = datetime.now(CN_TZ).replace(tzinfo=None)
    price = raw.get("price")
    prev_close = raw.get("last_close")
    return {
        "Code": compact,
        "Market": market,
        "Name": raw.get("stock_name") or raw.get("name") or "",
        "TotalHand": raw.get("vol"),
        "Amount": _amount(raw.get("amount")),
        "ServerTime": now.isoformat(timespec="seconds"),
        "SourceHost": host,
        "K": {
            "Open": _price(raw.get("open")),
            "High": _price(raw.get("high")),
            "Low": _price(raw.get("low")),
            "Close": _price(price),
            "Last": _price(prev_close),
        },
        "Bid1": _price(raw.get("bid1")),
        "Ask1": _price(raw.get("ask1")),
        "BidVol1": raw.get("bid_vol1"),
        "AskVol1": raw.get("ask_vol1"),
    }


def _bar_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Time": _format_dt(row.get("datetime")),
        "Open": _price(row.get("open")),
        "High": _price(row.get("high")),
        "Low": _price(row.get("low")),
        "Close": _price(row.get("close")),
        "Volume": _volume_hands(row.get("vol")),
        "Amount": _amount(row.get("amount")),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        api, host = _connect()
        api.disconnect()
        return {
            "code": 0,
            "status": "ok",
            "service": "tdx-api",
            "tdx_connected": True,
            "host": host,
            "message": "TDX quote server connected",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "code": 0,
            "status": "degraded",
            "service": "tdx-api",
            "tdx_connected": False,
            "host": None,
            "message": str(exc),
        }


@app.get("/api/quote")
def quote(code: str = Query(..., min_length=1)) -> dict[str, Any]:
    compact = _compact_code(code)
    market = _market(code)
    try:
        api, host = _connect()
        rows = api.get_security_quotes([(market, compact)]) or []
        api.disconnect()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"TDX quote failed: {exc}") from exc

    items = [_quote_payload(row, compact, market, host) for row in rows if isinstance(row, dict)]
    return {"code": 0, "message": "success", "data": items}


@app.get("/api/quotes")
def quotes(codes: str = Query(..., min_length=1)) -> dict[str, Any]:
    code_list = [
        item.strip()
        for item in codes.replace("，", ",").split(",")
        if item.strip()
    ]
    if not code_list:
        return {"code": 0, "message": "success", "data": []}

    try:
        api, host = _connect()
        items: list[dict[str, Any]] = []
        # pytdx 批量 quote 一次传太多可能不稳定，按小块请求更稳。
        for start in range(0, len(code_list), 80):
            chunk = code_list[start:start + 80]
            query = [(_market(code), _compact_code(code)) for code in chunk]
            rows = api.get_security_quotes(query) or []
            indexed = {(row.get("market"), row.get("code")): row
                       for row in rows if isinstance(row, dict)}
            for original in chunk:
                compact = _compact_code(original)
                market = _market(original)
                row = indexed.get((market, compact))
                if row is None:
                    continue
                items.append(_quote_payload(row, compact, market, host))
        api.disconnect()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"TDX quotes failed: {exc}") from exc

    return {"code": 0, "message": "success", "data": items}


@app.get("/api/kline")
def kline(
    code: str = Query(..., min_length=1),
    type: str = Query("day"),
    limit: int = Query(800, ge=1, le=24000),
) -> dict[str, Any]:
    compact = _compact_code(code)
    market = _market(code)
    category = CATEGORY_MAP.get(str(type).strip().lower())
    if category is None:
        raise HTTPException(status_code=400, detail=f"unsupported kline type: {type}")

    try:
        api, host = _connect(require_minute=category in {0, 1, 2, 3, 7})
        rows: list[dict[str, Any]] = []
        offset = 0
        remaining = limit
        while remaining > 0:
            batch_size = min(800, remaining)
            batch = api.get_security_bars(category, market, compact, offset, batch_size) or []
            if not batch:
                break
            rows.extend(row for row in batch if isinstance(row, dict))
            if len(batch) < batch_size:
                break
            offset += batch_size
            remaining -= batch_size
        api.disconnect()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"TDX kline failed: {exc}") from exc

    payload = [_bar_payload(row) for row in rows]
    payload.sort(key=lambda item: str(item.get("Time") or ""))
    return {
        "code": 0,
        "message": "success",
        "data": {
            "Code": compact,
            "Market": market,
            "Type": type,
            "Limit": limit,
            "SourceHost": host,
            "List": payload,
        },
    }


@app.get("/api/search")
def search(q: str = "") -> dict[str, Any]:
    return {"code": 0, "message": "success", "data": []}


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    uvicorn.run(
        "app:app",
        host=os.getenv("TDX_API_HOST", "0.0.0.0"),
        port=int(os.getenv("TDX_API_PORT", "8080")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
