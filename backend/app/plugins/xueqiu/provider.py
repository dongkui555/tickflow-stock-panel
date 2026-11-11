"""雪球行情适配器。

雪球的轻量实时快照在未登录时也可能可用；历史日K通常需要雪球会话。
本模块只承诺 realtime/daily，明确不把雪球网页接口当作分钟、资金流或 L2 源。
"""
from __future__ import annotations

import contextlib
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

import polars as pl
import requests

from app.data_providers.base import AssetType
from app.services.security_identity import exchange_for_code

logger = logging.getLogger(__name__)

CN_TZ = ZoneInfo("Asia/Shanghai")
_DATASETS = ("realtime", "daily")
_DAILY_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
_WIRE_RE = re.compile(r"^(SH|SZ|BJ)(\d{6})$")


class XueqiuError(RuntimeError):
    """雪球返回不可用、需要认证或结构变化。"""


@dataclass
class _Config:
    name: str = "xueqiu"
    display_name: str = "雪球（实时快照/日K备源）"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _canonical_symbol(value: object) -> str | None:
    """Only accept an explicit or structurally unambiguous A-share identity."""
    raw = str(value or "").strip().upper()
    code_match = re.fullmatch(r"(\d{6})(?:\.(SH|SZ|BJ))?", raw)
    prefixed = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", raw)
    if prefixed:
        exchange, code = prefixed.group(1), prefixed.group(2)
    elif code_match:
        code, exchange = code_match.group(1), code_match.group(2)
    else:
        return None
    inferred = exchange_for_code(code, "stock")
    if exchange and inferred and exchange != inferred:
        return None
    exchange = exchange or inferred
    return f"{code}.{exchange}" if exchange else None


def _wire_symbol(symbol: object) -> str | None:
    canonical = _canonical_symbol(symbol)
    if not canonical:
        return None
    code, exchange = canonical.split(".")
    return f"{exchange}{code}"


def _canonical_from_wire(value: object) -> str | None:
    match = _WIRE_RE.fullmatch(str(value or "").strip().upper())
    return _canonical_symbol(f"{match.group(2)}.{match.group(1)}") if match else None


def _timestamp_ms(value: object) -> int | None:
    number = _number(value)
    if number is None:
        return None
    if number < 10_000_000_000:
        number *= 1000
    return int(number)


def _date_from_ms(value: object) -> date | None:
    stamp = _timestamp_ms(value)
    if stamp is None:
        return None
    try:
        return datetime.fromtimestamp(stamp / 1000, CN_TZ).date()
    except (OverflowError, OSError, ValueError):
        return None


def _parse_quote_rows(payload: Any, requested: dict[str, str]) -> list[dict[str, Any]]:
    """Map realtime/quotec.json while enforcing requested exchange identity."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return []
    wire_requested = {
        _wire_symbol(symbol): (symbol, asset_type)
        for symbol, asset_type in requested.items()
        if asset_type == "stock" and _wire_symbol(symbol)
    }
    rows: list[dict[str, Any]] = []
    for raw in payload["data"]:
        if not isinstance(raw, dict):
            continue
        canonical = _canonical_from_wire(raw.get("symbol"))
        wire = _wire_symbol(canonical) if canonical else None
        if not canonical or wire not in wire_requested or canonical != wire_requested[wire][0]:
            continue
        values = {
            "last_price": _number(raw.get("current")),
            "prev_close": _number(raw.get("last_close")),
            "open": _number(raw.get("open")),
            "high": _number(raw.get("high")),
            "low": _number(raw.get("low")),
            "volume": _number(raw.get("volume")),
            "amount": _number(raw.get("amount")),
            "timestamp": _timestamp_ms(raw.get("timestamp")),
        }
        required = ("last_price", "prev_close", "open", "high", "low", "timestamp")
        if any(values[name] is None for name in required):
            continue
        if any(values[name] is not None and values[name] < 0 for name in ("volume", "amount")):
            continue
        prev = values["prev_close"]
        last = values["last_price"]
        change_amount = _number(raw.get("chg"))
        if change_amount is None and prev not in (None, 0) and last is not None:
            change_amount = last - prev
        rows.append({
            "symbol": canonical,
            "name": raw.get("name"),
            **values,
            "change_amount": change_amount,
            "change_pct": (_number(raw.get("percent")) or 0.0) / 100.0,
            "turnover_rate": (_number(raw.get("turnover_rate")) or 0.0) / 100.0,
            "amplitude": (_number(raw.get("amplitude")) or 0.0) / 100.0,
            "source": "xueqiu",
            "asset_type": "stock",
        })
    return rows


def _parse_kline_payload(
    payload: Any,
    symbol: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """Parse Xueqiu's column + item K-line response into raw daily OHLCV."""
    canonical = _canonical_symbol(symbol)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not canonical or not isinstance(data, dict):
        return pl.DataFrame()
    returned = _canonical_from_wire(data.get("symbol"))
    if returned != canonical or not isinstance(data.get("column"), list) or not isinstance(data.get("item"), list):
        return pl.DataFrame()
    columns = [str(column) for column in data["column"]]
    required = {"timestamp", "open", "high", "low", "close", "volume", "amount"}
    if not required.issubset(columns):
        return pl.DataFrame()
    index = {name: columns.index(name) for name in required}
    start_d = start.date() if start else None
    end_d = end.date() if end else None
    rows: list[dict[str, Any]] = []
    for item in data["item"]:
        if not isinstance(item, (list, tuple)) or len(item) < len(columns):
            continue
        stamp = _timestamp_ms(item[index["timestamp"]])
        day = _date_from_ms(stamp)
        if day is None or (start_d and day < start_d) or (end_d and day > end_d):
            continue
        values = {name: _number(item[index[name]]) for name in ("open", "high", "low", "close", "volume", "amount")}
        if any(value is None for value in values.values()):
            continue
        if min(values[name] for name in ("open", "high", "low", "close")) <= 0:
            continue
        if min(values["volume"], values["amount"]) < 0:
            continue
        if values["high"] < max(values["open"], values["close"], values["low"]):
            continue
        if values["low"] > min(values["open"], values["close"], values["high"]):
            continue
        rows.append({"symbol": canonical, "date": day, **values})
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).select(_DAILY_COLS).unique(subset=["symbol", "date"], keep="last").sort("date")


def _get_secret(field: str, env_name: str) -> str:
    with contextlib.suppress(Exception):
        from app import secrets_store

        return secrets_store.get_env_backed_secret(field, env_name)
    return os.getenv(env_name, "").strip()


def get_token() -> str:
    return _get_secret("xueqiu_api_key", "XUEQIU_TOKEN")


def get_cookie() -> str:
    return os.getenv("XUEQIU_COOKIE", "").strip()


def history_auth_configured() -> bool:
    """Whether the optional credential needed by the historical K-line endpoint exists."""
    return bool(get_token() or get_cookie())


def availability() -> tuple[bool, str]:
    # realtime/quotec.json was verified to work anonymously; daily K-line may need a cookie.
    return True, "匿名实时快照可用；历史日K按需配置 XUEQIU_TOKEN 或 XUEQIU_COOKIE"


def probe_api_key(api_key: str) -> tuple[bool, str]:
    try:
        provider = XueqiuProvider(token=api_key)
        try:
            end = datetime.now(CN_TZ).replace(tzinfo=None)
            frame = provider.get_daily(["600000.SH"], end - timedelta(days=14), end)
        finally:
            provider.close()
        return (True, "ok") if not frame.is_empty() else (False, "Token 未能返回日K，可能不是 xq_a_token 或已过期")
    except XueqiuError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"雪球接口探测失败: {type(exc).__name__}"


@dataclass
class XueqiuProvider:
    name: str = "xueqiu"
    builtin: bool = True
    token: str | None = None

    def __post_init__(self) -> None:
        self.config = _Config()
        self.realtime_scope = "symbols"
        self.timeout = max(2.0, float(os.getenv("XUEQIU_TIMEOUT_SECONDS", "8")))
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": os.getenv("XUEQIU_USER_AGENT", "Mozilla/5.0"),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://xueqiu.com/",
        })
        cookie = get_cookie()
        token = str(self.token if self.token is not None else get_token()).strip()
        if cookie:
            self._session.headers["Cookie"] = cookie
        elif token:
            self._session.headers["Cookie"] = f"xq_a_token={token}"

    def close(self) -> None:
        self._session.close()

    def history_auth_configured(self) -> bool:
        return bool((self.token or "").strip() or get_token() or get_cookie())

    def _request_json(self, path: str, params: dict[str, Any]) -> Any:
        try:
            response = self._session.get(
                f"https://stock.xueqiu.com{path}",
                params=params,
                timeout=(3, self.timeout),
            )
            if response.status_code in {401, 403}:
                raise XueqiuError("雪球接口需要有效会话凭据")
            if response.status_code == 400:
                raise XueqiuError("雪球接口拒绝请求；历史日K请配置 XUEQIU_TOKEN 或 XUEQIU_COOKIE")
            response.raise_for_status()
            payload = response.json()
        except XueqiuError:
            raise
        except requests.RequestException as exc:
            raise XueqiuError(f"雪球网络请求失败: {type(exc).__name__}") from exc
        except ValueError as exc:
            raise XueqiuError("雪球返回格式不是 JSON") from exc
        if isinstance(payload, dict) and payload.get("error_code") not in (None, 0, "0"):
            raise XueqiuError(str(payload.get("error_description") or "雪球接口返回错误"))
        return payload

    def get_realtime(self, universes=None, symbols=None) -> pl.DataFrame:
        requested = {
            canonical: "stock"
            for symbol in symbols or []
            if (canonical := _canonical_symbol(symbol))
        }
        if not requested:
            return pl.DataFrame()
        rows: list[dict[str, Any]] = []
        symbols_wire = list(requested)
        for offset in range(0, len(symbols_wire), 50):
            group = symbols_wire[offset:offset + 50]
            try:
                payload = self._request_json(
                    "/v5/stock/realtime/quotec.json",
                    {"symbol": ",".join(_wire_symbol(symbol) for symbol in group)},
                )
                rows.extend(_parse_quote_rows(payload, {symbol: requested[symbol] for symbol in group}))
            except XueqiuError as exc:
                logger.warning("雪球实时快照批次失败: %s", exc)
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if asset_type != "stock":
            return pl.DataFrame()
        end = end_time or datetime.now()
        start = start_time or (end - timedelta(days=365))
        begin = int(end.replace(tzinfo=CN_TZ).timestamp() * 1000)
        count = max(60, min(5000, int((end.date() - start.date()).days * 1.7 + 30)))
        frames: list[pl.DataFrame] = []
        normalized = [_canonical_symbol(symbol) for symbol in symbols]
        normalized = [symbol for symbol in normalized if symbol]
        for position, symbol in enumerate(normalized, 1):
            try:
                payload = self._request_json(
                    "/v5/stock/chart/kline.json",
                    {
                        "symbol": _wire_symbol(symbol),
                        "begin": begin,
                        "period": "day",
                        "type": "before",
                        "count": -count,
                        "indicator": "kline",
                    },
                )
                frame = _parse_kline_payload(payload, symbol, start=start, end=end)
                if not frame.is_empty():
                    frames.append(frame)
            except XueqiuError as exc:
                logger.warning("雪球日K %s 请求失败: %s", symbol, exc)
            if on_chunk_done:
                on_chunk_done(position, len(normalized))
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed").select(_DAILY_COLS).unique(
            subset=["symbol", "date"], keep="last"
        ).sort(["symbol", "date"])

    def get_minute(self, symbols, start_time, end_time, asset_type="stock", freq="1m", on_chunk_done=None):
        return pl.DataFrame()

    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        test_symbols = symbols or ["600000.SH"]
        if dataset == "realtime":
            frame = self.get_realtime(symbols=test_symbols)
        elif dataset == "daily":
            end = datetime.now(CN_TZ).replace(tzinfo=None)
            frame = self.get_daily(test_symbols, end - timedelta(days=14), end)
        else:
            raise ValueError(f"雪球不支持数据集: {dataset}")
        return {
            "provider": self.name,
            "dataset": dataset,
            "rows": frame.height,
            "columns": frame.columns,
            "preview": frame.head(5).to_dicts() if not frame.is_empty() else [],
        }
