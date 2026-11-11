"""Tushare 数据源 provider。"""
from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable
from zoneinfo import ZoneInfo

import httpx

from app.services.security_identity import exchange_for_code, explicit_exchange

logger = logging.getLogger(__name__)

_CN_TZ = ZoneInfo("Asia/Shanghai")


class TushareMinuteSourceError(RuntimeError):
    """A failed request is not evidence that a trading date has no data."""

    def __init__(self, message, *, reason="source_error", retryable=False):
        super().__init__(message)
        self.reason = reason
        self.retryable = retryable


class TushareMinuteRateLimitError(TushareMinuteSourceError):
    """The minute endpoint exhausted bounded rate-limit retries."""

    def __init__(self, message):
        super().__init__(message, reason="rate_limited", retryable=True)


class TushareMinutePendingError(TushareMinuteSourceError):
    """The gateway accepted a request but has not produced its dataset yet."""

    def __init__(self, retry_after_seconds):
        super().__init__("Tushare minute data is still being prepared by the gateway",
                         reason="source_pending", retryable=True)
        self.retry_after_seconds = max(1, math.ceil(retry_after_seconds))


def _pending_retry_after(value: str) -> int:
    try:
        delay = float(value)
    except (ValueError, TypeError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            delay = 30.0
    return math.ceil(delay) if math.isfinite(delay) and delay > 0 else 30


def minute_error_details(exc: Exception) -> dict:
    """Only return stable categories, never exception text or credentials."""
    if isinstance(exc, TushareMinutePendingError):
        return {"reason": exc.reason, "retryable": True, "http_status": 202,
                "retry_after_seconds": exc.retry_after_seconds}
    if isinstance(exc, TushareMinuteSourceError):
        result = {"reason": exc.reason, "retryable": exc.retryable}
        if getattr(exc, "retry_after_seconds", None):
            result["retry_after_seconds"] = exc.retry_after_seconds
        return result
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        reason = {429: "rate_limited", 401: "permission_denied", 403: "permission_denied",
                  404: "endpoint_not_found"}.get(status, "upstream_unavailable" if status >= 500 else "request_rejected")
        result = {"reason": reason, "retryable": status == 429 or status >= 500, "http_status": status}
        if result["retryable"]:
            result["retry_after_seconds"] = _pending_retry_after(exc.response.headers.get("Retry-After", ""))
        return result
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or type(exc).__name__ in {"Timeout", "ReadTimeout", "ConnectTimeout"}:
        return {"reason": "request_timeout", "retryable": True}
    if isinstance(exc, httpx.TransportError) or type(exc).__name__ == "ConnectionError":
        return {"reason": "transport_error", "retryable": True}
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return {"reason": "source_schema_invalid", "retryable": False}
    return {"reason": "source_error_unclassified", "retryable": False}

_DATASETS = ("daily", "adj_factor", "financial", "auction", "moneyflow", "news", "minute")
_STOCK_BASIC_EXCHANGES = ("SSE", "SZSE", "BSE")


def _dedupe_stock_basic_rows(rows: list[dict], *, include_status: bool = False) -> list[dict]:
    """Merge exchange-partitioned stock_basic responses without losing status."""
    result: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        code = str(row.get("code") or "").strip()
        exchange = str(row.get("exchange") or "").strip().upper()
        status = str((row.get("ext") or {}).get("list_status") or "").strip().upper()
        key = (code, exchange, status if include_status else "")
        if not code or not exchange:
            continue
        current = result.get(key)
        if current is None:
            result[key] = row
            continue
        current_ext = current.setdefault("ext", {})
        for name, value in (row.get("ext") or {}).items():
            if value not in (None, "", "nan"):
                current_ext[name] = value
        for name in ("name", "region", "type"):
            if not current.get(name) and row.get(name):
                current[name] = row[name]
    return list(result.values())
_DAILY_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
_ADJ_COLS = ["symbol", "trade_date", "ex_factor"]
_AUCTION_COLS = [
    "symbol", "date", "market", "open", "high", "low", "close",
    "volume", "amount", "vwap",
]
_MONEYFLOW_COLS = [
    "symbol", "date",
    "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount",
    "buy_md_vol", "buy_md_amount", "sell_md_vol", "sell_md_amount",
    "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount",
    "buy_elg_vol", "buy_elg_amount", "sell_elg_vol", "sell_elg_amount",
    "net_mf_vol", "net_mf_amount",
]
_NEWS_COLS = [
    "id", "date", "published_at", "market", "market_scope", "source",
    "title", "summary", "content", "channels", "symbols", "url",
    "language", "sentiment", "raw",
]
_MINUTE_COLS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]
_FINANCIAL_METHODS = {
    "metrics": "fina_indicator",
    "income": "income",
    "balance_sheet": "balancesheet",
    "cash_flow": "cashflow",
    "shares": "share_float",
}

_FINANCIAL_MAP: dict[str, dict[str, str]] = {
    "metrics": {
        "eps": "eps_basic",
        "dt_eps": "eps_diluted",
        "bps": "bps",
        "roe": "roe",
        "roe_waa": "roe_diluted",
        "roa": "roa",
        "grossprofit_margin": "gross_margin",
        "netprofit_margin": "net_margin",
        "debt_to_assets": "debt_to_asset_ratio",
        "tr_yoy": "revenue_yoy",
        "netprofit_yoy": "net_income_yoy",
        "ocf_to_revenue": "operating_cash_to_revenue",
        "ocfps": "ocfps",
        "inv_turn": "inventory_turnover",
    },
    "income": {
        "total_revenue": "revenue",
        "revenue": "revenue",
        "oper_cost": "operating_cost",
        "operate_profit": "operating_profit",
        "sell_exp": "selling_expense",
        "admin_exp": "admin_expense",
        "rd_exp": "rd_expense",
        "fin_exp": "financial_expense",
        "non_oper_income": "non_operating_income",
        "non_oper_exp": "non_operating_expense",
        "total_profit": "total_profit",
        "income_tax": "income_tax",
        "n_income": "net_income",
        "n_income_attr_p": "net_income_attributable",
        "n_income_deducted": "net_income_deducted",
        "basic_eps": "basic_eps",
        "diluted_eps": "diluted_eps",
    },
    "balance_sheet": {
        "total_assets": "total_assets",
        "total_cur_assets": "total_current_assets",
        "total_nca": "total_non_current_assets",
        "money_cap": "cash_and_equivalents",
        "accounts_receiv": "accounts_receivable",
        "inventories": "inventory",
        "fix_assets": "fixed_assets",
        "intan_assets": "intangible_assets",
        "goodwill": "goodwill",
        "total_liab": "total_liabilities",
        "total_cur_liab": "total_current_liabilities",
        "total_ncl": "total_non_current_liabilities",
        "st_borr": "short_term_borrowing",
        "lt_borr": "long_term_borrowing",
        "acct_payable": "accounts_payable",
        "total_hldr_eqy_inc_min_int": "total_equity",
        "total_hldr_eqy_exc_min_int": "equity_attributable",
        "undistr_porfit": "retained_earnings",
        "minority_int": "minority_interest",
    },
    "cash_flow": {
        "n_cashflow_act": "net_operating_cash_flow",
        "n_cashflow_inv_act": "net_investing_cash_flow",
        "n_cash_flows_fnc_act": "net_financing_cash_flow",
        "n_cashflow_fnc_act": "net_financing_cash_flow",
        "c_pay_acq_const_fiolta": "capex",
        "c_pay_acq_intan_assets": "intangible_asset_capex",
        "n_incr_cash_cash_equ": "net_cash_change",
    },
    "shares": {
        "float_share": "float_shares",
        "total_share": "total_shares",
    },
}

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

AssetType = str


def _pd():
    import pandas as pd  # noqa: WPS433

    return pd


def _pl():
    import polars as pl  # noqa: WPS433

    return pl


@dataclass
class _Config:
    name: str = "tushare"
    display_name: str = "Tushare（日线/分钟/财务/资金）"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def _compact_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if "." in value:
        value = value.split(".", 1)[0]
    return value.zfill(6) if value.isdigit() else value


def _to_ts_code(symbol: str) -> str:
    code = _compact_symbol(symbol)
    market = exchange_for_code(code, "stock")
    declared = explicit_exchange(symbol)
    if not market or (declared and declared != market):
        raise ValueError(f"无法确认 Tushare 股票身份: {symbol}")
    return f"{code}.{market}"


def _minute_symbol_windows(symbol: str, start: datetime, end: datetime):
    # Issuer announcement 2025-028: same legal entity, code changed on 2025-02-17.
    # https://static.cninfo.com.cn/finalpage/2025-02-15/1222544408.PDF
    boundary = datetime(2025, 2, 17)
    if _compact_symbol(symbol) != "302132" or start >= boundary:
        return [(symbol, start, end)]
    windows = [("300114", start, min(end, boundary - timedelta(microseconds=1)))]
    if end >= boundary:
        windows.append((symbol, boundary, end))
    return windows


def _to_yyyymmdd(value: datetime | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    return value.strftime("%Y%m%d")


def _normalize_dates(df: "pd.DataFrame", column: str) -> "pd.DataFrame":
    pd = _pd()
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce").dt.date
    return df


def _as_polars(rows: list[dict]) -> "pl.DataFrame":
    pl = _pl()
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    if "date" in df.columns and df.schema["date"] != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date, strict=False))
    if "trade_date" in df.columns and df.schema["trade_date"] != pl.Date:
        df = df.with_columns(pl.col("trade_date").cast(pl.Date, strict=False))
    numeric_cols = {
        "open", "high", "low", "close", "volume", "amount", "vwap", "ex_factor",
        *[column for column in _MONEYFLOW_COLS if column.endswith("_vol") or column.endswith("_amount")],
    }
    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
    keep = list(dict.fromkeys(
        c
        for c in (*_DAILY_COLS, *_ADJ_COLS, *_AUCTION_COLS, *_MONEYFLOW_COLS)
        if c in df.columns
    ))
    return df.select(keep) if keep else pl.DataFrame()


def _minute_freq(freq: str) -> str:
    value = str(freq or "1m").strip().lower()
    digits = "".join(ch for ch in value if ch.isdigit()) or "1"
    return f"{digits}min"


def _to_beijing_wallclock(value: datetime) -> datetime:
    """Normalize minute query bounds to the same naive Beijing wallclock used in local minute bars."""
    if value.tzinfo is not None and value.utcoffset() is not None:
        return value.astimezone(_CN_TZ).replace(tzinfo=None)
    return value


def _normalize_minute_frame(frame: "pd.DataFrame", source_symbol: str | None = None) -> "pl.DataFrame":
    pd = _pd()
    pl = _pl()
    if frame is None or frame.empty:
        return pl.DataFrame()

    if "trade_time" in frame.columns:
        dt = pd.to_datetime(frame["trade_time"], format="mixed", errors="coerce")
    elif "datetime" in frame.columns:
        dt = pd.to_datetime(frame["datetime"], format="mixed", errors="coerce")
    elif "date" in frame.columns:
        dt = pd.to_datetime(frame["date"], format="mixed", errors="coerce")
    else:
        return pl.DataFrame()

    out = frame.copy()
    out["datetime"] = dt.dt.floor("min")
    out = out[out["datetime"].notna()]
    if out.empty:
        return pl.DataFrame()

    # 保留源实际提供的09:30记录；丢弃09:25及15:00后的盘后记录。
    # 不把竞价差额合成分钟线，也不把源时间戳平移到另一个分钟。
    minutes = out["datetime"].dt.strftime("%H:%M")
    regular = (
        ((minutes >= "09:30") & (minutes <= "11:30"))
        | ((minutes >= "13:01") & (minutes <= "15:00"))
    )
    out = out[regular]
    if out.empty:
        return pl.DataFrame()

    if "ts_code" in out.columns:
        out["symbol"] = out["ts_code"].astype(str).map(_compact_symbol)
    elif "symbol" not in out.columns and source_symbol:
        out["symbol"] = _compact_symbol(source_symbol)
    elif "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).map(_compact_symbol)
    else:
        return pl.DataFrame()

    rename_map = {
        "vol": "volume",
        "amt": "amount",
    }
    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        else:
            out[column] = None

    # stk_mins / rt_min_daily 成交量为股，目标分钟库为手；成交额仍为元。
    out["volume"] = out["volume"] / 100.0
    rows = out[_MINUTE_COLS].to_dict(orient="records")
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    if df.is_empty():
        return df
    return (
        df.with_columns([
            pl.col("symbol").cast(pl.Utf8),
            pl.col("datetime").cast(pl.Datetime("us"), strict=False),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in ("open", "high", "low", "close", "volume", "amount")],
        ])
        .drop_nulls(["symbol", "datetime", "open", "high", "low", "close"])
        .unique(subset=["symbol", "datetime"], keep="last")
        .sort(["symbol", "datetime"])
        .select(_MINUTE_COLS)
    )


def _parse_news_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = _pd().to_datetime(value, errors="coerce")
    except Exception:
        return None
    if _pd().isna(parsed):
        return None
    if hasattr(parsed, "to_pydatetime"):
        parsed = parsed.to_pydatetime()
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed if isinstance(parsed, datetime) else None


def _extract_cn_symbols(text: str) -> str:
    symbols = sorted(set(re.findall(r"(?<!\d)[03468]\d{5}(?!\d)", text or "")))
    return ",".join(symbols[:20])


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_rows(payload: Any) -> list[dict]:
    """兼容官方 Tushare / 代理网关 / 自定义聚合网关的响应格式。"""
    if payload is None:
        return []
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []

    candidates: list[Any] = []
    data = payload.get("data")
    if isinstance(data, dict):
        fields = data.get("fields") or payload.get("fields")
        items = data.get("items") or data.get("list") or payload.get("items") or payload.get("list")
        if isinstance(items, list) and items:
            first = items[0]
            if isinstance(first, dict):
                return [dict(row) for row in items]
            if isinstance(first, list) and fields:
                return [dict(zip(fields, row)) for row in items]
        candidates.extend([data.get("rows"), data.get("result")])
    elif isinstance(data, list):
        candidates.append(data)
    for key in ("rows", "items", "list", "result"):
        candidates.append(payload.get(key))
    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            first = candidate[0]
            if isinstance(first, dict):
                return [dict(row) for row in candidate]
            if isinstance(first, list):
                fields = payload.get("fields")
                if isinstance(fields, list) and fields:
                    return [dict(zip(fields, row)) for row in candidate]
    return []


class TushareAuctionSourceError(RuntimeError):
    """Safe auction failure metadata; never retain upstream URLs or credentials."""

    def __init__(self, reason: str, *, source_failures: list[dict] | None = None) -> None:
        self.reason = reason
        self.source_failures = source_failures or []
        super().__init__(f"Tushare auction: {reason}")


def _auction_source_error(exc: Exception) -> TushareAuctionSourceError:
    if isinstance(exc, TushareAuctionSourceError):
        return exc
    if isinstance(exc, TushareAPIFailoverError):
        return TushareAuctionSourceError('all_sources_failed', source_failures=exc.source_failures)
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        reason = "network_timeout"
    elif isinstance(exc, httpx.TransportError):
        reason = "network_unavailable"
    elif isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        reason = "permission_denied" if code in (401, 403) else "rate_limited" if code == 429 else "gateway_rejected"
    elif isinstance(exc, (ValueError, KeyError)):
        reason = "invalid_response"
    else:
        reason = "request_failed"
    return TushareAuctionSourceError(reason)


class _PromaxTushareAPI:
    """轻量 Tushare 兼容网关客户端。"""

    supports_verified_history = True

    def __init__(self, base_url: str, api_key: str, verify: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.verify = verify
        self._client = httpx.Client(timeout=60.0, verify=verify)
        self._minute_rate_lock = Lock()
        self._next_minute_request = 0.0
        # Promax's documented minute quota is below 200 requests/minute.
        # Keep a small permanent gap instead of waiting for the first 429;
        # concurrent historical repairs must not burst past the gateway.
        self._minute_gap = 0.31
        self._pending_minute_requests: dict[tuple, float] = {}

    def close(self) -> None:
        self._client.close()

    def _request(self, api_name: str, **params):
        if not self.base_url or not self.api_key:
            raise RuntimeError("Tushare Promax 网关未配置完整")
        url = f"{self.base_url}/{api_name.lstrip('/')}"
        headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
        }
        clean_params = {k: v for k, v in params.items() if v is not None and v != ""}
        minute = api_name in {"stk_mins", "rt_min_daily", "a_share_mins"}
        pending_key = (api_name, tuple(sorted((key, str(value)) for key, value in clean_params.items())))
        if minute:
            with self._minute_rate_lock:
                now = time.monotonic()
                self._pending_minute_requests = {
                    key: due for key, due in self._pending_minute_requests.items() if due > now
                }
                due = self._pending_minute_requests.get(pending_key, 0.0)
                if due > now:
                    raise TushareMinutePendingError(due - now)
        for attempt in range(4 if minute else 1):
            if minute:
                with self._minute_rate_lock:
                    now = time.monotonic()
                    due = max(now, self._next_minute_request)
                    if due - now > 30:
                        raise TushareMinuteRateLimitError("Tushare minute gateway is still in Retry-After cooldown")
                    self._next_minute_request = due + self._minute_gap
                if due > now:
                    time.sleep(due - now)
            try:
                resp = self._client.get(url, params=clean_params, headers=headers)
            except httpx.TransportError as exc:
                message = str(exc).lower()
                if not self.verify or not any(token in message for token in ("ssl", "certificate", "tls")):
                    raise
                logger.warning("Promax TLS 校验失败，自动以 verify=False 重试: %s", self.base_url)
                resp = httpx.get(url, params=clean_params, headers=headers, timeout=self._client.timeout, verify=False)
            if minute and resp.status_code == 202:
                delay = _pending_retry_after(resp.headers.get("Retry-After", ""))
                with self._minute_rate_lock:
                    self._pending_minute_requests[pending_key] = time.monotonic() + delay
                raise TushareMinutePendingError(delay)
            if not minute or resp.status_code != 429:
                break
            raw_delay = resp.headers.get("Retry-After", "")
            try:
                delay = float(raw_delay)
            except ValueError:
                try:
                    delay = parsedate_to_datetime(raw_delay).timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    delay = 2.0 ** (attempt + 1)
            with self._minute_rate_lock:
                self._minute_gap = max(self._minute_gap, 2.0)
                self._next_minute_request = max(self._next_minute_request, time.monotonic() + max(0.1, delay))
            if delay > 30 or attempt == 3:
                break
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and (
            payload.get("success") is False or payload.get("code") not in {None, 0, "0", 200, "200"}
        ):
            if minute:
                message = str(payload.get("msg") or payload.get("message") or "").lower()
                denied = any(word in message for word in ("权限", "积分", "permission", "unauthorized", "forbidden"))
                limited = any(word in message for word in ("频率", "限流", "rate limit", "too many"))
                if limited:
                    raise TushareMinuteRateLimitError("Tushare minute gateway rate limited")
                raise TushareMinuteSourceError("Tushare minute gateway returned an error response",
                                              reason="permission_denied" if denied else "gateway_rejected")
            raise RuntimeError(f"Tushare gateway returned an error response for {api_name}")
        rows = _extract_rows(payload)
        return _pd().DataFrame(rows) if rows else _pd().DataFrame()

    def __getattr__(self, name: str):
        return functools.partial(self._request, name)


class DatahubTushareAPI(_PromaxTushareAPI):
    """Tushare-compatible client for the datahubco GET gateway."""

    def _request(self, api_name: str, **params):
        try:
            return super()._request(api_name, **params)
        except RuntimeError as exc:
            # Keep credentials and upstream payloads out of application logs.
            raise RuntimeError(f"Tushare datahub request failed for {api_name}") from exc


class TushareAPIFailoverError(RuntimeError):
    """Retain source categories without upstream exception text or credentials."""

    def __init__(self, source_failures: list[dict]) -> None:
        self.source_failures = source_failures
        super().__init__('Tushare request failed on all configured gateways')


class TushareAPIFailover:
    """Call a backup Tushare-compatible API only when the primary raises.

    An empty DataFrame is a valid upstream answer and therefore must not trigger
    failover.  This distinction prevents a backup with different publication
    timing from fabricating a result for a legitimately empty query.
    """

    def __init__(self, primary, backup, primary_name: str = "promax", backup_name: str = "datahub") -> None:
        self.primary = primary
        self.backup = backup
        self.primary_name = primary_name
        self.backup_name = backup_name
        self.last_source: str | None = None
        self.supports_verified_history = bool(getattr(primary, "supports_verified_history", False))

    def close(self) -> None:
        seen: set[int] = set()
        for client in (self.primary, self.backup):
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(**params):
            self.last_source = None
            try:
                result = getattr(self.primary, name)(**params)
                self.last_source = self.primary_name
                return result
            except Exception as primary_exc:  # noqa: BLE001
                logger.warning("Tushare %s failed on %s; trying %s", name, self.primary_name, self.backup_name)
                try:
                    result = getattr(self.backup, name)(**params)
                    self.last_source = self.backup_name
                    return result
                except Exception as backup_exc:  # noqa: BLE001
                    if name not in {'stk_auction_o', 'stk_auction_c'}:
                        raise RuntimeError(f"Tushare {name} failed on all configured gateways") from backup_exc
                    failures = [
                        {'source': self.primary_name, 'reason': _auction_source_error(primary_exc).reason},
                        {'source': self.backup_name, 'reason': _auction_source_error(backup_exc).reason},
                    ]
                    raise TushareAPIFailoverError(failures) from None

        return call


def build_datahub_client() -> DatahubTushareAPI:
    """Build the optional datahub client without exposing its credential."""
    from app.config import settings

    base_url = (
        os.getenv("TUSHARE_DATAHUB_URL", "").strip()
        or settings.tushare_datahub_url.strip()
    )
    api_key = (
        os.getenv("TUSHARE_DATAHUB_KEY", "").strip()
        or settings.tushare_datahub_key.strip()
        or os.getenv("TUSHARE_PROMAX_KEY", "").strip()
        or settings.tushare_promax_key.strip()
    )
    verify = (
        _env_bool("TUSHARE_DATAHUB_VERIFY_TLS", False)
        if os.getenv("TUSHARE_DATAHUB_VERIFY_TLS") is not None
        else bool(settings.tushare_datahub_verify_tls)
    )
    if not base_url or not api_key:
        raise RuntimeError("Tushare datahub gateway is not configured")
    return DatahubTushareAPI(base_url, api_key, verify=verify)


class TushareProvider:
    adj_factor_kind = "cumulative"

    name = "tushare"
    builtin = True

    def __init__(self) -> None:
        self.config = _Config()
        self._api = None
        self._auction_client = None
        self._datahub_client = None

    def close(self) -> None:
        if self._api is not None:
            try:
                self._api.close()
            except Exception:
                pass
        self._api = None
        if self._auction_client is not None:
            try:
                self._auction_client.close()
            except Exception:
                pass
        self._auction_client = None
        self._datahub_client = None

    def _connection_mode(self) -> str:
        from app.config import settings
        return str(os.getenv("TUSHARE_MODE", "").strip() or settings.tushare_mode or "official").strip().lower()

    def _tushare_token(self) -> str:
        from app.config import settings
        return os.getenv("TUSHARE_TOKEN", "").strip() or settings.tushare_token.strip()

    def _promax_client(self):
        if self._auction_client is not None:
            return self._auction_client
        from app.config import settings
        base_url = os.getenv("TUSHARE_PROMAX_URL", "").strip() or settings.tushare_promax_url.strip()
        api_key = os.getenv("TUSHARE_PROMAX_KEY", "").strip() or settings.tushare_promax_key.strip()
        verify = (
            _env_bool("TUSHARE_PROMAX_VERIFY_TLS", True)
            if os.getenv("TUSHARE_PROMAX_VERIFY_TLS") is not None
            else bool(settings.tushare_promax_verify_tls)
        )
        if not base_url or not api_key:
            raise RuntimeError("未配置 TUSHARE_PROMAX_URL / TUSHARE_PROMAX_KEY")
        self._auction_client = _PromaxTushareAPI(base_url, api_key, verify=verify)
        return self._auction_client

    def _ensure_api(self):
        if self._api is not None:
            return self._api
        mode = self._connection_mode()
        if mode == "promax":
            primary = self._promax_client()
            primary_name = "promax"
        else:
            token = self._tushare_token()
            if not token:
                raise RuntimeError("未设置 TUSHARE_TOKEN")
            import tushare as ts  # noqa: WPS433

            from app.config import settings

            ts.set_token(token)
            primary = ts.pro_api()
            http_url = os.getenv("TUSHARE_HTTP_URL", "").strip() or settings.tushare_http_url.strip()
            if http_url:
                try:
                    primary._DataApi__http_url = http_url
                except Exception as exc:  # noqa: BLE001
                    logger.debug("set tushare http_url failed: %s", exc)
            primary_name = "official"

        from app.config import settings
        enabled = (
            _env_bool("TUSHARE_DATAHUB_ENABLED", False)
            if os.getenv("TUSHARE_DATAHUB_ENABLED") is not None
            else bool(settings.tushare_datahub_enabled)
        )
        if enabled:
            try:
                self._datahub_client = build_datahub_client()
                self._api = TushareAPIFailover(primary, self._datahub_client, primary_name, "datahub")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tushare datahub backup unavailable: %s", type(exc).__name__)
                self._api = primary
        else:
            self._api = primary
        return self._api

    def get_instruments(self, asset_type: str = "stock") -> list[dict]:
        if asset_type != "stock":
            return []
        rows: list[dict] = []
        for exchange in _STOCK_BASIC_EXCHANGES:
            rows.extend(self._normalise_stock_basic(
                self._stock_basic_status("L", exchange=exchange), list_status="L",
            ))
        if not rows:
            rows = self._normalise_stock_basic(self._stock_basic_status("L"), list_status="L")
        return _dedupe_stock_basic_rows(rows, include_status=True)

    def get_trade_calendar(self, start: str, end: str) -> list[dict]:
        """Return explicit open/closed exchange calendar rows.

        Unlike ``_trade_dates`` this method does not fall back to weekdays:
        callers use it as historical evidence, so an unavailable endpoint
        must raise and leave the previous local calendar untouched.
        """
        api = self._ensure_api()
        frame = api.trade_cal(
            exchange="", start_date=start, end_date=end,
            fields="cal_date,is_open",
        )
        if frame is None or getattr(frame, "empty", True):
            return []
        return [
            {
                "trade_date": str(row.get("cal_date") or ""),
                "is_open": int(row.get("is_open")) if row.get("is_open") is not None else None,
            }
            for row in frame.to_dict(orient="records")
            if row.get("cal_date") is not None
        ]

    def get_historical_suspensions(self, start: str, end: str) -> list[dict]:
        """Return historical ``suspend_d`` rows with identity and interval.

        A missing/unauthorized endpoint is intentionally not converted to an
        empty result.  The caller must distinguish source failure from a
        genuine no-suspension response before updating the local snapshot.
        """
        api = self._ensure_api()
        method = getattr(api, "suspend_d", None)
        if not callable(method):
            raise TushareMinuteSourceError(
                "Tushare suspend_d endpoint is unavailable",
                reason="endpoint_not_found", retryable=False,
            )
        frame = method(
            start_date=start, end_date=end,
            fields="ts_code,suspend_date,resume_date,suspend_reason,suspend_type,suspend_timing",
        )
        if frame is None or getattr(frame, "empty", True):
            return []
        rows = []
        for row in frame.to_dict(orient="records"):
            symbol = str(row.get("ts_code") or row.get("symbol") or "").strip()
            suspend_date = row.get("suspend_date") or row.get("trade_date")
            if not symbol or suspend_date is None:
                continue
            rows.append({
                "symbol": symbol,
                "suspend_date": str(suspend_date),
                "resume_date": str(row.get("resume_date") or ""),
                "reason": str(row.get("suspend_reason") or "historical_suspension"),
                "suspend_type": str(row.get("suspend_type") or ""),
                "suspend_timing": str(row.get("suspend_timing") or ""),
            })
        return rows

    def _stock_basic_status(self, list_status: str, *, exchange: str | None = None):
        """Read one Tushare stock_basic lifecycle snapshot.

        The Promax gateway may reject an explicit ``fields`` argument, while
        the official/Datahub clients accept it.  Retry once without fields so
        a gateway schema quirk does not erase the lifecycle snapshot.
        """
        api = self._ensure_api()
        params = {"list_status": list_status}
        if exchange:
            params["exchange"] = exchange
        fields = (
            "ts_code,symbol,name,area,industry,market,list_date,delist_date,"
            "list_status,exchange"
        )
        if self._connection_mode() != "promax":
            params["fields"] = fields
        try:
            frame = api.stock_basic(**params)
        except Exception as exc:  # noqa: BLE001
            if "fields" not in params:
                logger.warning("tushare stock_basic(%s,%s) 失败: %s", list_status, exchange or "all", exc)
                return None
            try:
                frame = api.stock_basic(**params)
            except Exception as retry_exc:  # noqa: BLE001
                logger.warning("tushare stock_basic(%s,%s) 重试失败: %s", list_status, exchange or "all", retry_exc)
                return None
        return frame

    def _normalise_stock_basic(self, frame, list_status: str = "L") -> list[dict]:
        if frame is None or getattr(frame, "empty", True):
            return []
        rows: list[dict] = []
        for _, row in frame.iterrows():
            ts_code = str(row.get("ts_code") or "").strip()
            if not ts_code:
                continue
            code = _compact_symbol(ts_code)
            rows.append({
                "symbol": code,
                "name": row.get("name") or code,
                "code": str(row.get("symbol") or code),
                "exchange": ts_code.split(".")[-1] if "." in ts_code else "",
                "region": row.get("area"),
                "type": "stock",
                "ext": {
                    "listing_date": row.get("list_date"),
                    "delist_date": row.get("delist_date"),
                    "list_status": row.get("list_status") or list_status,
                    "market": row.get("market"),
                    "industry": row.get("industry"),
                },
            })
        return rows

    def get_instrument_lifecycle(self) -> list[dict]:
        """Return L/D/P stock identities with effective lifecycle dates.

        This is separate from ``get_instruments``: the live universe must keep
        only listed securities, while historical validation needs delisted and
        suspended snapshots as well.
        """
        rows: list[dict] = []
        for status in ("L", "D", "P"):
            status_rows: list[dict] = []
            for exchange in _STOCK_BASIC_EXCHANGES:
                status_rows.extend(self._normalise_stock_basic(
                    self._stock_basic_status(status, exchange=exchange), list_status=status,
                ))
            if not status_rows:
                status_rows = self._normalise_stock_basic(
                    self._stock_basic_status(status), list_status=status,
                )
            rows.extend(status_rows)
        rows = _dedupe_stock_basic_rows(rows, include_status=True)
        # Some Promax snapshots expose D/P rows but omit delist_date.  Query
        # the configured Datahub client directly to enrich the same identity;
        # a missing date must remain visible to the validator, never be
        # silently interpreted as active or delisted for every historical day.
        if self._datahub_client is not None:
            by_key = {(str(r.get("code")), str((r.get("ext") or {}).get("list_status"))): r for r in rows}
            for status in ("D", "P"):
                try:
                    frame = self._datahub_client.stock_basic(
                        list_status=status,
                        fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status,exchange",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("datahub stock_basic(%s) 补充失败: %s", status, type(exc).__name__)
                    continue
                for item in self._normalise_stock_basic(frame, list_status=status):
                    key = (str(item.get("code")), status)
                    current = by_key.get(key)
                    if current is None:
                        rows.append(item)
                        by_key[key] = item
                        continue
                    current_ext = current.setdefault("ext", {})
                    item_ext = item.get("ext") or {}
                    for field in ("listing_date", "delist_date", "list_status", "market", "industry"):
                        if item_ext.get(field) not in (None, "", "nan"):
                            current_ext[field] = item_ext[field]
        return rows

    def _normalize_trade_dates(
        self,
        trade_dates: list[str | date | datetime] | None,
        start: str,
        end: str,
    ) -> list[str]:
        if not trade_dates:
            return []
        normalized: set[str] = set()
        start_key = re.sub(r"\D", "", start or "")[:8]
        end_key = re.sub(r"\D", "", end or "")[:8]
        for value in trade_dates:
            if isinstance(value, (date, datetime)):
                text = value.strftime("%Y%m%d")
            else:
                text = re.sub(r"\D", "", str(value or ""))[:8]
            if len(text) != 8:
                continue
            if start_key and text < start_key:
                continue
            if end_key and text > end_key:
                continue
            normalized.add(text)
        return sorted(normalized)

    def _calendar_trade_dates(self, start: str, end: str) -> list[str]:
        """Promax trade_cal 不稳定时的兜底日期集。

        只用于按 trade_date 的全市场接口。真实休市日会返回空数据，比退回 5000+
        只股票逐个请求更可控。
        """
        start_key = re.sub(r"\D", "", start or "")[:8]
        end_key = re.sub(r"\D", "", end or "")[:8]
        if len(start_key) != 8 or len(end_key) != 8:
            return []
        try:
            cur = datetime.strptime(start_key, "%Y%m%d").date()
            end_date = datetime.strptime(end_key, "%Y%m%d").date()
        except ValueError:
            return []
        if cur > end_date:
            return []

        dates: list[str] = []
        while cur <= end_date:
            if cur.weekday() < 5:
                dates.append(cur.strftime("%Y%m%d"))
            cur += timedelta(days=1)
        return dates

    def _trade_dates(
        self,
        start: str,
        end: str,
        trade_dates: list[str | date | datetime] | None = None,
    ) -> list[str]:
        local_dates = self._normalize_trade_dates(trade_dates, start, end)
        if local_dates:
            return local_dates
        api = self._ensure_api()
        try:
            frame = api.trade_cal(exchange="", start_date=start, end_date=end, is_open=1, fields="cal_date,is_open")
        except Exception as exc:  # noqa: BLE001
            logger.debug("tushare trade_cal failed %s~%s: %s", start, end, exc)
            return self._calendar_trade_dates(start, end) if self._connection_mode() == "promax" else []
        if frame is None or frame.empty or "cal_date" not in frame.columns:
            return self._calendar_trade_dates(start, end) if self._connection_mode() == "promax" else []
        if "is_open" in frame.columns:
            frame = frame[frame["is_open"].astype(str).isin(["1", "1.0"])]
            if frame.empty:
                return []
        dates: set[str] = set()
        for value in frame["cal_date"].dropna().tolist():
            text = re.sub(r"\D", "", str(value or ""))[:8]
            if len(text) == 8 and start <= text <= end:
                dates.add(text)
        if dates or "is_open" in frame.columns:
            return sorted(dates)
        return self._calendar_trade_dates(start, end) if self._connection_mode() == "promax" else []

    def _daily_by_trade_dates(
        self,
        symbols: list[str],
        start: str,
        end: str,
        on_chunk_done: Callable[[int, int], None] | None = None,
        trade_dates: list[str | date | datetime] | None = None,
    ) -> "pl.DataFrame":
        pd = _pd()
        pl = _pl()
        api = self._ensure_api()
        wanted = {_to_ts_code(symbol) for symbol in symbols}
        trade_dates = self._trade_dates(start, end, trade_dates)
        if not trade_dates:
            return pl.DataFrame()
        frames: list[pl.DataFrame] = []
        total = len(trade_dates)
        for index, trade_date in enumerate(trade_dates, 1):
            try:
                if self._connection_mode() == "promax":
                    frame = self._promax_market_day(api.daily, trade_date, page_size=5000)
                else:
                    params = {"trade_date": trade_date}
                    params["fields"] = "ts_code,trade_date,open,high,low,close,vol,amount"
                    frame = api.daily(**params)
            except Exception as exc:  # noqa: BLE001
                logger.debug("tushare daily failed on %s: %s", trade_date, exc)
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if frame is None or frame.empty:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if "ts_code" not in frame.columns:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if wanted:
                frame = frame[frame["ts_code"].isin(wanted)]
            if frame.empty:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            frame = _normalize_dates(frame, "trade_date")
            frame["symbol"] = frame["ts_code"].astype(str).map(_compact_symbol)
            frame["date"] = frame["trade_date"]
            frame["volume"] = pd.to_numeric(frame.get("vol"), errors="coerce") * 100
            frame["amount"] = pd.to_numeric(frame.get("amount"), errors="coerce") * 1000
            for col in ("open", "high", "low", "close"):
                if col in frame.columns:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
            rows = frame[["symbol", "date", "open", "high", "low", "close", "volume", "amount"]].to_dict(orient="records")
            frames.append(_as_polars(rows))
            if on_chunk_done:
                on_chunk_done(index, total)
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def _adj_by_trade_dates(
        self,
        symbols: list[str],
        start: str,
        end: str,
        on_chunk_done: Callable[[int, int], None] | None = None,
        trade_dates: list[str | date | datetime] | None = None,
    ) -> "pl.DataFrame":
        pd = _pd()
        pl = _pl()
        api = self._ensure_api()
        wanted = {_to_ts_code(symbol) for symbol in symbols}
        trade_dates = self._trade_dates(start, end, trade_dates)
        if not trade_dates:
            return pl.DataFrame()
        frames: list[pl.DataFrame] = []
        total = len(trade_dates)
        for index, trade_date in enumerate(trade_dates, 1):
            try:
                params = {"trade_date": trade_date}
                if self._connection_mode() != "promax":
                    params["fields"] = "ts_code,trade_date,adj_factor"
                frame = api.adj_factor(**params)
            except Exception as exc:  # noqa: BLE001
                logger.debug("tushare adj_factor failed on %s: %s", trade_date, exc)
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if frame is None or frame.empty:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if "ts_code" not in frame.columns:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if wanted:
                frame = frame[frame["ts_code"].isin(wanted)]
            if frame.empty:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            frame = _normalize_dates(frame, "trade_date")
            frame["symbol"] = frame["ts_code"].astype(str).map(_compact_symbol)
            frame["trade_date"] = frame["trade_date"]
            frame["ex_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce")
            rows = frame[["symbol", "trade_date", "ex_factor"]].to_dict(orient="records")
            frames.append(_as_polars(rows))
            if on_chunk_done:
                on_chunk_done(index, total)
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def _auction_by_trade_dates(
        self,
        symbols: list[str],
        start: str,
        end: str,
        on_chunk_done: Callable[[int, int], None] | None = None,
        market: str = "o",
        trade_dates: list[str | date | datetime] | None = None,
        strict: bool = False,
    ) -> "pl.DataFrame":
        pd = _pd()
        pl = _pl()
        api = self._ensure_api()
        wanted = {_to_ts_code(symbol) for symbol in symbols}
        trade_dates = self._trade_dates(start, end, trade_dates)
        if not trade_dates:
            return pl.DataFrame()
        frames: list[pl.DataFrame] = []
        total = len(trade_dates)
        api_name = "stk_auction_o" if market == "o" else "stk_auction_c"
        for index, trade_date in enumerate(trade_dates, 1):
            try:
                method = getattr(api, api_name)
                if self._connection_mode() == "promax":
                    frame = self._promax_market_day(method, trade_date)
                else:
                    frame = method(trade_date=trade_date)
            except Exception as exc:  # noqa: BLE001
                failure = _auction_source_error(exc)
                if strict:
                    raise failure from None
                logger.debug("tushare %s failed on %s: %s", api_name, trade_date, failure.reason)
                continue
            if frame is None or frame.empty:
                if strict:
                    raise TushareAuctionSourceError("upstream_empty")
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if wanted and "ts_code" in frame.columns:
                frame = frame[frame["ts_code"].isin(wanted)]
            if frame.empty:
                if strict:
                    raise TushareAuctionSourceError("no_matching_rows")
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            frame = _normalize_dates(frame, "trade_date")
            if "ts_code" in frame.columns:
                frame["symbol"] = frame["ts_code"].astype(str).map(_compact_symbol)
            else:
                frame["symbol"] = frame.get("symbol", "").astype(str).map(_compact_symbol)
            frame["date"] = frame["trade_date"]
            frame["market"] = "open" if market == "o" else "close"
            if "auction_price" in frame.columns and "open" not in frame.columns:
                frame["open"] = pd.to_numeric(frame["auction_price"], errors="coerce")
            if "auction_vol" in frame.columns and "volume" not in frame.columns:
                frame["volume"] = pd.to_numeric(frame["auction_vol"], errors="coerce")
            if "auction_amount" in frame.columns and "amount" not in frame.columns:
                frame["amount"] = pd.to_numeric(frame["auction_amount"], errors="coerce")
            rename_map = {
                "auction_price": "close",
                "auction_vol": "volume",
                "vol": "volume",
                "auction_amount": "amount",
                "vwap": "vwap",
            }
            for src, dst in rename_map.items():
                if src in frame.columns and dst not in frame.columns:
                    frame[dst] = pd.to_numeric(frame[src], errors="coerce")
            if "close" not in frame.columns and "price" in frame.columns:
                frame["close"] = pd.to_numeric(frame["price"], errors="coerce")
            if "high" not in frame.columns and "auction_high" in frame.columns:
                frame["high"] = pd.to_numeric(frame["auction_high"], errors="coerce")
            if "low" not in frame.columns and "auction_low" in frame.columns:
                frame["low"] = pd.to_numeric(frame["auction_low"], errors="coerce")
            if "open" not in frame.columns and "price" in frame.columns:
                frame["open"] = pd.to_numeric(frame["price"], errors="coerce")
            if "vwap" not in frame.columns and "avg_price" in frame.columns:
                frame["vwap"] = pd.to_numeric(frame["avg_price"], errors="coerce")
            rows = frame.to_dict(orient="records")
            frames.append(_as_polars(rows))
            if on_chunk_done:
                on_chunk_done(index, total)
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    @staticmethod
    def _promax_market_day(method, trade_date: str, *, page_size: int = 6000, max_pages: int = 10):
        """Avoid cached short default pages and reject incomplete pagination."""
        pd = _pd()
        expected_date = datetime.strptime(trade_date, "%Y%m%d").date()
        frames, seen = [], set()
        for index in range(max_pages):
            page = method(trade_date=trade_date, limit=page_size, offset=index * page_size)
            if page is None or page.empty:
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            raw_page_size = len(page)
            page = page.drop_duplicates().copy()
            if not {"ts_code", "trade_date"} <= set(page.columns):
                raise ValueError("Gateway market page lacks stock/date identifiers")
            dates = pd.to_datetime(page["trade_date"].astype("string"), errors="coerce", format="mixed").dt.date
            if dates.isna().any() or not (dates == expected_date).all():
                raise ValueError("Gateway market page returned another trading date")
            codes = page["ts_code"]
            if codes.isna().any() or codes.duplicated().any() or seen.intersection(codes):
                raise ValueError("Gateway market pagination returned repeated stock-date keys")
            seen.update(codes)
            frames.append(page.copy())
            if raw_page_size < page_size:
                return pd.concat(frames, ignore_index=True)
        raise ValueError("Gateway market pagination reached the page limit without exhaustion")

    def _moneyflow_by_trade_dates(
        self,
        symbols: list[str],
        start: str,
        end: str,
        on_chunk_done: Callable[[int, int], None] | None = None,
        trade_dates: list[str | date | datetime] | None = None,
    ) -> "pl.DataFrame":
        """按交易日获取全市场资金流向，避免逐股请求造成不必要的限流。"""
        pd = _pd()
        pl = _pl()
        api = self._ensure_api()
        wanted = {_to_ts_code(symbol) for symbol in symbols}
        trade_dates = self._trade_dates(start, end, trade_dates)
        if not trade_dates:
            return pl.DataFrame()

        fields = ",".join((
            "ts_code,trade_date",
            "buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount",
            "buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount",
            "buy_lg_vol,buy_lg_amount,sell_lg_vol,sell_lg_amount",
            "buy_elg_vol,buy_elg_amount,sell_elg_vol,sell_elg_amount",
            "net_mf_vol,net_mf_amount",
        ))
        amount_columns = [column for column in _MONEYFLOW_COLS if column.endswith("_amount")]
        frames: list[pl.DataFrame] = []
        total = len(trade_dates)

        for index, trade_date in enumerate(trade_dates, 1):
            try:
                params = {"trade_date": trade_date}
                if self._connection_mode() == "promax":
                    frame = self._promax_market_day(api.moneyflow, trade_date)
                else:
                    params["fields"] = fields
                    frame = api.moneyflow(**params)
            except Exception as exc:  # noqa: BLE001
                logger.debug("tushare moneyflow failed on %s: %s", trade_date, exc)
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if frame is None or frame.empty:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if wanted:
                frame = frame[frame["ts_code"].isin(wanted)]
            if frame.empty:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue

            frame = _normalize_dates(frame, "trade_date")
            frame["symbol"] = frame["ts_code"].astype(str).map(_compact_symbol)
            frame["date"] = frame["trade_date"]
            # Tushare moneyflow 金额单位为万元；本地统一存人民币元，避免下游重复换算。
            for column in amount_columns:
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce") * 10000
            for column in _MONEYFLOW_COLS:
                if column.endswith("_vol") and column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frames.append(_as_polars(frame.to_dict(orient="records")))
            if on_chunk_done:
                on_chunk_done(index, total)

        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def _moneyflow_by_symbols(
        self,
        symbols: list[str],
        start: str,
        end: str,
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> "pl.DataFrame":
        """Promax 单股区间查询，用于快速测试和个股分析。"""
        if self._connection_mode() != "promax":
            return _pl().DataFrame()
        pd = _pd()
        pl = _pl()
        api = self._ensure_api()
        amount_columns = [column for column in _MONEYFLOW_COLS if column.endswith("_amount")]
        frames: list[pl.DataFrame] = []
        total = len(symbols)
        for index, symbol in enumerate(symbols, 1):
            try:
                frame = api.moneyflow(
                    ts_code=_to_ts_code(symbol),
                    start_date=start,
                    end_date=end,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("tushare moneyflow symbol failed for %s: %s", symbol, exc)
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if frame is not None and not frame.empty:
                frame = _normalize_dates(frame, "trade_date")
                frame["symbol"] = frame["ts_code"].astype(str).map(_compact_symbol)
                frame["date"] = frame["trade_date"]
                for column in amount_columns:
                    if column in frame.columns:
                        frame[column] = pd.to_numeric(frame[column], errors="coerce") * 10000
                for column in _MONEYFLOW_COLS:
                    if column.endswith("_vol") and column in frame.columns:
                        frame[column] = pd.to_numeric(frame[column], errors="coerce")
                frames.append(_as_polars(frame.to_dict(orient="records")))
            if on_chunk_done:
                on_chunk_done(index, total)
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> "pl.DataFrame":
        pl = _pl()
        if asset_type in ("index", "etf"):
            return self.get_asset_daily(symbols, start_time, end_time, asset_type, on_chunk_done)
        if asset_type != "stock" or not symbols:
            return pl.DataFrame()
        end_value = end_time or datetime.now()
        start_value = start_time or (end_value - timedelta(days=365))
        start = _to_yyyymmdd(start_value)
        end = _to_yyyymmdd(end_value)
        if not start or not end:
            return pl.DataFrame()
        try:
            df = self._daily_by_trade_dates(symbols, start, end, on_chunk_done=on_chunk_done)
            if not df.is_empty():
                return df.sort(["symbol", "date"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("tushare 日线批量失败: %s", exc)
        if len(symbols) > 200:
            logger.info(
                "tushare 日线全市场查询未返回数据 %s~%s，跳过逐股兜底以避免长时间阻塞",
                start,
                end,
            )
            return pl.DataFrame()
        # 兜底:按标的逐个拉单股历史
        frames: list["pl.DataFrame"] = []
        api = self._ensure_api()
        total = len(symbols)
        pd = _pd()
        for index, symbol in enumerate(symbols, 1):
            try:
                frame = api.daily(
                    ts_code=_to_ts_code(symbol),
                    start_date=start,
                    end_date=end,
                    fields="ts_code,trade_date,open,high,low,close,vol,amount",
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("tushare daily failed for %s: %s", symbol, exc)
                continue
            if frame is None or frame.empty:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            frame = _normalize_dates(frame, "trade_date")
            frame["symbol"] = frame["ts_code"].astype(str).map(_compact_symbol)
            frame["date"] = frame["trade_date"]
            frame["volume"] = pd.to_numeric(frame.get("vol"), errors="coerce") * 100
            frame["amount"] = pd.to_numeric(frame.get("amount"), errors="coerce") * 1000
            for col in ("open", "high", "low", "close"):
                if col in frame.columns:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
            rows = frame[["symbol", "date", "open", "high", "low", "close", "volume", "amount"]].to_dict(orient="records")
            frames.append(_as_polars(rows))
            if on_chunk_done:
                on_chunk_done(index, total)
        return pl.concat(frames, how="diagonal_relaxed").sort(["symbol", "date"]) if frames else pl.DataFrame()

    def get_asset_daily(
        self, symbols, start_time, end_time, asset_type, on_chunk_done=None,
    ):
        """Use asset-specific endpoints; an index must never go through stock daily."""
        if asset_type not in ("index", "etf"):
            raise ValueError("Unsupported daily asset type")
        pl, pd = _pl(), _pd()
        if not symbols:
            return pl.DataFrame()
        end = (end_time or datetime.now()).date()
        start = (start_time or datetime.combine(end - timedelta(days=365), datetime.min.time())).date()
        wanted = {str(s).upper() for s in symbols}
        if any("." not in s for s in wanted):
            raise ValueError("Index/ETF daily requires exchange-qualified symbols")
        api = self._ensure_api()
        frames = []

        def normalize(frame):
            if frame is None or frame.empty:
                return
            required = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}
            if not required <= set(frame.columns):
                raise ValueError("Asset daily response lacks required fields")
            frame = _normalize_dates(frame.copy(), "trade_date")
            frame["symbol"] = frame["ts_code"].astype(str).str.upper()
            frame["date"] = pd.to_datetime(frame["trade_date"]).dt.date
            if not frame["date"].between(start, end).all():
                raise ValueError("Asset daily returned another trading date")
            frame = frame[frame["symbol"].isin(wanted)].copy()
            for col in ("open", "high", "low", "close", "vol", "amount"):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            valid = (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
            valid &= frame["high"] >= frame[["open", "close", "low"]].max(axis=1)
            valid &= frame["low"] <= frame[["open", "close"]].min(axis=1)
            valid &= (frame[["vol", "amount"]] >= 0).all(axis=1)
            frame = frame[valid].copy()
            frame["volume"] = frame["vol"] * 100
            frame["amount"] = frame["amount"] * 1000
            if not frame.empty:
                frames.append(_as_polars(frame[_DAILY_COLS].to_dict(orient="records")))

        if asset_type == "etf":
            dates = self._trade_dates(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
            for i, day in enumerate(dates, 1):
                normalize(self._promax_market_day(api.fund_daily, day))
                if on_chunk_done:
                    on_chunk_done(i, len(dates))
        else:
            # index_daily requires one index code. Bound concurrency and stop a
            # failed wave instead of issuing hundreds of requests to a dead gateway.
            from concurrent.futures import ThreadPoolExecutor
            ordered = sorted(wanted)

            def fetch(symbol):
                return api.index_daily(ts_code=symbol, start_date=start.strftime("%Y%m%d"),
                                       end_date=end.strftime("%Y%m%d"))

            failures = 0
            last_error = None
            with ThreadPoolExecutor(max_workers=2) as executor:
                for offset in range(0, len(ordered), 2):
                    futures = [executor.submit(fetch, s) for s in ordered[offset:offset + 2]]
                    for future in futures:
                        try:
                            normalize(future.result())
                            failures = 0
                        except Exception as exc:
                            failures += 1
                            last_error = exc
                            logger.warning("index_daily request failed: %s", type(exc).__name__)
                    if on_chunk_done:
                        on_chunk_done(min(offset + 2, len(ordered)), len(ordered))
                    if failures >= 3:
                        break
            if not frames and last_error is not None:
                raise last_error
        return (pl.concat(frames, how="diagonal_relaxed").unique(["symbol", "date"]).sort(["symbol", "date"])
                if frames else pl.DataFrame())

    def get_adj_factors(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> "pl.DataFrame":
        pl = _pl()
        if asset_type != "stock" or not symbols:
            return pl.DataFrame()
        end_value = end_time or datetime.now()
        start_value = start_time or (end_value - timedelta(days=365))
        start = _to_yyyymmdd(start_value)
        end = _to_yyyymmdd(end_value)
        if not start or not end:
            return pl.DataFrame()
        try:
            df = self._adj_by_trade_dates(symbols, start, end, on_chunk_done=on_chunk_done)
            if not df.is_empty():
                return df.sort(["symbol", "trade_date"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("tushare 复权因子批量失败: %s", exc)
        if len(symbols) > 200:
            logger.info(
                "tushare 复权因子全市场查询未返回数据 %s~%s，跳过逐股兜底以避免长时间阻塞",
                start,
                end,
            )
            return pl.DataFrame()
        # 兜底:按标的逐个拉单股历史
        frames: list["pl.DataFrame"] = []
        api = self._ensure_api()
        total = len(symbols)
        pd = _pd()
        for index, symbol in enumerate(symbols, 1):
            try:
                frame = api.adj_factor(
                    ts_code=_to_ts_code(symbol),
                    start_date=start,
                    end_date=end,
                    fields="ts_code,trade_date,adj_factor",
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("tushare adj_factor failed for %s: %s", symbol, exc)
                continue
            if frame is None or frame.empty:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            frame = _normalize_dates(frame, "trade_date")
            frame["symbol"] = frame["ts_code"].astype(str).map(_compact_symbol)
            frame["ex_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce")
            rows = frame[["symbol", "trade_date", "ex_factor"]].to_dict(orient="records")
            frames.append(_as_polars(rows))
            if on_chunk_done:
                on_chunk_done(index, total)
        return pl.concat(frames, how="diagonal_relaxed").sort(["symbol", "trade_date"]) if frames else pl.DataFrame()

    def _shares_from_daily_basic_by_trade_dates(
        self,
        symbols: list[str],
        start: str,
        end: str,
        *,
        latest_only: bool,
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> "pl.DataFrame":
        """用 daily_basic 按交易日批量同步流通股本。

        share_float 是逐股事件表，覆盖和速度都不稳定；daily_basic 的
        total_share/float_share 是日快照，更适合换手率、回测和实时异动兜底。
        Tushare 单位为万股，本地统一落成股。
        """
        pd = _pd()
        pl = _pl()
        api = self._ensure_api()
        wanted = {_to_ts_code(symbol) for symbol in symbols}
        trade_dates = self._trade_dates(start, end)
        if not trade_dates:
            return pl.DataFrame()
        if latest_only:
            trade_dates = list(reversed(trade_dates))

        fields = "ts_code,trade_date,total_share,float_share"
        frames: list[pl.DataFrame] = []
        seen: set[str] = set()
        total = len(trade_dates)

        for index, trade_date in enumerate(trade_dates, 1):
            try:
                params = {"trade_date": trade_date}
                if self._connection_mode() != "promax":
                    params["fields"] = fields
                frame = api.daily_basic(**params)
            except Exception as exc:  # noqa: BLE001
                logger.debug("tushare daily_basic shares failed on %s: %s", trade_date, exc)
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if frame is None or frame.empty or "ts_code" not in frame.columns:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue
            if wanted:
                frame = frame[frame["ts_code"].isin(wanted)]
            if frame.empty:
                if on_chunk_done:
                    on_chunk_done(index, total)
                continue

            frame = _normalize_dates(frame, "trade_date")
            frame["symbol"] = frame["ts_code"].astype(str).map(_compact_symbol)
            if latest_only:
                frame = frame[~frame["symbol"].isin(seen)]
                if frame.empty:
                    if on_chunk_done:
                        on_chunk_done(index, total)
                    continue
                seen.update(frame["symbol"].dropna().astype(str).tolist())

            frame["period_end"] = frame["trade_date"]
            frame["announce_date"] = None
            for source, target in _FINANCIAL_MAP["shares"].items():
                if source in frame.columns:
                    frame[target] = pd.to_numeric(frame[source], errors="coerce") * 10000
            keep = [
                column
                for column in ("symbol", "period_end", "announce_date", "float_shares", "total_shares")
                if column in frame.columns
            ]
            out = pl.from_pandas(frame[keep], include_index=False)
            out = out.with_columns(
                pl.col("period_end").cast(pl.Date, strict=False),
                pl.col("announce_date").cast(pl.Date, strict=False),
                pl.col("float_shares").cast(pl.Float64, strict=False),
                pl.col("total_shares").cast(pl.Float64, strict=False),
            ).filter(pl.col("float_shares").is_not_null() | pl.col("total_shares").is_not_null())
            if not out.is_empty():
                frames.append(out)
            if on_chunk_done:
                on_chunk_done(index, total)
            if latest_only and wanted and {_compact_symbol(symbol) for symbol in symbols} <= seen:
                break

        if not frames:
            return pl.DataFrame()
        result = pl.concat(frames, how="diagonal_relaxed")
        if latest_only:
            return result.sort(["symbol", "period_end"]).unique(subset=["symbol"], keep="last")
        return result.unique(subset=["symbol", "period_end"], keep="last").sort(["symbol", "period_end"])

    def _normalize_financial_frame(
        self,
        frame: "pd.DataFrame",
        table: str,
        symbol: str,
    ) -> "pl.DataFrame":
        """把 Tushare 财务字段映射成项目现有财务表字段。"""
        pd = _pd()
        pl = _pl()
        if frame is None or frame.empty:
            return pl.DataFrame()

        frame = frame.copy()
        frame["symbol"] = _compact_symbol(symbol)
        if "end_date" in frame.columns:
            frame["period_end"] = frame["end_date"]
        elif "float_date" in frame.columns:
            frame["period_end"] = frame["float_date"]
        elif "trade_date" in frame.columns:
            frame["period_end"] = frame["trade_date"]
        if "ann_date" in frame.columns:
            frame["announce_date"] = frame["ann_date"]
        if "f_ann_date" in frame.columns:
            # Restated reports become available at the actual announcement,
            # never retroactively at the original report's announcement.
            actual_text = frame["f_ann_date"].astype("string").str.strip()
            actual = pd.to_datetime(actual_text, format="mixed", errors="coerce")
            announced = pd.to_datetime(
                frame.get("announce_date", pd.Series(None, index=frame.index)).astype("string"),
                format="mixed", errors="coerce",
            )
            effective = pd.concat([announced, actual], axis=1).max(axis=1)
            invalid_actual = actual_text.notna() & actual_text.ne("") & actual.isna()
            frame["announce_date"] = effective.mask(invalid_actual)

        for source, target in _FINANCIAL_MAP.get(table, {}).items():
            if source in frame.columns and target not in frame.columns:
                frame[target] = frame[source]

        if table == "shares":
            # Tushare 股本接口的 share 单位为万股；项目内部统一保存为股。
            for column in ("float_shares", "total_shares"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce") * 10000

        for column in ("period_end", "announce_date"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column].astype("string"), format="mixed", errors="coerce").dt.date

        base = ["symbol", "period_end", "announce_date"]
        mapped = list(dict.fromkeys(_FINANCIAL_MAP.get(table, {}).values()))
        keep = [column for column in dict.fromkeys([*base, *mapped]) if column in frame.columns]
        if "symbol" not in keep or "period_end" not in keep:
            return pl.DataFrame()
        result = pl.from_pandas(frame[keep], include_index=False)
        for column in mapped:
            if column in result.columns:
                result = result.with_columns(pl.col(column).cast(pl.Float64, strict=False))
        return result

    def _fetch_financial_frame(
        self,
        table: str,
        symbol: str,
        latest_only: bool,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> "pl.DataFrame":
        """按股票拉取财务表；财务接口通常按单股返回，避免跨接口字段错配。"""
        pl = _pl()
        api = self._ensure_api()
        method_name = _FINANCIAL_METHODS.get(table)
        if not method_name:
            return pl.DataFrame()
        method = getattr(api, method_name, None)
        if method is None:
            return pl.DataFrame()

        kwargs: dict[str, Any] = {"ts_code": _to_ts_code(symbol)}
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        if table == "shares":
            kwargs["fields"] = "ts_code,ann_date,float_date,float_share,total_share"

        try:
            frame = method(**kwargs)
        except TypeError:
            # 少数兼容网关不接受日期参数，回退到逐股全量响应后在本地过滤。
            if table == "shares":
                frame = method(
                    ts_code=_to_ts_code(symbol),
                    fields="ts_code,ann_date,float_date,float_share,total_share",
                )
            else:
                frame = method(ts_code=_to_ts_code(symbol))
        except Exception as exc:  # noqa: BLE001
            logger.debug("tushare %s failed for %s: %s", table, symbol, exc)
            return pl.DataFrame()
        if frame is None or frame.empty:
            if table != "shares":
                return pl.DataFrame()
            # share_float 权限或历史数据不足时，使用 daily_basic 的流通/总股本快照兜底。
            try:
                frame = api.daily_basic(
                    ts_code=_to_ts_code(symbol),
                    start_date=(datetime.now() - timedelta(days=3650)).strftime("%Y%m%d"),
                    end_date=datetime.now().strftime("%Y%m%d"),
                    fields="ts_code,trade_date,total_share,float_share",
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("tushare daily_basic shares failed for %s: %s", symbol, exc)
                return pl.DataFrame()
            if frame is None or frame.empty:
                return pl.DataFrame()
            frame = frame.rename(columns={"trade_date": "float_date"})

        if latest_only:
            sort_column = "end_date" if "end_date" in frame.columns else (
                "float_date" if "float_date" in frame.columns else "trade_date"
            )
            if sort_column in frame.columns:
                frame = frame.sort_values(sort_column, ascending=False).head(1)
        return self._normalize_financial_frame(frame, table, symbol)

    def get_financial_updates(self, table: str, day: date) -> "pl.DataFrame":
        """Fetch market-wide announcements, failing closed if a gateway ignores ann_date."""
        pl = _pl()
        if table == "shares":
            stamp = day.strftime("%Y%m%d")
            frame = self._shares_from_daily_basic_by_trade_dates([], stamp, stamp, latest_only=False)
            if not frame.is_empty() and frame.filter(
                pl.col("period_end").is_null() | (pl.col("period_end") != day)
            ).height:
                raise ValueError("shares gateway ignored requested trading date")
            return frame
        if table not in _FINANCIAL_METHODS:
            raise ValueError("unsupported announcement table")
        frame = getattr(self._ensure_api(), _FINANCIAL_METHODS[table])(
            ann_date=day.strftime("%Y%m%d"),
        )
        if frame is None or frame.empty:
            return pl.DataFrame()
        if not {"ts_code", "ann_date", "end_date"} <= set(frame.columns):
            raise ValueError("financial response is missing announcement identifiers")
        if len(frame) >= 5000:
            raise ValueError("financial response may be truncated; use paginated/offline history")
        announcements = _pd().to_datetime(frame["ann_date"], errors="coerce").dt.date
        if announcements.isna().any() or not (announcements == day).all():
            raise ValueError("financial gateway ignored requested announcement date")
        pieces = [self._normalize_financial_frame(group, table, str(symbol))
                  for symbol, group in frame.groupby("ts_code")]
        return pl.concat(pieces, how="diagonal_relaxed") if pieces else pl.DataFrame()

    def get_financials(
        self,
        table: str,
        symbols: list[str],
        latest_only: bool = True,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> "pl.DataFrame":
        pl = _pl()
        if table not in _FINANCIAL_METHODS or not symbols:
            return pl.DataFrame()
        frames: list["pl.DataFrame"] = []
        start_date = _to_yyyymmdd(start_time)
        end_date = _to_yyyymmdd(end_time)
        if table == "shares":
            end_value = end_time or datetime.now()
            start_value = start_time or (
                end_value - timedelta(days=14 if latest_only else 3650)
            )
            batch_start = start_date or _to_yyyymmdd(start_value)
            batch_end = end_date or _to_yyyymmdd(end_value)
            if batch_start and batch_end:
                try:
                    df = self._shares_from_daily_basic_by_trade_dates(
                        symbols,
                        batch_start,
                        batch_end,
                        latest_only=latest_only,
                        on_chunk_done=on_chunk_done,
                    )
                    if not df.is_empty():
                        return df
                except Exception as exc:  # noqa: BLE001
                    logger.warning("tushare daily_basic 股本批量同步失败: %s", exc)

        total = len(symbols)
        for index, symbol in enumerate(symbols, 1):
            frame = self._fetch_financial_frame(
                table,
                symbol,
                latest_only,
                start_date=start_date,
                end_date=end_date,
            )
            if not frame.is_empty():
                frames.append(frame)
            if on_chunk_done:
                on_chunk_done(index, total)
        if not frames:
            return pl.DataFrame()
        result = pl.concat(frames, how="diagonal_relaxed")
        if not latest_only and "period_end" in result.columns:
            if start_time is not None:
                result = result.filter(pl.col("period_end") >= start_time.date())
            if end_time is not None:
                result = result.filter(pl.col("period_end") <= end_time.date())
        return result.unique(subset=["symbol", "period_end"], keep="last").sort(["symbol", "period_end"])

    def get_auction(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        market: str = "o",
        on_chunk_done: Callable[[int, int], None] | None = None,
        trade_dates: list[str | date | datetime] | None = None,
        strict: bool = False,
    ) -> "pl.DataFrame":
        pl = _pl()
        if not symbols:
            return pl.DataFrame()
        end_value = end_time or datetime.now()
        start_value = start_time or (end_value - timedelta(days=365))
        start = _to_yyyymmdd(start_value)
        end = _to_yyyymmdd(end_value)
        if not start or not end:
            return pl.DataFrame()
        try:
            df = self._auction_by_trade_dates(
                symbols,
                start,
                end,
                on_chunk_done=on_chunk_done,
                market=market,
                trade_dates=trade_dates,
                strict=strict,
            )
            if not df.is_empty():
                return df.sort(["symbol", "date"])
        except Exception as exc:  # noqa: BLE001
            failure = _auction_source_error(exc)
            if strict:
                raise failure from None
            logger.warning("tushare 集合竞价批量失败: %s", failure.reason)
        return pl.DataFrame()

    def get_moneyflow(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        on_chunk_done: Callable[[int, int], None] | None = None,
        trade_dates: list[str | date | datetime] | None = None,
    ) -> "pl.DataFrame":
        pl = _pl()
        if not symbols:
            return pl.DataFrame()
        end_value = end_time or datetime.now()
        start_value = start_time or (end_value - timedelta(days=365))
        start = _to_yyyymmdd(start_value)
        end = _to_yyyymmdd(end_value)
        if not start or not end:
            return pl.DataFrame()
        try:
            if self._connection_mode() == "promax" and len(symbols) <= 50 and trade_dates is None:
                df = self._moneyflow_by_symbols(symbols, start, end, on_chunk_done=on_chunk_done)
                if not df.is_empty():
                    return df.sort(["symbol", "date"])
            df = self._moneyflow_by_trade_dates(
                symbols,
                start,
                end,
                on_chunk_done=on_chunk_done,
                trade_dates=trade_dates,
            )
            if not df.is_empty():
                return df.sort(["symbol", "date"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("tushare 资金流向批量失败: %s", exc)
        return pl.DataFrame()

    def get_news(
        self,
        sources: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> "pl.DataFrame":
        """读取 Tushare 历史财经新闻并规整为本地新闻库字段。"""
        pl = _pl()
        empty_schema = {column: pl.Utf8 for column in _NEWS_COLS}
        if not sources:
            return pl.DataFrame(schema=empty_schema)
        start_value = start_time or (datetime.now() - timedelta(days=30))
        end_value = end_time or datetime.now()
        start = start_value.strftime("%Y-%m-%d %H:%M:%S")
        end = end_value.strftime("%Y-%m-%d %H:%M:%S")
        api = self._ensure_api()
        frames = []

        for index, source in enumerate(sources, start=1):
            src = str(source or "").strip()
            if not src:
                continue
            try:
                frame = api.news(src=src, start_date=start, end_date=end)
            except TypeError:
                frame = api.news(src=src, start_date=start, end_date=end, fields="")
            except Exception as exc:  # noqa: BLE001
                logger.debug("tushare news failed for %s: %s", src, exc)
                if on_chunk_done:
                    on_chunk_done(index, len(sources))
                continue
            if frame is not None and not frame.empty:
                frames.append(self._normalize_news_frame(frame, src))
            if on_chunk_done:
                on_chunk_done(index, len(sources))

        if not frames:
            return pl.DataFrame(schema=empty_schema)
        result = pl.concat(frames, how="diagonal_relaxed")
        return result.unique(subset=["id"], keep="last").sort(["date", "published_at"])

    @staticmethod
    def _normalize_news_frame(frame: "pd.DataFrame", source: str) -> "pl.DataFrame":
        pl = _pl()
        rows: list[dict[str, Any]] = []
        for item in frame.to_dict(orient="records"):
            published = _parse_news_datetime(
                item.get("datetime") or item.get("pub_time") or item.get("date")
            )
            if published is None:
                continue
            title = str(item.get("title") or item.get("name") or "").strip()
            content = str(item.get("content") or item.get("summary") or "").strip()
            if not title and not content:
                continue
            row_source = str(item.get("src") or source or "tushare").strip()
            fingerprint = "|".join((row_source, published.isoformat(), title, content[:600]))
            rows.append({
                "id": hashlib.sha1(fingerprint.encode("utf-8")).hexdigest(),
                "date": published.date(),
                "published_at": published.strftime("%Y-%m-%d %H:%M:%S"),
                "market": "domestic",
                "market_scope": "A股/国内财经",
                "source": f"Tushare/{row_source}",
                "title": title or content[:80],
                "summary": content[:800],
                "content": content,
                "channels": str(item.get("channels") or ""),
                "symbols": _extract_cn_symbols(f"{title} {content}"),
                "url": str(item.get("url") or ""),
                "language": "zh",
                "sentiment": "",
                "raw": str(item),
            })
        if not rows:
            return pl.DataFrame(schema={column: pl.Utf8 for column in _NEWS_COLS})
        result = pl.DataFrame(rows)
        return result.with_columns(pl.col("date").cast(pl.Date, strict=False)).select(_NEWS_COLS)

    def get_minute_diagnostics(self, *args, **kwargs):
        """Optional maintenance capability: bars plus per-day validation outcomes."""
        issues = []
        try:
            frame = self.get_minute(*args, **kwargs, _issues=issues)
        except TushareMinuteSourceError as exc:
            if not issues:
                issues.append(minute_error_details(exc))
            frame = _pl().DataFrame()
        return frame, issues

    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        freq: str = "1m",
        on_chunk_done: Callable[[int, int], None] | None = None,
        *,
        _issues: list[dict] | None = None,
    ) -> "pl.DataFrame":
        pl = _pl()
        if asset_type != "stock" or not symbols:
            return pl.DataFrame()

        api = self._ensure_api()
        raw_end_value = end_time or datetime.now()
        raw_start_value = start_time or (raw_end_value - timedelta(days=5))
        end_value = _to_beijing_wallclock(raw_end_value)
        start_value = _to_beijing_wallclock(raw_start_value)
        if start_value > end_value:
            return pl.DataFrame()

        if any(_minute_symbol_windows(s, start_value, end_value)[0][0] != s for s in symbols):
            frames, errors = [], []
            for index, symbol in enumerate(symbols, 1):
                for query, begin, finish in _minute_symbol_windows(symbol, start_value, end_value):
                    outcomes = []
                    try:
                        part = self.get_minute([query], begin, finish, asset_type, freq, _issues=outcomes)
                        if not part.is_empty():
                            frames.append(part.with_columns(pl.lit(_compact_symbol(symbol)).alias("symbol")))
                    except TushareMinuteSourceError as exc:
                        errors.append(exc)
                    if _issues is not None:
                        _issues.extend({**item, "symbol": _compact_symbol(symbol), "source_symbol": _compact_symbol(query)} for item in outcomes)
                if on_chunk_done:
                    on_chunk_done(index, len(symbols))
            if frames:
                return pl.concat(frames, how="diagonal_relaxed").sort(["symbol", "datetime"])
            if errors:
                raise errors[-1]
            return pl.DataFrame()

        freq_value = _minute_freq(freq)
        start_text = start_value.strftime("%Y-%m-%d %H:%M:%S")
        end_text = end_value.strftime("%Y-%m-%d %H:%M:%S")
        now_cn = datetime.now(_CN_TZ).replace(tzinfo=None)
        today = now_cn.date()
        include_today = start_value.date() <= today <= end_value.date()
        closed_end = end_value.date() < today or (end_value.date() == today and (now_cn.hour, now_cn.minute) >= (15, 10))
        verified_history = bool(getattr(api, "supports_verified_history", isinstance(api, _PromaxTushareAPI))) \
            and freq_value == "1min" and closed_end
        use_rt = include_today and freq_value == "1min" and not verified_history
        if verified_history:
            # Promax may exclude an exact start bound and label the closing
            # auction 15:01. Fetch full boundaries, validate, then trim output.
            start_text = start_value.replace(hour=9, minute=25, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            end_text = end_value.replace(hour=15, minute=10, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")

        frames: list["pl.DataFrame"] = []
        request_errors: list[Exception] = []
        consecutive_transient_errors = 0
        total = len(symbols)
        for index, symbol in enumerate(symbols, 1):
            errors_before = len(request_errors)
            ts_code = _to_ts_code(symbol)
            request_context = {"symbol": _compact_symbol(symbol), "start": str(start_value.date()), "end": str(end_value.date())}
            # 盘中当日分钟优先 rt_min_daily；今天的增量轮如果同时调用 stk_mins
            # 和 rt_min_daily，会把 Tushare 网关请求量翻倍，还容易触发 503。
            should_call_history = not (use_rt and start_value.date() == today)
            if should_call_history:
                try:
                    try:
                        frame = api.stk_mins(
                            ts_code=ts_code, freq=freq_value, start_date=start_text, end_date=end_text,
                        )
                    except Exception as primary_error:
                        detail = minute_error_details(primary_error)
                        if not verified_history or detail["reason"] not in {"request_timeout", "transport_error", "endpoint_not_found"}:
                            raise
                        # Promax's local-history endpoint only reliably supports one
                        # stock per request. The same strict daily gate applies below.
                        logger.warning("Promax stk_mins %s for %s; trying a_share_mins",
                                       detail["reason"], ts_code)
                        frame = api.a_share_mins(ts_code=ts_code, freq="1MIN", limit=5000,
                                                 start_date=start_text, end_date=end_text)
                    if verified_history and frame is not None and not frame.empty:
                        from pathlib import Path
                        from app.config import settings
                        from app.plugins.tushare.minute_contract import normalize_history

                        daily = api.daily(ts_code=ts_code, start_date=start_value.strftime("%Y%m%d"), end_date=end_value.strftime("%Y%m%d"))
                        df = normalize_history(frame, daily, symbol, Path(settings.data_dir) / "import_reports" / "gateway_minute_contract", outcomes=_issues)
                        if df.is_empty():
                            raise TushareMinuteSourceError("Promax historical minute validation failed; see gateway_minute_contract audit", reason="validation_failed")
                    else:
                        df = _normalize_minute_frame(frame, symbol)
                    if not df.is_empty():
                        frames.append(df)
                    elif _issues is not None:
                        _issues.append({**request_context, "reason": "source_empty", "retryable": False})
                except Exception as exc:  # noqa: BLE001
                    request_errors.append(exc)
                    if _issues is not None and getattr(exc, "reason", None) != "validation_failed":
                        _issues.append({**request_context, **minute_error_details(exc)})
                    logger.debug("tushare stk_mins failed for %s: %s", ts_code, type(exc).__name__)

            # 当日盘中分钟优先用 rt_min_daily 覆盖。这个接口通常更接近实时；
            # 若用户没有权限或网关不支持，历史 stk_mins 结果仍可使用。
            if use_rt:
                try:
                    rt = api.rt_min_daily(ts_code=ts_code)
                    rt_df = _normalize_minute_frame(rt, symbol)
                    if not rt_df.is_empty():
                        frames.append(rt_df)
                except Exception as exc:  # noqa: BLE001
                    request_errors.append(exc)
                    if _issues is not None:
                        _issues.append({**request_context, **minute_error_details(exc)})
                    logger.debug("tushare rt_min_daily failed for %s: %s", ts_code, type(exc).__name__)

            if on_chunk_done:
                on_chunk_done(index, total)

            # A market-wide request must yield when the gateway is preparing
            # every sampled symbol, instead of holding the sync slot for hours.
            detail = minute_error_details(request_errors[-1]) if len(request_errors) > errors_before else {}
            consecutive_transient_errors = consecutive_transient_errors + 1 if detail.get("retryable") else 0
            if consecutive_transient_errors >= 3:
                if _issues is not None:
                    if _issues:
                        _issues[-1]["batch_aborted"] = True
                    _issues.extend({"symbol": _compact_symbol(s), "start": str(start_value.date()),
                                    "end": str(end_value.date()), **detail, "reason": "not_requested",
                                    "cause": detail["reason"]} for s in symbols[index:])
                if frames:
                    break
                if isinstance(request_errors[-1], TushareMinuteSourceError):
                    raise request_errors[-1]
                raise TushareMinuteSourceError("Minute source repeatedly unavailable; retry this increment later",
                                              reason=detail["reason"], retryable=True) from None

        if not frames:
            if request_errors:
                if any(isinstance(exc, TushareMinuteRateLimitError) or getattr(getattr(exc, "response", None), "status_code", None) == 429 for exc in request_errors):
                    raise TushareMinuteRateLimitError("Tushare minute endpoint rate limited (429); not an empty dataset") from None
                if isinstance(request_errors[-1], TushareMinutePendingError):
                    raise request_errors[-1] from None
                detail = minute_error_details(request_errors[-1])
                raise TushareMinuteSourceError("Tushare minute request or normalization failed; not an empty dataset",
                                              reason=detail["reason"], retryable=detail["retryable"]) from None
            return pl.DataFrame()

        combined = pl.concat(frames, how="diagonal_relaxed")
        return (
            combined
            .filter((pl.col("datetime") >= start_value) & (pl.col("datetime") <= end_value))
            .unique(subset=["symbol", "datetime"], keep="last")
            .sort(["symbol", "datetime"])
            .select(_MINUTE_COLS)
        )

    def get_realtime(self, universes=None, symbols=None) -> "pl.DataFrame":
        pl = _pl()
        return pl.DataFrame()

    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        test_symbols = symbols or ["000001.SZ"]
        end = datetime.now()
        start = end - timedelta(days=7)
        if dataset == "daily":
            df = self.get_daily(test_symbols, start, end)
        elif dataset == "adj_factor":
            df = self.get_adj_factors(test_symbols, start, end)
        elif dataset == "financial":
            df = self.get_financials("metrics", test_symbols, latest_only=True)
        elif dataset == "auction":
            df = self.get_auction(test_symbols, start, end)
        elif dataset == "moneyflow":
            df = self.get_moneyflow(test_symbols, start, end)
        elif dataset == "news":
            df = self.get_news(["cls"], start, end)
        elif dataset == "minute":
            df = self.get_minute(test_symbols, start, end)
        else:
            raise ValueError(f"Tushare 不支持数据集: {dataset}")
        return {
            "provider": self.name,
            "dataset": dataset,
            "rows": df.height,
            "columns": df.columns,
            "preview": df.head(5).to_dicts() if not df.is_empty() else [],
        }
