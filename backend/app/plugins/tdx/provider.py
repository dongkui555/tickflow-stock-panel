"""旧 aiagents-stock TDX HTTP API 的 TickFlow provider 适配。"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable

import polars as pl
import requests

from app.data_providers.base import AssetType
from app.services.security_identity import exchange_for_code, explicit_exchange

logger = logging.getLogger(__name__)

_DATASETS = ("daily", "minute", "realtime")
_DAILY_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
_MINUTE_COLUMNS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]


@dataclass
class _Config:
    name: str = "tdx"
    display_name: str = "本地 TDX（实时/分钟K）"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def _compact_symbol(symbol: str) -> str:
    return "".join(ch for ch in str(symbol or "") if ch.isdigit())[-6:].zfill(6)


def _tickflow_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    code = _compact_symbol(raw)
    market = exchange_for_code(code, "stock")
    declared = explicit_exchange(raw)
    if not market or (declared and declared != market):
        raise ValueError(f"无法确认 TDX 股票身份: {symbol}")
    return f"{code}.{market}"


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1000
        try:
            return datetime.fromtimestamp(seconds)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d%H%M%S",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _tdx_wallclock(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    # TDX 返回的是本地交易时间的 naive datetime；比较窗口也按同一墙钟口径处理。
    return value.replace(tzinfo=None)


class TdxProvider:
    """把 TDX 的厘/手单位归一为 TickFlow 的元/手口径。"""

    name = "tdx"
    builtin = True
    # TDX 旧接口仅支持逐标的报价；QuoteService 据此改走自选股/监控请求。
    realtime_scope = "symbols"

    def __init__(self) -> None:
        self.config = _Config()
        self.base_url = os.getenv("TDX_BASE_URL", "").strip().rstrip("/")
        self.timeout = max(1.0, float(os.getenv("TDX_TIMEOUT_SECONDS", "5")))
        self.price_scale = max(1.0, float(os.getenv("TDX_PRICE_SCALE", "1000")))
        self.amount_scale = max(1.0, float(os.getenv("TDX_AMOUNT_SCALE", "1000")))
        # 旧接口成交量为手；系统内部日/分钟K也统一按手存储。
        self.volume_scale = max(1.0, float(os.getenv("TDX_VOLUME_SCALE", "1")))
        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    def _request(self, path: str, params: dict[str, Any]) -> Any:
        if not self.base_url:
            return None
        response = self._session.get(
            f"{self.base_url}{path}",
            params={key: value for key, value in params.items() if value is not None},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("code") not in (None, 0, "0"):
            raise RuntimeError(str(payload.get("message") or payload.get("msg") or "TDX 接口返回失败"))
        return payload.get("data") if isinstance(payload, dict) and "data" in payload else payload

    @staticmethod
    def _candidate_codes(symbol: str) -> list[str]:
        code = _compact_symbol(symbol)
        full = _tickflow_symbol(symbol)
        suffix = full.rsplit(".", 1)[-1]
        candidates = [code, f"{suffix}{code}"]
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _list_payload(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("List", "list", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    def _fetch_kline(self, symbol: str, kline_type: str, limit: int | None = None) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for code in self._candidate_codes(symbol):
            try:
                params = {"code": code, "type": kline_type}
                if limit is not None:
                    params["limit"] = limit
                rows = self._list_payload(self._request("/api/kline", params))
                if rows:
                    return rows
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error:
            if kline_type != "day":
                raise last_error
            logger.debug("TDX %s K线请求失败: %s", symbol, type(last_error).__name__)
        return []

    def _price(self, value: Any) -> float | None:
        raw = _number(value)
        return raw / self.price_scale if raw is not None else None

    def _amount(self, value: Any) -> float | None:
        raw = _number(value)
        return raw / self.amount_scale if raw is not None else None

    def _volume(self, value: Any) -> float | None:
        raw = _number(value)
        return raw * self.volume_scale if raw is not None else None

    def _kline_rows(
        self,
        symbol: str,
        kline_type: str,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> list[dict[str, Any]]:
        start_time = _tdx_wallclock(start_time)
        end_time = _tdx_wallclock(end_time)
        rows: list[dict[str, Any]] = []
        limit = None
        if kline_type == "minute1" and start_time is not None:
            from app.market_time import cn_today
            # The local bridge pages backwards from today, not from end_time.
            days = max(1, (cn_today() - start_time.date()).days + 1)
            limit = max(800, min(24000, days * 240))
        parsed = [(_parse_datetime(
                item.get("Time") or item.get("time") or item.get("datetime") or item.get("Date") or item.get("date")
            ), item) for item in self._fetch_kline(symbol, kline_type, limit=limit)]
        lunch_labels = set()
        if kline_type == "minute1":
            stamps = {when for when, _ in parsed if when is not None}
            for when in stamps:
                if (when.hour, when.minute, when.second, when.microsecond) != (13, 0, 0, 0):
                    continue
                first = when.replace(hour=9, minute=31)
                # This TDX session-end convention labels the last morning bar 13:00.
                # Require the complete preceding morning and no alternative close label.
                if (when.replace(hour=11, minute=30) not in stamps
                        and when.replace(hour=9, minute=30) not in stamps
                        and all(first + timedelta(minutes=i) in stamps for i in range(119))):
                    lunch_labels.add(when)
        for when, item in parsed:
            if when in lunch_labels:
                when = when.replace(hour=11, minute=30)
            if when is None or (start_time and when < start_time) or (end_time and when > end_time):
                continue
            close = self._price(item.get("Close") if "Close" in item else item.get("close"))
            if close is None:
                continue
            rows.append({
                "symbol": _tickflow_symbol(symbol),
                "datetime": when,
                "open": self._price(item.get("Open") if "Open" in item else item.get("open")),
                "high": self._price(item.get("High") if "High" in item else item.get("high")),
                "low": self._price(item.get("Low") if "Low" in item else item.get("low")),
                "close": close,
                "volume": self._volume(item.get("Volume") if "Volume" in item else item.get("volume")),
                "amount": self._amount(item.get("Amount") if "Amount" in item else item.get("amount")),
            })
        return rows

    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if asset_type != "stock" or not symbols:
            return pl.DataFrame()
        rows: list[dict[str, Any]] = []
        for index, symbol in enumerate(symbols, start=1):
            for row in self._kline_rows(symbol, "day", start_time, end_time):
                rows.append({key: value for key, value in row.items() if key != "datetime"} | {"date": row["datetime"].date()})
            if on_chunk_done:
                on_chunk_done(index, len(symbols))
        if not rows:
            return pl.DataFrame()
        return (
            pl.DataFrame(rows)
            .with_columns(pl.col("date").cast(pl.Date))
            .unique(subset=["symbol", "date"], keep="last")
            .sort(["symbol", "date"])
            .select(_DAILY_COLUMNS)
        )

    def get_adj_factors(self, *args, **kwargs) -> pl.DataFrame:
        return pl.DataFrame()

    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        freq: str = "1m",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if asset_type != "stock" or not symbols:
            return pl.DataFrame()
        freq_map = {
            "1m": "minute1",
            "5m": "minute5",
            "15m": "minute15",
            "30m": "minute30",
            "60m": "hour",
        }
        kline_type = freq_map.get(str(freq).lower(), "minute1")
        rows: list[dict[str, Any]] = []
        for index, symbol in enumerate(symbols, start=1):
            rows.extend(self._kline_rows(symbol, kline_type, start_time, end_time))
            if on_chunk_done:
                on_chunk_done(index, len(symbols))
        if not rows:
            return pl.DataFrame()
        return (
            pl.DataFrame(rows)
            .with_columns(pl.col("datetime").cast(pl.Datetime))
            .unique(subset=["symbol", "datetime"], keep="last")
            .sort(["symbol", "datetime"])
            .select(_MINUTE_COLUMNS)
        )

    def get_minute_diagnostics(self, symbols, start_time, end_time, asset_type="stock", freq="1m"):
        from app.plugins.tushare.provider import minute_error_details

        frames, issues = [], []
        for symbol in symbols:
            context = {"symbol": _compact_symbol(symbol)}
            try:
                frame = self.get_minute([symbol], start_time, end_time, asset_type=asset_type, freq=freq)
                if frame.is_empty():
                    issues.append({**context, "reason": "source_empty", "retryable": False})
                else:
                    frames.append(frame)
            except Exception as exc:
                issues.append({**context, **minute_error_details(exc)})
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(), issues

    def probe_minute(self, start_time, end_time):
        """Check usable bars, not merely the bridge's HTTP/TCP health."""
        from app.services.minute_failover import _intraday_reason

        end = _tdx_wallclock(end_time)
        start = max(_tdx_wallclock(start_time), end.replace(hour=9, minute=30, second=0, microsecond=0))
        reason = "source_empty"
        detail = {}
        for symbol in ("000001.SZ", "600000.SH"):
            frame, issues = self.get_minute_diagnostics([symbol], start, end)
            if frame.is_empty():
                detail = issues[-1] if issues else {}
                reason = detail.get("reason", "source_empty")
                continue
            frame = frame.filter(pl.col("datetime").is_between(start, end))
            reason = _intraday_reason(frame, end.date(), start, end)
            if reason == "accepted":
                return {"usable": True, "reason": "accepted", "rows": frame.height,
                        "last_time": str(frame["datetime"].max())}
        return {**detail, "usable": False, "reason": reason, "retryable": True}

    def _quote_for_symbol(self, symbol: str) -> dict[str, Any] | None:
        for code in self._candidate_codes(symbol):
            try:
                items = self._list_payload(self._request("/api/quote", {"code": code}))
            except Exception as exc:  # noqa: BLE001
                logger.debug("TDX %s 实时行情请求失败: %s", symbol, exc)
                continue
            if not items:
                continue
            record = self._quote_payload_to_record(items[0], fallback_symbol=symbol)
            if record is not None:
                return record
        return None

    @staticmethod
    def _symbol_from_quote(quote: dict[str, Any], fallback_symbol: str | None = None) -> str:
        code = _compact_symbol(quote.get("Code") or quote.get("code") or fallback_symbol or "")
        market = quote.get("Market") if "Market" in quote else quote.get("market")
        if market is not None:
            try:
                market_int = int(market)
            except (TypeError, ValueError):
                market_int = None
            if market_int == 1:
                return f"{code}.SH"
            if market_int == 0:
                market = exchange_for_code(code, "stock")
                if market not in {"BJ", "SZ"}:
                    raise ValueError(f"TDX 返回未知证券市场: {code}")
                return f"{code}.{market}"
        return _tickflow_symbol(fallback_symbol or code)

    def _quote_payload_to_record(
        self,
        quote: dict[str, Any],
        fallback_symbol: str | None = None,
    ) -> dict[str, Any] | None:
        kline = quote.get("K") if isinstance(quote.get("K"), dict) else quote
        last_price = self._price(kline.get("Close") if "Close" in kline else kline.get("close"))
        prev_close = self._price(kline.get("Last") if "Last" in kline else kline.get("prev_close"))
        if last_price is None:
            return None
        change_amount = last_price - prev_close if prev_close not in (None, 0) else None
        change_pct = change_amount / prev_close if prev_close not in (None, 0) else None
        server_time = _parse_datetime(quote.get("ServerTime") or quote.get("timestamp"))
        timestamp = int(server_time.timestamp() * 1000) if server_time else None
        return {
            "symbol": self._symbol_from_quote(quote, fallback_symbol=fallback_symbol),
            "name": str(quote.get("Name") or quote.get("name") or ""),
            "last_price": last_price,
            "prev_close": prev_close,
            "open": self._price(kline.get("Open") if "Open" in kline else kline.get("open")),
            "high": self._price(kline.get("High") if "High" in kline else kline.get("high")),
            "low": self._price(kline.get("Low") if "Low" in kline else kline.get("low")),
            "volume": self._volume(quote.get("TotalHand") if "TotalHand" in quote else quote.get("volume")),
            "amount": self._amount(quote.get("Amount") if "Amount" in quote else quote.get("amount")),
            "change_pct": change_pct,
            "change_amount": change_amount,
            "timestamp": timestamp,
            "source": "tdx",
        }

    def get_realtime(self, universes=None, symbols=None) -> pl.DataFrame:
        # 旧接口没有按 universe 拉取全市场的能力，只有明确标的时才请求。
        requested = [str(symbol).strip() for symbol in symbols or [] if str(symbol or "").strip()]
        if not requested:
            return pl.DataFrame()

        rows: list[dict[str, Any]] = []
        try:
            items = self._list_payload(self._request("/api/quotes", {"codes": ",".join(requested)}))
            for item in items:
                record = self._quote_payload_to_record(item)
                if record is not None:
                    rows.append(record)
        except Exception as exc:  # noqa: BLE001
            logger.debug("TDX 批量实时行情请求失败, 回退逐标的: %s", exc)

        if not rows:
            rows = [row for symbol in requested if (row := self._quote_for_symbol(symbol)) is not None]
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def get_instruments(self, asset_type: AssetType = "stock") -> list[dict]:
        return []

    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        test_symbols = symbols or ["000001.SZ"]
        end = datetime.now()
        if dataset == "daily":
            frame = self.get_daily(test_symbols, None, end)
        elif dataset == "minute":
            frame = self.get_minute(test_symbols, None, end)
        elif dataset == "realtime":
            frame = self.get_realtime(symbols=test_symbols)
        else:
            raise ValueError(f"TDX 不支持数据集: {dataset}")
        return {
            "provider": self.name,
            "dataset": dataset,
            "rows": frame.height,
            "columns": frame.columns,
            "preview": frame.head(5).to_dicts(),
        }
