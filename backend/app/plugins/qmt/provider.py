"""Normalized TickFlow provider for the local QMT HTTP bridge."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable

import polars as pl
import requests
from app.data_providers.base import AssetType
from app.plugins.qmt.bridge import available_datasets
from app.services.security_identity import exchange_for_code, explicit_exchange

logger = logging.getLogger(__name__)

_DATASETS = ("minute", "realtime", "depth5", "auction")
_MINUTE_COLUMNS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]
_DAILY_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]


@dataclass
class _Config:
    name: str = "qmt"
    display_name: str = "国信 QMT（内部策略实时/五档/分钟K）"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def _compact_symbol(symbol: str) -> str:
    return "".join(ch for ch in str(symbol or "") if ch.isdigit())[-6:].zfill(6)


def _qmt_symbol(symbol: str, asset_type: AssetType = "stock") -> str:
    raw = str(symbol or "").strip().upper()
    code = _compact_symbol(raw)
    declared = explicit_exchange(raw)
    market = exchange_for_code(code, asset_type) if asset_type == "stock" else declared
    if not market or (declared and declared != market):
        raise ValueError(f"无法确认 QMT 股票身份: {symbol}")
    return f"{code}.{market}"


def _number(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


class QmtProvider:
    name = "qmt"
    builtin = True
    realtime_scope = "symbols"

    def __init__(self) -> None:
        self.config = _Config()
        self.config.datasets = dict.fromkeys(available_datasets())
        self.base_url = os.getenv("QMT_BASE_URL", "http://host.docker.internal:8188").strip().rstrip("/")
        self.timeout = max(1.0, float(os.getenv("QMT_TIMEOUT_SECONDS", "12")))
        # This repository stores stock minute/depth volume in lots, matching QMT.
        self.volume_scale = max(1.0, float(os.getenv("QMT_VOLUME_SCALE", "1")))
        self._session = requests.Session()
        if api_key := os.getenv("QMT_BRIDGE_API_KEY", "").strip():
            self._session.headers["X-API-Key"] = api_key

    def close(self) -> None:
        self._session.close()

    def _request(self, path: str, params: dict[str, Any]) -> Any:
        response = self._session.get(
            f"{self.base_url}{path}",
            params={key: value for key, value in params.items() if value not in (None, "")},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("code") not in (None, 0, "0"):
            raise RuntimeError(str(payload.get("message") or "QMT bridge request failed"))
        return payload.get("data") if isinstance(payload, dict) and "data" in payload else payload

    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        freq: str = "1m",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if asset_type not in {"stock", "etf", "index"} or not symbols:
            return pl.DataFrame()
        normalized_freq = "60m" if str(freq).lower() in {"60m", "1h"} else str(freq).lower()
        def _fetch_one(symbol: str) -> list[dict[str, Any]]:
            qmt_symbol = _qmt_symbol(symbol, asset_type)
            try:
                items = self._request("/api/kline", {
                    "code": qmt_symbol,
                    "freq": normalized_freq,
                    "start": start_time.strftime("%Y%m%d%H%M%S") if start_time else "",
                    "end": end_time.strftime("%Y%m%d%H%M%S") if end_time else "",
                    "count": -1 if start_time or end_time else 500,
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("QMT %s %s K线请求失败: %s", qmt_symbol, normalized_freq, exc)
                items = []
            result: list[dict[str, Any]] = []
            for item in items or []:
                when = _parse_datetime(item.get("datetime"))
                close = _number(item.get("close"))
                if when is None or close is None or close <= 0:
                    continue
                if start_time and when < start_time or end_time and when > end_time:
                    continue
                result.append({
                    "symbol": qmt_symbol,
                    "datetime": when,
                    "open": _number(item.get("open")),
                    "high": _number(item.get("high")),
                    "low": _number(item.get("low")),
                    "close": close,
                    "volume": (_number(item.get("volume")) or 0.0) * self.volume_scale,
                    "amount": _number(item.get("amount")),
                })
            return result

        rows: list[dict[str, Any]] = []
        # QMT's local bridge serves one symbol per request.  Serial requests
        # make a 300+ stock Beijing repair take tens of minutes even though
        # the bridge itself is a threaded HTTP server.  Keep concurrency
        # bounded and configurable so the QMT client is not flooded.
        try:
            workers = int(os.getenv("QMT_MINUTE_WORKERS", "8"))
        except ValueError:
            workers = 8
        workers = max(1, min(workers, len(symbols), 16))
        if workers == 1:
            completed = 0
            for symbol in symbols:
                rows.extend(_fetch_one(symbol))
                completed += 1
                if on_chunk_done:
                    on_chunk_done(completed, len(symbols))
        else:
            completed = 0
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qmt-minute") as pool:
                futures = [pool.submit(_fetch_one, symbol) for symbol in symbols]
                for future in as_completed(futures):
                    rows.extend(future.result())
                    completed += 1
                    if on_chunk_done:
                        on_chunk_done(completed, len(symbols))
        if not rows:
            return pl.DataFrame()
        return (
            pl.DataFrame(rows)
            .with_columns(pl.col("datetime").cast(pl.Datetime))
            .unique(subset=["symbol", "datetime"], keep="last")
            .sort(["symbol", "datetime"])
            .select(_MINUTE_COLUMNS)
        )

    def get_realtime(self, universes=None, symbols=None) -> pl.DataFrame:
        normalized = [_qmt_symbol(symbol) for symbol in symbols or []]
        if not normalized:
            return pl.DataFrame()
        rows: list[dict[str, Any]] = []
        for offset in range(0, len(normalized), 50):
            batch = normalized[offset:offset + 50]
            try:
                items = self._request("/api/quote", {"symbols": ",".join(batch)})
            except Exception as exc:  # noqa: BLE001
                logger.warning("QMT 实时行情批次请求失败: %s", exc)
                continue
            for item in items or []:
                last_price = _number(item.get("lastPrice"))
                prev_close = _number(item.get("lastClose"))
                if last_price is None or last_price <= 0:
                    continue
                change_amount = last_price - prev_close if prev_close not in (None, 0) else None
                change_pct = change_amount / prev_close if change_amount is not None and prev_close else None
                when = _parse_datetime(item.get("datetime"))
                row = {
                    "symbol": _qmt_symbol(item.get("symbol") or ""),
                    "name": "",
                    "last_price": last_price,
                    "prev_close": prev_close,
                    "open": _number(item.get("open")),
                    "high": _number(item.get("high")),
                    "low": _number(item.get("low")),
                    "volume": (
                        _number(item.get("cumulativeVolume"))
                        or _number(item.get("pvolume"))
                        or _number(item.get("volume"))
                        or 0.0
                    ) * self.volume_scale,
                    "amount": _number(item.get("cumulativeAmount")) or _number(item.get("amount")),
                    "change_pct": change_pct,
                    "change_amount": change_amount,
                    "timestamp": int(when.timestamp() * 1000) if when else None,
                    "source": "qmt",
                    "depth_source": "qmt",
                }
                for side in ("bid", "ask"):
                    prices = list(item.get(f"{side}Price") or [])
                    volumes = list(item.get(f"{side}Vol") or [])
                    for level in range(1, 6):
                        row[f"{side}_price_{level}"] = _number(prices[level - 1]) if len(prices) >= level else None
                        volume = _number(volumes[level - 1]) if len(volumes) >= level else None
                        row[f"{side}_volume_{level}"] = volume * self.volume_scale if volume is not None else None
                rows.append(row)
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def get_ticks(self, symbol: str, start_time=None, end_time=None, count: int = 5000) -> list[dict[str, Any]]:
        return list(self._request("/api/tick", {
            "code": _qmt_symbol(symbol),
            "start": start_time.strftime("%Y%m%d%H%M%S") if start_time else "",
            "end": end_time.strftime("%Y%m%d%H%M%S") if end_time else "",
            "count": count,
        }) or [])

    def get_depth5(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Return the standard depth shape consumed by DepthService."""
        frame = self.get_realtime(symbols=symbols)
        result: dict[str, dict[str, Any]] = {}
        for row in frame.to_dicts():
            symbol = row.get("symbol")
            if not symbol:
                continue
            result[symbol] = {
                "symbol": symbol,
                "bid_prices": [row.get(f"bid_price_{level}") for level in range(1, 6)],
                "bid_volumes": [row.get(f"bid_volume_{level}") for level in range(1, 6)],
                "ask_prices": [row.get(f"ask_price_{level}") for level in range(1, 6)],
                "ask_volumes": [row.get(f"ask_volume_{level}") for level in range(1, 6)],
                "timestamp": row.get("timestamp"),
                "source": "qmt",
            }
        return result

    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        return self.get_asset_daily(
            symbols, start_time, end_time, "stock", on_chunk_done=on_chunk_done,
        )

    def get_asset_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType,
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        """Read daily bars captured by the authenticated in-client collector."""
        if asset_type not in {"stock", "etf", "index"} or not symbols:
            return pl.DataFrame()
        minute = self.get_minute(
            symbols, start_time, end_time, asset_type=asset_type, freq="1d",
            on_chunk_done=on_chunk_done,
        )
        if minute.is_empty():
            return pl.DataFrame()
        return (
            minute.with_columns(pl.col("datetime").dt.date().alias("date"))
            .select(_DAILY_COLUMNS)
            .unique(subset=["symbol", "date"], keep="last")
            .sort(["symbol", "date"])
        )

    def get_auction(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        market: str = "o",
        on_chunk_done: Callable[[int, int], None] | None = None,
        trade_dates: list[str | date | datetime] | None = None,
        strict: bool = False,  # noqa: ARG002
    ) -> pl.DataFrame:
        if market != "o" or not symbols:
            return pl.DataFrame()
        dates = trade_dates or [end_time or start_time or datetime.now()]
        rows: list[dict[str, Any]] = []
        normalized = [_qmt_symbol(symbol) for symbol in symbols]
        for index, value in enumerate(dates, start=1):
            if isinstance(value, datetime):
                day = value.date().isoformat()
            elif isinstance(value, date):
                day = value.isoformat()
            else:
                text = str(value)
                day = f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) >= 8 and "-" not in text else text[:10]
            try:
                params = {"date": day}
                # An all-market symbol list exceeds common HTTP request-line
                # limits. The bridge can return the entire date partition.
                if len(normalized) <= 100:
                    params["symbols"] = ",".join(normalized)
                items = self._request("/api/auction", params)
                rows.extend(items or [])
            except Exception as exc:  # noqa: BLE001
                logger.warning("QMT 集合竞价缓存请求失败 %s: %s", day, exc)
            if on_chunk_done:
                on_chunk_done(index, len(dates))
        if not rows:
            return pl.DataFrame()
        frame = (
            pl.DataFrame(rows)
            .with_columns(
                pl.col("symbol").cast(pl.String).str.split(".").list.first(),
                pl.col("date").cast(pl.Date, strict=False),
            )
            .unique(subset=["symbol", "date"], keep="last")
            .sort(["symbol", "date"])
        )
        requested = {_compact_symbol(symbol) for symbol in symbols}
        return frame.filter(pl.col("symbol").is_in(sorted(requested)))

    def get_adj_factors(self, *args, **kwargs) -> pl.DataFrame:
        return pl.DataFrame()

    def get_instruments(self, asset_type: AssetType = "stock") -> list[dict]:
        return []

    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        test_symbols = symbols or ["000001.SZ"]
        if dataset == "minute":
            frame = self.get_minute(test_symbols, None, datetime.now(), freq="1m")
        elif dataset == "realtime":
            frame = self.get_realtime(symbols=test_symbols)
        elif dataset == "depth5":
            rows = list(self.get_depth5(test_symbols).values())
            return {
                "provider": self.name,
                "dataset": dataset,
                "rows": len(rows),
                "columns": list(rows[0]) if rows else [],
                "preview": rows[:5],
            }
        elif dataset == "auction":
            frame = self.get_auction(test_symbols, datetime.now(), datetime.now())
        else:
            raise ValueError(f"QMT 不支持数据集: {dataset}")
        return {
            "provider": self.name,
            "dataset": dataset,
            "rows": frame.height,
            "columns": frame.columns,
            "preview": frame.head(5).to_dicts(),
        }
