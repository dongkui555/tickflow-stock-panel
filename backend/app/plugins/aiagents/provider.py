from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import polars as pl

from app.data_providers.base import AssetType
from app.services import aiagents_bridge

_DATASETS = ("daily", "minute", "realtime")


@dataclass
class _Config:
    name: str = "aiagents"
    display_name: str = "AI Agents 多源降级"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def _to_date(value: datetime | None) -> str | None:
    return value.strftime("%Y%m%d") if value else None


def _daily_frame(frame, symbol: str) -> pl.DataFrame:
    if frame is None or frame.empty:
        return pl.DataFrame()
    keep = [c for c in ("date", "open", "high", "low", "close", "volume", "amount") if c in frame.columns]
    if "date" not in keep or "close" not in keep:
        return pl.DataFrame()
    data = frame[keep].copy()
    data["symbol"] = aiagents_bridge._tickflow_symbol(symbol)
    data["source"] = "aiagents"
    return pl.from_pandas(data, include_index=False).select(
        [c for c in ("symbol", "date", "open", "high", "low", "close", "volume", "amount", "source") if c in data.columns]
    )


def _minute_frame(frame, symbol: str) -> pl.DataFrame:
    if frame is None or frame.empty or "bar_time" not in frame.columns:
        return pl.DataFrame()
    data = frame.rename(columns={"bar_time": "datetime"}).copy()
    data["symbol"] = aiagents_bridge._tickflow_symbol(symbol)
    data["source"] = "aiagents"
    keep = [c for c in ("symbol", "datetime", "open", "high", "low", "close", "volume", "amount", "source") if c in data.columns]
    return pl.from_pandas(data[keep], include_index=False)


class AIAgentsProvider:
    name = "aiagents"
    builtin = True
    # 该适配器只支持逐标的实时查询，不能作为全市场快照源。
    realtime_scope = "symbols"

    def __init__(self):
        self.config = _Config()

    def close(self) -> None:
        pass

    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if asset_type != "stock":
            return pl.DataFrame()
        frames = []
        total = len(symbols)
        for index, symbol in enumerate(symbols, 1):
            frames.append(_daily_frame(
                aiagents_bridge.get_daily(symbol, _to_date(start_time), _to_date(end_time), days=500),
                symbol,
            ))
            if on_chunk_done:
                on_chunk_done(index, total)
        frames = [frame for frame in frames if not frame.is_empty()]
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def get_adj_factors(self, *args, **kwargs) -> pl.DataFrame:
        return pl.DataFrame()

    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        freq: str = "5m",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if asset_type != "stock":
            return pl.DataFrame()
        period = "".join(ch for ch in str(freq) if ch.isdigit()) or "5"
        frames = []
        total = len(symbols)
        for index, symbol in enumerate(symbols, 1):
            frames.append(_minute_frame(
                aiagents_bridge.get_minute(symbol, _to_date(start_time), _to_date(end_time), period),
                symbol,
            ))
            if on_chunk_done:
                on_chunk_done(index, total)
        frames = [frame for frame in frames if not frame.is_empty()]
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def get_realtime(self, universes=None, symbols=None) -> pl.DataFrame:
        rows = []
        for symbol in symbols or []:
            quote = aiagents_bridge._get_quote(symbol)
            if quote.get("data_success"):
                rows.append({
                    "symbol": aiagents_bridge._tickflow_symbol(symbol),
                    "name": quote.get("name", ""),
                    "last_price": quote.get("price"),
                    "prev_close": quote.get("pre_close"),
                    "open": quote.get("open"),
                    "high": quote.get("high"),
                    "low": quote.get("low"),
                    "volume": quote.get("volume"),
                    "amount": quote.get("amount"),
                    "change_pct": quote.get("change_percent"),
                    "source": quote.get("data_source", "aiagents"),
                })
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def get_instruments(self, asset_type: str = "stock") -> list[dict]:
        return []

    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        symbol = (symbols or ["000001.SZ"])[0]
        if dataset == "daily":
            df = self.get_daily([symbol], None, None)
        elif dataset == "minute":
            df = self.get_minute([symbol], None, None)
        elif dataset == "realtime":
            df = self.get_realtime(symbols=[symbol])
        else:
            raise ValueError(f"AI Agents 不支持数据集: {dataset}")
        return {
            "provider": self.name,
            "dataset": dataset,
            "rows": df.height,
            "columns": df.columns,
            "preview": df.head(5).to_dicts(),
        }
