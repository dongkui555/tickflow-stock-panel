from __future__ import annotations

import json
import math
import logging
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from threading import Lock, RLock
from urllib.parse import quote_plus

import pandas as pd
import polars as pl
import requests

from app.config import settings

logger = logging.getLogger(__name__)

_RUNTIME_LOCK = RLock()
_RUNTIME_REPO: Any = None
_RUNTIME_QUOTE_SERVICE: Any = None


def configure_runtime(*, repo: Any | None = None, quote_service: Any | None = None) -> None:
    global _RUNTIME_REPO, _RUNTIME_QUOTE_SERVICE
    if repo is not None:
        _RUNTIME_REPO = repo
    if quote_service is not None:
        _RUNTIME_QUOTE_SERVICE = quote_service


def data_root() -> Path:
    return Path(settings.data_dir)


def legacy_root() -> Path:
    return data_root()


def _repo() -> Any:
    return _RUNTIME_REPO


def _quote_service() -> Any:
    return _RUNTIME_QUOTE_SERVICE


def _optional_import(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None


def _compact_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if "." in value:
        value = value.split(".", 1)[0]
    return value.zfill(6) if value.isdigit() else value


def _tickflow_symbol(symbol: str) -> str:
    code = _compact_symbol(symbol)
    if not code.isdigit():
        return code
    from app.services.security_identity import exchange_for_code
    market = exchange_for_code(code, "stock")
    return f"{code}.{market}" if market else code


def _is_chinese_stock(symbol: str) -> bool:
    return bool(symbol) and _compact_symbol(symbol).isdigit() and len(_compact_symbol(symbol)) == 6


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
        return result if pd.notna(result) and math.isfinite(result) else default
    except Exception:
        return default


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _parse_date(value: str | None, fallback: date | None = None) -> date:
    if not value:
        return fallback or date.today()
    text = str(value).strip().replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return fallback or date.today()


def _standard_daily(frame: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    aliases = {
        "日期": "date",
        "交易日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover_rate",
        "涨跌幅": "change_pct",
        "昨收": "prev_close",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    if "date" not in df.columns:
        for candidate in ("trade_date", "datetime", "bar_time"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "date"})
                break
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "amount", "turnover_rate", "change_pct", "prev_close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = _tickflow_symbol(symbol)
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def _frame_records(frame: pd.DataFrame | None, limit: int = 200) -> list[dict]:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    rows = frame.tail(limit).copy()
    return [_json_value(row) for row in rows.to_dict(orient="records")]


def _repo_to_pandas(df: Any) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    try:
        return pd.DataFrame(df)
    except Exception:
        return pd.DataFrame()


class LocalDataSourceManager:
    """TickFlow 内部行情/基本面/分钟线入口，优先本地仓库，失败再走公开接口。"""

    def __init__(self) -> None:
        self.tushare_token = os.getenv("TUSHARE_TOKEN", "").strip() or settings.tushare_token.strip()
        self.tushare_available = False
        self.tushare_api = None
        self.init_error = ""
        if self.tushare_token:
            try:
                import tushare as ts

                ts.set_token(self.tushare_token)
                self.tushare_api = ts.pro_api()
                self.tushare_available = True
            except Exception as exc:
                self.init_error = str(exc)
                logger.warning("Tushare 初始化失败: %s", exc)

    def _repo_daily(self, symbol: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
        repo = _repo()
        if repo is None:
            return pd.DataFrame()
        try:
            start = _parse_date(start_date, date.today() - timedelta(days=365))
            end = _parse_date(end_date, date.today())
            frame = repo.get_daily_asset(
                "stock",
                _tickflow_symbol(symbol),
                start,
                end,
                columns=["date", "open", "high", "low", "close", "volume", "amount", "turnover_rate", "change_pct", "prev_close"],
            )
            if frame is None or getattr(frame, "is_empty", lambda: True)():
                return pd.DataFrame()
            pdf = _repo_to_pandas(frame)
            return _standard_daily(pdf, symbol)
        except Exception as exc:
            logger.debug("repo 日线获取失败(%s): %s", symbol, exc)
            return pd.DataFrame()

    def _repo_minute(self, symbol: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
        repo = _repo()
        if repo is None:
            return pd.DataFrame()
        try:
            start = _parse_date(start_date, date.today() - timedelta(days=5))
            end = _parse_date(end_date, date.today())
            frame = repo.get_minute_range([_tickflow_symbol(symbol)], start, end, asset_type="stock")
            if frame is None or getattr(frame, "is_empty", lambda: True)():
                return pd.DataFrame()
            pdf = _repo_to_pandas(frame)
            if "datetime" in pdf.columns and "bar_time" not in pdf.columns:
                pdf = pdf.rename(columns={"datetime": "bar_time"})
            pdf["symbol"] = _tickflow_symbol(symbol)
            return pdf.sort_values("bar_time").reset_index(drop=True)
        except Exception as exc:
            logger.debug("repo 分钟线获取失败(%s): %s", symbol, exc)
            return pd.DataFrame()

    def _repo_quote(self, symbol: str) -> dict:
        code = _tickflow_symbol(symbol)
        quote_service = _quote_service()
        if quote_service is not None:
            try:
                frame = quote_service.get_quotes_compat()
                if frame is not None and not frame.is_empty() and "symbol" in frame.columns:
                    row = frame.filter(pl.col("symbol") == code).head(1)
                    if not row.is_empty():
                        return row.to_dicts()[0]
            except Exception as exc:
                logger.debug("quote_service 实时报价失败(%s): %s", symbol, exc)
        repo = _repo()
        if repo is not None:
            try:
                frame, _ = repo.get_enriched_latest_asset("stock", refresh=False)
                if frame is not None and not frame.is_empty() and "symbol" in frame.columns:
                    row = frame.filter(pl.col("symbol") == code).head(1)
                    if not row.is_empty():
                        return row.to_dicts()[0]
            except Exception as exc:
                logger.debug("repo 最新报价失败(%s): %s", symbol, exc)
        return {}

    def get_stock_hist_data(self, symbol: str, start_date: str | None = None, end_date: str | None = None, adjust: str = "qfq") -> pd.DataFrame:
        repo_df = self._repo_daily(symbol, start_date, end_date)
        if not repo_df.empty:
            return repo_df

        ak = _optional_import("akshare")
        if ak is not None:
            try:
                frame = ak.stock_zh_a_hist(
                    symbol=_compact_symbol(symbol),
                    period="daily",
                    start_date=(start_date or "").replace("-", ""),
                    end_date=(end_date or datetime.now().strftime("%Y%m%d")).replace("-", ""),
                    adjust=adjust,
                )
                return _standard_daily(frame, symbol)
            except Exception as exc:
                logger.debug("akshare 日线获取失败(%s): %s", symbol, exc)

        if self.tushare_available:
            try:
                import tushare as ts

                ts_code = self._convert_to_ts_code(symbol)
                frame = ts.pro_bar(
                    ts_code=ts_code,
                    start_date=(start_date or "").replace("-", ""),
                    end_date=(end_date or datetime.now().strftime("%Y%m%d")).replace("-", ""),
                    adj=adjust or None,
                )
                if frame is not None and not frame.empty:
                    frame = frame.rename(columns={"trade_date": "date", "vol": "volume"})
                    return _standard_daily(frame, symbol)
            except Exception as exc:
                logger.debug("tushare 日线获取失败(%s): %s", symbol, exc)

        return pd.DataFrame()

    def get_stock_minute_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str = "5",
        adjust: str = "",
    ) -> pd.DataFrame:
        repo_df = self._repo_minute(symbol, start_date, end_date)
        if not repo_df.empty:
            return repo_df

        ak = _optional_import("akshare")
        if ak is None:
            return pd.DataFrame()
        start = (start_date or (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")).replace("-", "/")[:10]
        end = (end_date or datetime.now().strftime("%Y-%m-%d")).replace("-", "/")[:10]
        try:
            frame = ak.stock_zh_a_hist_min_em(
                symbol=_compact_symbol(symbol),
                start_date=f"{start} 09:25:00",
                end_date=f"{end} 15:05:00",
                period=str(period),
                adjust=adjust,
            )
            source = "akshare_eastmoney_minute"
        except Exception:
            try:
                frame = ak.stock_zh_a_minute(
                    symbol=self._convert_to_tx_code(symbol),
                    period=str(period),
                    adjust=adjust,
                )
                source = "akshare_sina_minute"
            except Exception as exc:
                logger.debug("akshare 分钟线获取失败(%s): %s", symbol, exc)
                return pd.DataFrame()
        if frame is None or frame.empty:
            return pd.DataFrame()
        rename = {
            "时间": "bar_time",
            "日期": "bar_time",
            "day": "bar_time",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover_rate",
        }
        frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns})
        if "bar_time" not in frame.columns:
            return pd.DataFrame()
        frame["bar_time"] = pd.to_datetime(frame["bar_time"], errors="coerce")
        for column in ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["bar_time", "close"]).sort_values("bar_time")
        frame["symbol"] = _compact_symbol(symbol)
        frame["data_source"] = source
        frame["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return frame.reset_index(drop=True)

    def get_stock_turnover_history(self, symbol: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame | None:
        repo_df = self._repo_daily(symbol, start_date, end_date)
        if not repo_df.empty and "turnover_rate" in repo_df.columns:
            frame = repo_df[["date", "turnover_rate"]].copy()
            frame["turnover_source"] = "tickflow_repo"
            frame = frame.rename(columns={"date": "trade_date"})
            return frame

        start = (start_date or "").replace("-", "")
        end = (end_date or datetime.now().strftime("%Y%m%d")).replace("-", "")
        if self.tushare_available:
            try:
                frame = self.tushare_api.daily_basic(
                    ts_code=self._convert_to_ts_code(symbol),
                    start_date=start,
                    end_date=end,
                    fields="ts_code,trade_date,turnover_rate,volume_ratio",
                )
                if frame is not None and not frame.empty:
                    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
                    frame["turnover_rate"] = pd.to_numeric(frame["turnover_rate"], errors="coerce")
                    frame["turnover_source"] = "tushare_daily_basic"
                    return frame[["trade_date", "turnover_rate", "turnover_source"]]
            except Exception as exc:
                logger.debug("tushare 换手率失败(%s): %s", symbol, exc)

        ak = _optional_import("akshare")
        if ak is not None:
            try:
                frame = ak.stock_zh_a_hist(
                    symbol=_compact_symbol(symbol),
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust="",
                )
                if frame is not None and not frame.empty and "换手率" in frame.columns:
                    frame = frame.rename(columns={"日期": "trade_date", "换手率": "turnover_rate"})
                    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
                    frame["turnover_rate"] = pd.to_numeric(frame["turnover_rate"], errors="coerce")
                    frame["turnover_source"] = "akshare_eastmoney_turnover"
                    return frame[["trade_date", "turnover_rate", "turnover_source"]]
            except Exception as exc:
                logger.debug("akshare 换手率失败(%s): %s", symbol, exc)
        return None

    def get_stock_basic_info(self, symbol: str) -> dict:
        info = {
            "symbol": _compact_symbol(symbol),
            "name": "未知",
            "industry": "未知",
            "market": "未知",
        }

        repo = _repo()
        if repo is not None:
            try:
                inst = repo.get_instruments_asset("stock")
                if inst is not None and not inst.is_empty() and "symbol" in inst.columns:
                    row = inst.filter(pl.col("symbol") == _tickflow_symbol(symbol)).head(1)
                    if not row.is_empty():
                        record = row.to_dicts()[0]
                        info["name"] = record.get("name") or info["name"]
                        info["industry"] = record.get("industry") or info["industry"]
                        info["market"] = record.get("exchange") or info["market"]
                        return info
            except Exception as exc:
                logger.debug("repo 基本信息失败(%s): %s", symbol, exc)

        ak = _optional_import("akshare")
        if ak is not None:
            try:
                stock_info = ak.stock_individual_info_em(symbol=_compact_symbol(symbol))
                if stock_info is not None and not stock_info.empty:
                    for _, row in stock_info.iterrows():
                        key = str(row.get("item", ""))
                        value = row.get("value")
                        if key == "股票简称":
                            info["name"] = value
                        elif key == "所处行业":
                            info["industry"] = value
                        elif key == "上市时间":
                            info["list_date"] = value
                        elif key == "总市值":
                            info["market_cap"] = value
                        elif key == "流通市值":
                            info["circulating_market_cap"] = value
                    return info
            except Exception as exc:
                logger.debug("akshare 基本信息失败(%s): %s", symbol, exc)

        try:
            tx_code = self._convert_to_tx_code(symbol)
            r = requests.get(
                f"https://hq.sinajs.cn/list={tx_code}",
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": "https://finance.sina.com.cn/",
                },
                timeout=10,
            )
            if r.status_code == 200 and f"hq_str_{tx_code}" in r.text:
                start = r.text.find('"') + 1
                end = r.text.find('"', start)
                if start > 0 and end > start:
                    fields = r.text[start:end].split(",")
                    if fields and fields[0]:
                        info["name"] = fields[0]
                        info["market"] = "中国A股"
                        return info
        except Exception as exc:
            logger.debug("新浪个股基础信息失败(%s): %s", symbol, exc)

        if self.tushare_available:
            try:
                frame = self.tushare_api.stock_basic(
                    ts_code=self._convert_to_ts_code(symbol),
                    fields="ts_code,name,area,industry,market,list_date",
                )
                if frame is not None and not frame.empty:
                    row = frame.iloc[0]
                    info["name"] = row.get("name", info["name"])
                    info["industry"] = row.get("industry", info["industry"])
                    info["market"] = row.get("market", info["market"])
                    info["list_date"] = row.get("list_date")
                    return info
            except Exception as exc:
                logger.debug("tushare 基本信息失败(%s): %s", symbol, exc)

        return info

    def get_realtime_quotes(self, symbol: str) -> dict:
        code = _compact_symbol(symbol)
        quote = self._repo_quote(code)
        if quote:
            return self._normalize_quote(quote, code, "tickflow_repo")

        try:
            r = requests.get(
                f"https://hq.sinajs.cn/list={self._convert_to_tx_code(code)}",
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": "https://finance.sina.com.cn/",
                },
                timeout=10,
            )
            tx_code = self._convert_to_tx_code(code)
            if r.status_code == 200 and f"hq_str_{tx_code}" in r.text:
                start = r.text.find('"') + 1
                end = r.text.find('"', start)
                if start > 0 and end > start:
                    fields = r.text[start:end].split(",")
                    if len(fields) >= 32:
                        result = {
                            "symbol": code,
                            "name": fields[0] if fields[0] else "",
                            "open": _safe_float(fields[1], 0) or 0,
                            "pre_close": _safe_float(fields[2], 0) or 0,
                            "price": _safe_float(fields[3], 0) or 0,
                            "high": _safe_float(fields[4], 0) or 0,
                            "low": _safe_float(fields[5], 0) or 0,
                            "volume": _safe_float(fields[8], 0) or 0,
                            "amount": _safe_float(fields[9], 0) or 0,
                            "trade_date": fields[30] if len(fields) > 30 else "",
                            "trade_time": fields[31] if len(fields) > 31 else "",
                            "data_source": "sina_realtime",
                        }
                        if result["pre_close"] > 0:
                            result["change"] = round(result["price"] - result["pre_close"], 3)
                            result["change_percent"] = round((result["change"] / result["pre_close"]) * 100, 2)
                        result["data_success"] = bool(result["price"])
                        return result
        except Exception as exc:
            logger.debug("新浪实时行情失败(%s): %s", symbol, exc)

        if self.tushare_available:
            try:
                frame = self.tushare_api.daily(
                    ts_code=self._convert_to_ts_code(code),
                    start_date=datetime.now().strftime("%Y%m%d"),
                    end_date=datetime.now().strftime("%Y%m%d"),
                )
                if frame is not None and not frame.empty:
                    row = frame.iloc[0]
                    price = _safe_float(row.get("close"), 0) or 0
                    pre_close = _safe_float(row.get("pre_close"), price) or price
                    change = price - pre_close
                    result = {
                        "symbol": code,
                        "price": price,
                        "change_percent": _safe_float(row.get("pct_chg"), 0) or 0,
                        "volume": (_safe_float(row.get("vol"), 0) or 0) * 100,
                        "amount": (_safe_float(row.get("amount"), 0) or 0) * 1000,
                        "high": _safe_float(row.get("high"), price) or price,
                        "low": _safe_float(row.get("low"), price) or price,
                        "open": _safe_float(row.get("open"), price) or price,
                        "pre_close": pre_close,
                        "change": change,
                        "data_source": "tushare_daily",
                        "data_success": True,
                    }
                    return result
            except Exception as exc:
                logger.debug("tushare 实时行情失败(%s): %s", symbol, exc)

        daily = self.get_stock_hist_data(code, days_ago(30), None)
        if not daily.empty:
            last = daily.iloc[-1]
            price = _safe_float(last.get("close"), 0) or 0
            pre_close = _safe_float(last.get("prev_close"), price) or price
            result = {
                "symbol": code,
                "name": self.get_stock_basic_info(code).get("name", ""),
                "open": _safe_float(last.get("open"), price) or price,
                "pre_close": pre_close,
                "price": price,
                "high": _safe_float(last.get("high"), price) or price,
                "low": _safe_float(last.get("low"), price) or price,
                "volume": _safe_float(last.get("volume"), 0) or 0,
                "amount": _safe_float(last.get("amount"), 0) or 0,
                "data_source": "tickflow_daily_fallback",
                "data_success": True,
            }
            if pre_close > 0:
                result["change"] = round(price - pre_close, 3)
                result["change_percent"] = round(((price - pre_close) / pre_close) * 100, 2)
            return result
        return {"symbol": code, "data_success": False, "data_source": "unavailable"}

    def _normalize_quote(self, row: dict, symbol: str, source: str) -> dict:
        price = _safe_float(row.get("price") or row.get("last_price") or row.get("close"), 0) or 0
        pre_close = _safe_float(row.get("pre_close") or row.get("prev_close"), price) or price
        result = {
            "symbol": symbol,
            "name": row.get("name", ""),
            "open": _safe_float(row.get("open"), price) or price,
            "pre_close": pre_close,
            "price": price,
            "high": _safe_float(row.get("high"), price) or price,
            "low": _safe_float(row.get("low"), price) or price,
            "volume": _safe_float(row.get("volume"), 0) or 0,
            "amount": _safe_float(row.get("amount"), 0) or 0,
            "trade_date": row.get("trade_date") or row.get("date") or "",
            "trade_time": row.get("trade_time") or "",
            "data_source": source,
            "data_success": bool(price),
        }
        change = _safe_float(row.get("change"), None)
        change_pct = _safe_float(row.get("change_percent"), None)
        if change is None and price and pre_close:
            change = price - pre_close
        if change_pct is None and price and pre_close:
            change_pct = round((change / pre_close) * 100, 2) if pre_close else None
        if change is not None:
            result["change"] = change
        if change_pct is not None:
            result["change_percent"] = change_pct
        return result

    @staticmethod
    def _convert_to_ts_code(symbol: str) -> str:
        code = _compact_symbol(symbol)
        from app.services.security_identity import exchange_for_code
        market = exchange_for_code(code, "stock")
        return f"{code}.{market}" if market else ""

    @staticmethod
    def _convert_to_tx_code(symbol: str) -> str:
        code = _compact_symbol(symbol)
        from app.services.security_identity import exchange_for_code
        market = exchange_for_code(code, "stock")
        return f"{market.lower()}{code}" if market else ""


def days_ago(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


class FundFlowAkshareDataFetcher:
    """个股主力/大单资金流向，多源兜底。"""

    def __init__(self) -> None:
        self.days = int(os.getenv("FUND_FLOW_DAYS", "60"))
        self.available = True

    def get_fund_flow_data(self, symbol: str, days: int | None = None) -> dict:
        if days is not None:
            self.days = max(1, int(days))
        data = {
            "symbol": _compact_symbol(symbol),
            "fund_flow_data": None,
            "data_success": False,
            "source": "akshare",
        }
        if not _is_chinese_stock(symbol):
            data["error"] = "资金流向数据仅支持中国A股股票"
            return data
        try:
            market = self._get_market(symbol)
            fund_flow_data = self._get_individual_fund_flow(symbol, market)
            if fund_flow_data:
                data["fund_flow_data"] = fund_flow_data
                data["source"] = fund_flow_data.get("source", data["source"])
                data["data_success"] = True
            else:
                data["error"] = "未能获取到资金流向数据"
        except Exception as exc:
            data["error"] = str(exc)
        return data

    @staticmethod
    def _get_market(symbol: str) -> str:
        code = _compact_symbol(symbol)
        if code.startswith("60") or code.startswith("688"):
            return "sh"
        if code.startswith("00") or code.startswith("30"):
            return "sz"
        if code.startswith(("8", "4")):
            return "bj"
        return "sz"

    @staticmethod
    def _safe_number(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return default
            if isinstance(value, (int, float)):
                return float(value)
            text = str(value).strip().replace(",", "").replace("%", "")
            if text in ("", "-", "--", "None", "nan", "N/A"):
                return default
            multiplier = 1.0
            if text.endswith("亿元"):
                multiplier = 100000000.0
                text = text[:-2]
            elif text.endswith("万元"):
                multiplier = 10000.0
                text = text[:-2]
            elif text.endswith("亿"):
                multiplier = 100000000.0
                text = text[:-1]
            elif text.endswith("万"):
                multiplier = 10000.0
                text = text[:-1]
            return float(text) * multiplier
        except Exception:
            return default

    def _pick_value(self, row: pd.Series, candidates: list[str], default: Any = None) -> Any:
        for col in candidates:
            if col in row.index:
                value = row.get(col)
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    return value
        return default

    @staticmethod
    def _format_trade_date(value: Any) -> str:
        if value is None:
            return datetime.now().strftime("%Y-%m-%d")
        text = str(value)
        if len(text) == 8 and text.isdigit():
            return f"{text[:4]}-{text[4:6]}-{text[6:]}"
        return text

    @staticmethod
    def _parse_trade_date(value: Any):
        if value is None:
            return None
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%m-%d"):
            try:
                parsed = datetime.strptime(text, fmt)
                if fmt == "%m-%d":
                    parsed = parsed.replace(year=datetime.now().year)
                return parsed
            except ValueError:
                continue
        return None

    def _limit_fund_flow_rows(self, rows: list[dict]) -> list[dict]:
        dated_rows = []
        undated_rows = []
        for index, item in enumerate(rows):
            parsed_date = self._parse_trade_date(item.get("日期"))
            if parsed_date:
                dated_rows.append((parsed_date, index, item))
            else:
                undated_rows.append((index, item))
        if dated_rows:
            dated_rows.sort(key=lambda x: (x[0], x[1]))
            return [item for _, _, item in dated_rows][-self.days :]
        return [item for _, item in undated_rows][-self.days :]

    def _standardize_fund_flow_rows(self, df: pd.DataFrame) -> list[dict]:
        rows: list[dict] = []
        for _, row in df.iterrows():
            is_tushare_moneyflow = any(col in row.index for col in ["net_mf_amount", "buy_elg_amount", "sell_elg_amount"])
            super_net = self._safe_number(self._pick_value(row, ["超大单净流入-净额", "超大单净额", "buy_elg_amount"]))
            big_net = self._safe_number(self._pick_value(row, ["大单净流入-净额", "大单净额", "buy_lg_amount"]))
            mid_net = self._safe_number(self._pick_value(row, ["中单净流入-净额", "中单净额", "buy_md_amount"]))
            small_net = self._safe_number(self._pick_value(row, ["小单净流入-净额", "小单净额", "buy_sm_amount"]))
            if {"buy_elg_amount", "sell_elg_amount"}.issubset(set(row.index)):
                super_net = (self._safe_number(row.get("buy_elg_amount")) - self._safe_number(row.get("sell_elg_amount"))) * 10000
            if {"buy_lg_amount", "sell_lg_amount"}.issubset(set(row.index)):
                big_net = (self._safe_number(row.get("buy_lg_amount")) - self._safe_number(row.get("sell_lg_amount"))) * 10000
            if {"buy_md_amount", "sell_md_amount"}.issubset(set(row.index)):
                mid_net = (self._safe_number(row.get("buy_md_amount")) - self._safe_number(row.get("sell_md_amount"))) * 10000
            if {"buy_sm_amount", "sell_sm_amount"}.issubset(set(row.index)):
                small_net = (self._safe_number(row.get("buy_sm_amount")) - self._safe_number(row.get("sell_sm_amount"))) * 10000
            main_net = self._safe_number(self._pick_value(row, ["主力净流入-净额", "主力净流入净额", "主力净额", "主力净流入", "净额", "net_mf_amount"]))
            if is_tushare_moneyflow and "net_mf_amount" in row.index:
                main_net *= 10000
            if main_net == 0 and (super_net or big_net):
                main_net = super_net + big_net
            item = {
                "日期": self._format_trade_date(self._pick_value(row, ["日期", "交易日期", "trade_date"])),
                "收盘价": self._pick_value(row, ["收盘价", "收盘", "close"], "N/A"),
                "涨跌幅": self._safe_number(self._pick_value(row, ["涨跌幅", "pct_chg"], 0)),
                "主力净流入-净额": main_net,
                "主力净流入-净占比": self._safe_number(self._pick_value(row, ["主力净流入-净占比", "主力净占比", "主力净流入净占比"], 0)),
                "超大单净流入-净额": super_net,
                "超大单净流入-净占比": self._safe_number(self._pick_value(row, ["超大单净流入-净占比", "超大单净占比"], 0)),
                "大单净流入-净额": big_net,
                "大单净流入-净占比": self._safe_number(self._pick_value(row, ["大单净流入-净占比", "大单净占比"], 0)),
                "中单净流入-净额": mid_net,
                "中单净流入-净占比": self._safe_number(self._pick_value(row, ["中单净流入-净占比", "中单净占比"], 0)),
                "小单净流入-净额": small_net,
                "小单净流入-净占比": self._safe_number(self._pick_value(row, ["小单净流入-净占比", "小单净占比"], 0)),
            }
            rows.append(item)
        return rows

    def _get_fund_flow_from_tushare(self, symbol: str):
        if not data_source_manager.tushare_available:
            return None
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=max(self.days * 3, 120))).strftime("%Y%m%d")
            df = data_source_manager.tushare_api.moneyflow(
                ts_code=data_source_manager._convert_to_ts_code(symbol),
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and not df.empty:
                return df.copy()
        except Exception as exc:
            logger.debug("tushare 资金流获取失败(%s): %s", symbol, exc)
        return None

    def _get_fund_flow_from_eastmoney_direct(self, symbol: str, market: str):
        try:
            market_map = {"sh": 1, "sz": 0, "bj": 0}
            params = {
                "lmt": "0",
                "klt": "101",
                "secid": f"{market_map.get(market, 0)}.{_compact_symbol(symbol)}",
                "fields1": "f1,f2,f3,f7",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
                "ut": "b2884a393a59ad64002292a3e90d46a5",
                "_": int(datetime.now().timestamp() * 1000),
            }
            response = requests.get(
                "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
                params=params,
                headers={
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Cache-Control": "no-cache",
                    "Referer": "https://data.eastmoney.com/zjlx/detail.html",
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            klines = (payload.get("data") or {}).get("klines") or []
            if not klines:
                return None
            rows = [item.split(",") for item in klines]
            df = pd.DataFrame(rows)
            df.columns = [
                "日期",
                "主力净流入-净额",
                "小单净流入-净额",
                "中单净流入-净额",
                "大单净流入-净额",
                "超大单净流入-净额",
                "主力净流入-净占比",
                "小单净流入-净占比",
                "中单净流入-净占比",
                "大单净流入-净占比",
                "超大单净流入-净占比",
                "收盘价",
                "涨跌幅",
                "_",
                "__",
            ]
            keep_cols = [
                "日期",
                "收盘价",
                "涨跌幅",
                "主力净流入-净额",
                "主力净流入-净占比",
                "超大单净流入-净额",
                "超大单净流入-净占比",
                "大单净流入-净额",
                "大单净流入-净占比",
                "中单净流入-净额",
                "中单净流入-净占比",
                "小单净流入-净额",
                "小单净流入-净占比",
            ]
            df = df[keep_cols]
            for col in keep_cols:
                if col != "日期":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as exc:
            logger.debug("东方财富直连资金流失败(%s): %s", symbol, exc)
            return None

    def _get_fund_flow_from_rank(self, symbol: str):
        ak = _optional_import("akshare")
        if ak is None:
            return None
        for kwargs in ({"indicator": "今日"}, {"indicator": "5日"}, {}):
            try:
                df = ak.stock_individual_fund_flow_rank(**kwargs)
                if df is None or df.empty:
                    continue
                code_col = "代码" if "代码" in df.columns else None
                if not code_col:
                    for col in df.columns:
                        if "代码" in str(col) or str(col).lower() in ("code", "symbol"):
                            code_col = col
                            break
                if not code_col:
                    continue
                stock_data = df[df[code_col].astype(str).str[-6:] == _compact_symbol(symbol)]
                if not stock_data.empty:
                    return stock_data.head(1).copy()
            except Exception as exc:
                logger.debug("资金排名表兜底失败(%s): %s", symbol, exc)
        return None

    def _get_individual_fund_flow(self, symbol: str, market: str):
        df = self._get_fund_flow_from_tushare(symbol)
        source_used = "tushare" if df is not None and not df.empty else ""
        if df is None or df.empty:
            df = self._get_fund_flow_from_eastmoney_direct(symbol, market)
            source_used = "eastmoney_direct" if df is not None and not df.empty else ""
        if df is None or df.empty:
            ak = _optional_import("akshare")
            if ak is not None:
                try:
                    df = ak.stock_individual_fund_flow(stock=_compact_symbol(symbol), market=market)
                    source_used = "akshare_individual" if df is not None and not df.empty else ""
                except Exception as exc:
                    logger.debug("akshare 资金流失败(%s): %s", symbol, exc)
                    df = None
        if df is None or df.empty:
            df = self._get_fund_flow_from_rank(symbol)
            source_used = "akshare_rank" if df is not None and not df.empty else ""
        if df is None or df.empty:
            return None
        df = df.reset_index(drop=True)
        data_list = self._limit_fund_flow_rows(self._standardize_fund_flow_rows(df))
        return {
            "data": data_list,
            "days": len(data_list),
            "requested_days": self.days,
            "columns": df.columns.tolist(),
            "market": market,
            "source": source_used or "unknown",
            "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


class DomesticNewsFetcher:
    """A 股新闻多源抓取。"""

    def __init__(self) -> None:
        self.max_items = 30
        self.available = True

    def get_stock_news(self, symbol: str) -> dict:
        data = {
            "symbol": _compact_symbol(symbol),
            "news_data": None,
            "data_success": False,
            "source": "akshare_multi_source",
        }
        if not _is_chinese_stock(symbol):
            data["error"] = "新闻数据仅支持中国A股股票"
            return data
        try:
            items = self._get_news_data(symbol)
            if items:
                data["news_data"] = items
                data["data_success"] = True
        except Exception as exc:
            data["error"] = str(exc)
        return data

    def _get_stock_name(self, symbol: str) -> str | None:
        try:
            info = data_source_manager.get_stock_basic_info(symbol)
            name = info.get("name")
            if name and name != "未知":
                return name
        except Exception:
            pass
        return None

    def _append_dataframe_items(self, news_items: list[dict], df: pd.DataFrame, source: str, limit: int) -> None:
        if limit <= 0 or df is None or df.empty:
            return
        for _, row in df.head(limit).iterrows():
            item = {"source": source}
            for col in df.columns:
                value = row.get(col)
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    continue
                item[col] = _json_value(value)
            if len(item) > 1:
                news_items.append(item)

    def _dedupe_items(self, items: list[dict]) -> list[dict]:
        seen = set()
        unique_items = []
        title_fields = ["title", "标题", "新闻标题", "公告标题", "content", "内容", "url", "链接"]
        for item in items:
            key_parts = []
            for field in title_fields:
                if item.get(field):
                    key_parts.append(str(item.get(field)).strip())
            key = "|".join(key_parts)[:300] or str(item)[:300]
            if key in seen:
                continue
            seen.add(key)
            unique_items.append(item)
        return unique_items

    def _get_news_data(self, symbol: str) -> dict | None:
        ak = _optional_import("akshare")
        if ak is None:
            return None
        news_items: list[dict] = []
        stock_name = self._get_stock_name(symbol)
        try:
            df = ak.stock_news_em(symbol=_compact_symbol(symbol))
            if df is not None and not df.empty:
                self._append_dataframe_items(news_items, df, "东方财富", self.max_items)
        except Exception as exc:
            logger.debug("东方财富新闻失败(%s): %s", symbol, exc)
        if len(news_items) < 5 and stock_name:
            try:
                df = ak.stock_news_sina(symbol=stock_name)
                if df is not None and not df.empty:
                    self._append_dataframe_items(news_items, df, "新浪财经", self.max_items - len(news_items))
            except Exception as exc:
                logger.debug("新浪新闻失败(%s): %s", symbol, exc)
        if len(news_items) < 5:
            try:
                df = ak.stock_news_cls()
                if df is not None and not df.empty:
                    title_series = df["标题"].astype(str) if "标题" in df.columns else pd.Series("", index=df.index)
                    content_series = df["内容"].astype(str) if "内容" in df.columns else pd.Series("", index=df.index)
                    pattern = _compact_symbol(symbol)
                    if stock_name:
                        pattern = f"{pattern}|{re.escape(str(stock_name))}"
                    df_filtered = df[
                        content_series.str.contains(pattern, na=False, regex=True)
                        | title_series.str.contains(pattern, na=False, regex=True)
                    ]
                    if not df_filtered.empty:
                        self._append_dataframe_items(news_items, df_filtered, "财联社", self.max_items - len(news_items))
            except Exception as exc:
                logger.debug("财联社新闻失败(%s): %s", symbol, exc)
        if not news_items:
            return None
        news_items = self._dedupe_items(news_items)[: self.max_items]
        return {
            "items": news_items,
            "count": len(news_items),
            "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": "最近新闻",
        }


class OverseasMarketData:
    """海外行情、海外新闻和 A 股隔夜影响。"""

    QUOTE_TTL_MINUTES = 15
    NEWS_TTL_MINUTES = 30
    REQUEST_TIMEOUT = 12

    INSTRUMENTS = [
        {"symbol": "^GSPC", "name": "标普500", "group": "美股指数"},
        {"symbol": "^IXIC", "name": "纳斯达克综合", "group": "美股指数"},
        {"symbol": "^DJI", "name": "道琼斯工业", "group": "美股指数"},
        {"symbol": "SPY", "name": "标普500ETF", "group": "美股ETF"},
        {"symbol": "QQQ", "name": "纳指100ETF", "group": "美股ETF"},
        {"symbol": "DIA", "name": "道指ETF", "group": "美股ETF"},
        {"symbol": "^VIX", "name": "VIX波动率", "group": "宏观资产"},
        {"symbol": "SOXX", "name": "半导体ETF", "group": "行业ETF"},
        {"symbol": "KWEB", "name": "中概互联网ETF", "group": "行业ETF"},
        {"symbol": "FXI", "name": "中国大盘ETF", "group": "行业ETF"},
        {"symbol": "ASHR", "name": "沪深300A股ETF", "group": "行业ETF"},
        {"symbol": "MCHI", "name": "MSCI中国ETF", "group": "行业ETF"},
        {"symbol": "NVDA", "name": "英伟达", "group": "美股科技"},
        {"symbol": "AMD", "name": "AMD", "group": "美股科技"},
        {"symbol": "TSM", "name": "台积电ADR", "group": "美股科技"},
        {"symbol": "MSFT", "name": "微软", "group": "美股科技"},
        {"symbol": "AAPL", "name": "苹果", "group": "美股科技"},
        {"symbol": "AMZN", "name": "亚马逊", "group": "美股科技"},
        {"symbol": "META", "name": "Meta", "group": "美股科技"},
        {"symbol": "GOOGL", "name": "谷歌A", "group": "美股科技"},
        {"symbol": "TSLA", "name": "特斯拉", "group": "美股科技"},
        {"symbol": "BABA", "name": "阿里巴巴ADR", "group": "中概股"},
        {"symbol": "PDD", "name": "拼多多ADR", "group": "中概股"},
        {"symbol": "JD", "name": "京东ADR", "group": "中概股"},
        {"symbol": "NIO", "name": "蔚来ADR", "group": "中概股"},
        {"symbol": "XPEV", "name": "小鹏汽车ADR", "group": "中概股"},
        {"symbol": "LI", "name": "理想汽车ADR", "group": "中概股"},
        {"symbol": "CL=F", "name": "WTI原油", "group": "宏观资产"},
        {"symbol": "GC=F", "name": "COMEX黄金", "group": "宏观资产"},
        {"symbol": "USDCNY=X", "name": "美元/人民币", "group": "宏观资产"},
    ]

    NEWS_FEEDS = [
        {
            "name": "Google News - 美国股市",
            "topic": "美国股市",
            "url": (
                "https://news.google.com/rss/search?q="
                + quote_plus('"US stocks" OR "Federal Reserve" OR "S&P 500"')
                + "&hl=en-US&gl=US&ceid=US:en"
            ),
        },
        {
            "name": "Google News - AI与半导体",
            "topic": "AI与半导体",
            "url": (
                "https://news.google.com/rss/search?q="
                + quote_plus('semiconductor OR "AI chips" OR Nvidia OR TSMC')
                + "&hl=en-US&gl=US&ceid=US:en"
            ),
        },
        {
            "name": "Google News - 中国资产",
            "topic": "中国资产",
            "url": (
                "https://news.google.com/rss/search?q="
                + quote_plus('"China stocks" OR "Chinese ADRs" OR "US-China trade"')
                + "&hl=en-US&gl=US&ceid=US:en"
            ),
        },
        {
            "name": "CNBC Top News",
            "topic": "海外财经",
            "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        },
    ]

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = Path(db_path or (settings.data_dir / "aiagents_overseas.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._instrument_map = {item["symbol"]: item for item in self.INSTRUMENTS}
        self._quote_refresh_lock = Lock()
        self._quote_retry_at = 0.0
        self._quote_last_error = ""
        self._init_database()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS overseas_quotes (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    market_group TEXT,
                    price REAL,
                    previous_close REAL,
                    change_pct REAL,
                    quote_time TEXT,
                    fetched_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    payload TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS overseas_news_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _now_text() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _to_float(value) -> float | None:
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _is_fresh(timestamp: str, ttl: timedelta) -> bool:
        if not timestamp:
            return False
        try:
            return datetime.now() - datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") <= ttl
        except Exception:
            return False

    def _read_quote_cache(self) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM overseas_quotes ORDER BY market_group, symbol").fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["cache_fresh"] = self._is_fresh(item.get("fetched_at"), timedelta(minutes=self.QUOTE_TTL_MINUTES))
            item["cache_hit"] = True
            items.append(item)
        return items

    def _upsert_quotes(self, quotes: list[dict]) -> None:
        if not quotes:
            return
        now = self._now_text()
        with self._connection() as conn:
            for quote in quotes:
                conn.execute(
                    """
                    INSERT INTO overseas_quotes (
                        symbol, name, market_group, price, previous_close, change_pct,
                        quote_time, fetched_at, source, payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        name=excluded.name,
                        market_group=excluded.market_group,
                        price=excluded.price,
                        previous_close=excluded.previous_close,
                        change_pct=excluded.change_pct,
                        quote_time=excluded.quote_time,
                        fetched_at=excluded.fetched_at,
                        source=excluded.source,
                        payload=excluded.payload
                    """,
                    (
                        quote["symbol"],
                        quote.get("name", quote["symbol"]),
                        quote.get("market_group", "其他"),
                        quote.get("price"),
                        quote.get("previous_close"),
                        quote.get("change_pct"),
                        quote.get("quote_time", now),
                        now,
                        quote.get("source", "yfinance"),
                        json.dumps(quote, ensure_ascii=False),
                    ),
                )

    def _get_multi_ticker_history(self, symbols: list[str], period: str = "7d") -> pd.DataFrame:
        yf = _optional_import("yfinance")
        if yf is None:
            raise RuntimeError("yfinance 未安装")
        if not symbols:
            return pd.DataFrame()
        from curl_cffi.requests import Session

        deadline = time.monotonic() + 30

        class DeadlineSession(Session):
            def request(self, *args, **kwargs):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Yahoo refresh deadline exceeded")
                # yfinance metadata/cookie calls use their own 10-second
                # timeout. Enforce the budget at the transport boundary too.
                kwargs["timeout"] = min(4, remaining)
                return super().request(*args, **kwargs)

        with DeadlineSession(impersonate="chrome") as session:
            kwargs = dict(period=period, interval="1d", group_by="ticker",
                          auto_adjust=False, progress=False, timeout=4, session=session)
            # A blocked Yahoo connection used to spend two timeouts on each of
            # 30 symbols. Test two first, then cap concurrency for the rest.
            first = yf.download(tickers=symbols[:2], threads=False, **kwargs)
            if first is None or first.empty or first.dropna(how="all").empty:
                return pd.DataFrame()
            if len(symbols) <= 2:
                return first
            rest = yf.download(tickers=symbols[2:], threads=4, **kwargs)
            return pd.concat([first, rest], axis=1) if rest is not None and not rest.empty else first

    @staticmethod
    def _select_ticker_frame(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        if not isinstance(data.columns, pd.MultiIndex):
            return data.copy()
        if symbol in data.columns.get_level_values(0):
            return data[symbol].copy()
        if symbol in data.columns.get_level_values(-1):
            return data.xs(symbol, axis=1, level=-1).copy()
        return pd.DataFrame()

    def _fetch_quotes(self, instruments: list[dict]) -> list[dict]:
        symbols = [item["symbol"] for item in instruments]
        data = self._get_multi_ticker_history(symbols, period="7d")
        results = []
        for instrument in instruments:
            frame = self._select_ticker_frame(data, instrument["symbol"])
            if frame.empty or "Close" not in frame:
                continue
            closes = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            if closes.empty:
                continue
            price = self._to_float(closes.iloc[-1])
            previous = self._to_float(closes.iloc[-2]) if len(closes) > 1 else None
            change_pct = None
            if price is not None and previous not in (None, 0):
                change_pct = round((price - previous) / previous * 100, 4)
            results.append(
                {
                    "symbol": instrument["symbol"],
                    "name": instrument["name"],
                    "market_group": instrument["group"],
                    "price": price,
                    "previous_close": previous,
                    "change_pct": change_pct,
                    "quote_time": pd.Timestamp(closes.index[-1]).isoformat(),
                    "source": "Yahoo Finance / yfinance",
                    "cache_hit": False,
                    "cache_fresh": True,
                }
            )
        return results

    def get_market_snapshot(self, force_refresh: bool = False) -> dict:
        cached = self._read_quote_cache()
        cached_by_symbol = {item["symbol"]: item for item in cached}
        fresh = len(cached_by_symbol) >= len(self.INSTRUMENTS) and all(item.get("cache_fresh") for item in cached_by_symbol.values())
        if fresh and not force_refresh:
            return self._build_snapshot(cached, from_cache=True, fallback=False)
        if time.monotonic() < self._quote_retry_at:
            return self._build_snapshot(cached, from_cache=bool(cached),
                                        fallback=True, error=self._quote_last_error)
        if not self._quote_refresh_lock.acquire(blocking=False):
            return self._build_snapshot(cached, from_cache=bool(cached), fallback=True,
                                        error="海外行情正在刷新，暂用已有缓存")
        try:
            # Recheck after acquiring the lock: concurrent requests must not
            # turn a failed refresh (including a forced one) into a retry storm.
            if time.monotonic() < self._quote_retry_at:
                return self._build_snapshot(cached, from_cache=bool(cached),
                                            fallback=True, error=self._quote_last_error)
            fetched = self._fetch_quotes(self.INSTRUMENTS)
            if fetched:
                self._upsert_quotes(fetched)
                partial = len({quote["symbol"] for quote in fetched}) < len(self.INSTRUMENTS)
                self._quote_last_error = "海外行情仅返回部分标的，缺失部分保留原缓存" if partial else ""
                self._quote_retry_at = time.monotonic() + self.QUOTE_TTL_MINUTES * 60
                return self._build_snapshot(self._read_quote_cache(), from_cache=False,
                                            fallback=partial, error=self._quote_last_error)
            raise RuntimeError("海外行情接口未返回有效数据")
        except Exception as exc:
            self._quote_last_error = f"海外行情暂不可用（{type(exc).__name__}），15分钟后重试"
            self._quote_retry_at = time.monotonic() + self.QUOTE_TTL_MINUTES * 60
            logger.warning("%s", self._quote_last_error)
            if cached:
                return self._build_snapshot(cached, from_cache=True, fallback=True, error=self._quote_last_error)
            return self._build_snapshot([], from_cache=False, fallback=True, error=self._quote_last_error)
        finally:
            self._quote_refresh_lock.release()

    def _build_snapshot(self, quotes: list[dict], from_cache: bool, fallback: bool, error: str = "") -> dict:
        latest_time = max((item.get("fetched_at", "") for item in quotes), default="")
        impact = self.build_a_share_impact(quotes)
        return {
            "data_success": bool(quotes),
            "quotes": quotes,
            "impact": impact,
            "quote_count": len(quotes),
            "updated_at": latest_time,
            "source": "Yahoo Finance / yfinance",
            "cache_hit": from_cache,
            "cache_fallback": fallback,
            "error": error,
        }

    def _read_news_cache(self, cache_key: str) -> dict | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM overseas_news_cache WHERE cache_key = ?", (cache_key,)).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row["payload"])
        except Exception:
            return None
        payload["fetched_at"] = row["fetched_at"]
        payload["cache_hit"] = True
        payload["cache_fresh"] = self._is_fresh(row["fetched_at"], timedelta(minutes=self.NEWS_TTL_MINUTES))
        return payload

    def _save_news_cache(self, cache_key: str, payload: dict, source: str) -> None:
        now = self._now_text()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO overseas_news_cache (cache_key, payload, fetched_at, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload=excluded.payload,
                    fetched_at=excluded.fetched_at,
                    source=excluded.source
                """,
                (cache_key, json.dumps(payload, ensure_ascii=False), now, source),
            )

    def _fetch_rss(self, feed: dict, limit: int) -> list[dict]:
        response = requests.get(
            feed["url"],
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36"
            },
            timeout=self.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items = []
        for node in root.findall(".//item")[:limit]:
            source_node = node.find("source")
            source = source_node.text.strip() if source_node is not None and source_node.text else feed["name"]
            link_node = node.find("link")
            items.append(
                {
                    "title": (node.findtext("title") or "").strip(),
                    "summary": (node.findtext("description") or "").strip(),
                    "source": source,
                    "feed": feed["name"],
                    "topic": feed["topic"],
                    "published_at": (node.findtext("pubDate") or "").strip(),
                    "url": (link_node.text if link_node is not None else "") or "",
                }
            )
        return [item for item in items if item["title"]]

    @staticmethod
    def _dedupe_news(items: list[dict]) -> list[dict]:
        seen = set()
        results = []
        for item in items:
            key = item.get("title", "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(item)
        return results

    def get_overseas_news(self, force_refresh: bool = False, limit_per_feed: int = 8) -> dict:
        cache_key = "global_market_news_v1"
        cached = self._read_news_cache(cache_key)
        if cached and cached.get("cache_fresh") and not force_refresh:
            return cached
        items: list[dict] = []
        errors: list[str] = []
        for feed in self.NEWS_FEEDS:
            try:
                items.extend(self._fetch_rss(feed, limit_per_feed))
            except Exception as exc:
                errors.append(f"{feed['name']}: {exc}")
        items = self._dedupe_news(items)
        if items:
            payload = {
                "data_success": True,
                "items": items,
                "count": len(items),
                "updated_at": self._now_text(),
                "sources": [feed["name"] for feed in self.NEWS_FEEDS],
                "cache_hit": False,
                "cache_fresh": True,
                "cache_fallback": False,
                "errors": errors,
            }
            self._save_news_cache(cache_key, payload, "RSS")
            return payload
        if cached:
            cached["cache_fallback"] = True
            cached["error"] = "；".join(errors)
            return cached
        return {
            "data_success": False,
            "items": [],
            "count": 0,
            "updated_at": "",
            "sources": [feed["name"] for feed in self.NEWS_FEEDS],
            "cache_hit": False,
            "cache_fallback": True,
            "error": "；".join(errors) or "海外新闻源未返回数据",
        }

    def get_symbol_news(self, symbol: str, name: str = "", force_refresh: bool = False) -> dict:
        normalized = (symbol or "").upper().strip()
        cache_key = f"symbol_news_{normalized}"
        cached = self._read_news_cache(cache_key)
        if cached and cached.get("cache_fresh") and not force_refresh:
            return cached
        query = f'"{normalized}"'
        if name:
            query += f' OR "{name}"'
        feed = {
            "name": "Google News - 个股",
            "topic": "个股新闻",
            "url": (
                "https://news.google.com/rss/search?q="
                + quote_plus(query)
                + "&hl=en-US&gl=US&ceid=US:en"
            ),
        }
        try:
            items = self._fetch_rss(feed, 12)
            payload = {
                "data_success": bool(items),
                "items": items,
                "count": len(items),
                "updated_at": self._now_text(),
                "sources": [feed["name"]],
                "cache_hit": False,
                "cache_fresh": True,
                "cache_fallback": False,
            }
            if items:
                self._save_news_cache(cache_key, payload, "RSS")
            return payload
        except Exception as exc:
            if cached:
                cached["cache_fallback"] = True
                cached["error"] = str(exc)
                return cached
            return {
                "data_success": False,
                "items": [],
                "count": 0,
                "updated_at": "",
                "sources": [feed["name"]],
                "cache_fallback": True,
                "error": str(exc),
            }

    def get_context_for_analysis(
        self,
        symbol: str = "",
        stock_name: str = "",
        include_symbol_news: bool = False,
        force_refresh: bool = False,
    ) -> dict:
        market = self.get_market_snapshot(force_refresh=force_refresh)
        global_news = self.get_overseas_news(force_refresh=force_refresh)
        symbol_news = {}
        if include_symbol_news and symbol:
            symbol_news = self.get_symbol_news(symbol, stock_name, force_refresh=force_refresh)
        return {
            "market": market,
            "global_news": global_news,
            "symbol_news": symbol_news,
            "impact": market.get("impact", {}),
            "data_success": bool(market.get("data_success") or global_news.get("data_success")),
        }

    @staticmethod
    def build_a_share_impact(quotes: list[dict]) -> dict:
        quote_map = {item.get("symbol"): item for item in quotes}

        def change(symbol: str) -> float:
            value = quote_map.get(symbol, {}).get("change_pct")
            try:
                return float(value)
            except Exception:
                return 0.0

        components = [
            ("纳斯达克综合", change("^IXIC"), 1.4),
            ("标普500", change("^GSPC"), 1.0),
            ("半导体ETF", change("SOXX"), 1.25),
            ("中概互联网ETF", change("KWEB"), 0.9),
            ("中国大盘ETF", change("FXI"), 0.8),
        ]
        raw_score = 50 + sum(value * weight * 6 for _, value, weight in components)
        usdcny_change = change("USDCNY=X")
        if usdcny_change > 0.25:
            raw_score -= min(8, usdcny_change * 10)
        elif usdcny_change < -0.25:
            raw_score += min(5, abs(usdcny_change) * 8)
        score = int(max(0, min(100, round(raw_score))))
        if score >= 65:
            direction, level = "偏正面", "较低"
        elif score <= 35:
            direction, level = "偏负面", "较高"
        else:
            direction, level = "中性偏谨慎", "中等"
        ordered = sorted(components, key=lambda item: abs(item[1] * item[2]), reverse=True)
        highlights = [f"{name}{value:+.2f}%" for name, value, _ in ordered[:3] if value != 0]
        if usdcny_change:
            highlights.append(f"美元/人民币{usdcny_change:+.2f}%")
        summary = (
            f"隔夜海外环境{direction}（{score}/100），"
            + ("；".join(highlights) if highlights else "等待有效海外行情更新")
            + "。仅用于判断A股开盘风险偏好，个股交易仍需结合自身趋势、量价和资金确认。"
        )
        return {
            "score": score,
            "direction": direction,
            "risk_level": level,
            "highlights": highlights,
            "summary": summary,
        }

    def get_instrument_options(self) -> list[dict]:
        return list(self.INSTRUMENTS)


data_source_manager = LocalDataSourceManager()
fund_flow_fetcher = FundFlowAkshareDataFetcher()
overseas_market_data = OverseasMarketData()


def build_unified_news_context(symbol: str, stock_name: str = "") -> dict:
    # 历史同步写入本地新闻库后，分析优先读取本地归档，再叠加实时公开源。
    # 这样新闻分析不会因为一次 RSS/AKShare 请求失败而完全失去历史上下文。
    from app.services.news_sync import read_local_news

    def _local_item(item: dict, default_scope: str) -> dict:
        return {
            "title": str(item.get("title") or ""),
            "date": str(item.get("published_at") or item.get("date") or ""),
            "source": str(item.get("source") or "本地新闻库"),
            "content": str(item.get("summary") or item.get("content") or ""),
            "url": str(item.get("url") or ""),
            "topic": str(item.get("channels") or default_scope),
            "market_scope": str(item.get("market_scope") or default_scope),
            "symbols": str(item.get("symbols") or ""),
            "sentiment": str(item.get("sentiment") or ""),
        }

    def _dedupe(items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        result: list[dict] = []
        for item in items:
            key = str(item.get("url") or "").strip() or "|".join((
                str(item.get("date") or ""),
                str(item.get("source") or ""),
                str(item.get("title") or ""),
            ))
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    is_domestic_stock = _is_chinese_stock(symbol)
    domestic_data: dict = {}
    domestic_items: list[dict] = []
    if is_domestic_stock:
        local_domestic = read_local_news(symbol=symbol, market="domestic", limit=30)
        domestic_items.extend([_local_item(item, "A股/国内财经") for item in local_domestic])
        domestic_data = DomesticNewsFetcher().get_stock_news(symbol)
        domestic_items.extend(list((domestic_data.get("news_data") or {}).get("items") or []))

    local_overseas = read_local_news(
        symbol="" if is_domestic_stock else symbol,
        market="overseas",
        limit=30,
    )
    overseas_context = overseas_market_data.get_context_for_analysis(
        symbol=symbol,
        stock_name=stock_name,
        include_symbol_news=not is_domestic_stock,
    )
    overseas_items: list[dict] = [_local_item(item, "美股/海外财经") for item in local_overseas]
    for news_key in ("symbol_news", "global_news"):
        news_data = overseas_context.get(news_key) or {}
        for item in news_data.get("items", [])[:15]:
            overseas_items.append(
                {
                    "title": item.get("title", ""),
                    "date": item.get("published_at", ""),
                    "source": item.get("source", item.get("feed", "海外新闻")),
                    "content": item.get("summary", ""),
                    "url": item.get("url", ""),
                    "topic": item.get("topic", "海外财经"),
                    "market_scope": "海外",
                }
            )
    domestic_items = _dedupe(domestic_items)
    overseas_items = _dedupe(overseas_items)
    all_items = domestic_items + overseas_items
    data_success = bool(all_items or overseas_context.get("market", {}).get("data_success"))
    source_parts = []
    if domestic_items:
        source_parts.append("A股多源新闻")
    if local_overseas:
        source_parts.append("海外新闻本地归档")
    if overseas_context.get("global_news", {}).get("data_success"):
        source_parts.append("海外RSS新闻")
    if overseas_context.get("market", {}).get("data_success"):
        source_parts.append("美股与宏观行情")
    return {
        "symbol": _compact_symbol(symbol),
        "news_data": {
            "items": all_items[:45],
            "count": len(all_items[:45]),
            "query_time": overseas_context.get("market", {}).get("updated_at", ""),
            "date_range": "最新本地缓存与公开新闻",
        },
        "domestic_news": domestic_data,
        "overseas_context": overseas_context,
        "data_success": data_success,
        "source": " + ".join(source_parts) or "海外市场本地缓存",
    }
