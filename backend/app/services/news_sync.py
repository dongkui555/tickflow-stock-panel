"""国内外新闻历史同步、归一化与本地归档。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

import polars as pl
import requests

from app.config import settings

logger = logging.getLogger(__name__)

NEWS_COLUMNS = [
    "id", "date", "published_at", "market", "market_scope", "source",
    "title", "summary", "content", "channels", "symbols", "url",
    "language", "sentiment", "raw",
]


def _empty_news() -> pl.DataFrame:
    return pl.DataFrame({
        "id": pl.Series([], dtype=pl.Utf8),
        "date": pl.Series([], dtype=pl.Date),
        **{
            column: pl.Series([], dtype=pl.Utf8)
            for column in NEWS_COLUMNS
            if column not in {"id", "date"}
        },
    }).select(NEWS_COLUMNS)


def _split_env(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
        "%Y%m%dT%H%M%S", "%Y-%m-%d", "%Y%m%d",
    ):
        try:
            return datetime.strptime(text[: len(datetime.now().strftime(fmt))], fmt)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None)
    except ValueError:
        pass
    # RSS 的 pubDate 通常是 RFC 2822 格式，例如
    # "Thu, 27 Aug 2026 20:49:00 GMT"。
    try:
        parsed = parsedate_to_datetime(text)
        return parsed.replace(tzinfo=None)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _text(value: Any, limit: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    result = str(value).strip()
    return result[:limit] if limit else result


def _fingerprint(*parts: Any) -> str:
    value = "|".join(_text(part) for part in parts)
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _normalize_alphavantage_feed(feed: list[dict[str, Any]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in feed:
        published = _parse_datetime(item.get("time_published"))
        title = _text(item.get("title"))
        summary = _text(item.get("summary"), 1200)
        if published is None or not title:
            continue
        ticker_items = item.get("ticker_sentiment") or []
        tickers = []
        for ticker in ticker_items:
            if isinstance(ticker, dict) and ticker.get("ticker"):
                tickers.append(_text(ticker.get("ticker")))
        topics = []
        for topic in item.get("topics") or []:
            if isinstance(topic, dict) and topic.get("topic"):
                topics.append(_text(topic.get("topic")))
        source = _text(item.get("source"), 200) or "Alpha Vantage"
        rows.append({
            "id": _fingerprint(source, published.isoformat(), title, item.get("url")),
            "date": published.date(),
            "published_at": published.strftime("%Y-%m-%d %H:%M:%S"),
            "market": "overseas",
            "market_scope": "美股/海外财经",
            "source": f"Alpha Vantage/{source}",
            "title": title,
            "summary": summary,
            "content": summary,
            "channels": ",".join(topics),
            "symbols": ",".join(sorted(set(tickers))),
            "url": _text(item.get("url")),
            "language": "en",
            "sentiment": _text(item.get("overall_sentiment_label")),
            "raw": _text(item),
        })
    if not rows:
        return _empty_news()
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).select(NEWS_COLUMNS)


def _normalize_rss_items(items: list[dict[str, Any]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        published = _parse_datetime(item.get("published_at") or item.get("date"))
        title = _text(item.get("title"))
        if published is None or not title:
            continue
        source = _text(item.get("source") or item.get("feed"), 200) or "公开 RSS"
        rows.append({
            "id": _fingerprint(source, published.isoformat(), title, item.get("url")),
            "date": published.date(),
            "published_at": published.strftime("%Y-%m-%d %H:%M:%S"),
            "market": "overseas",
            "market_scope": "美股/海外财经",
            "source": f"RSS/{source}",
            "title": title,
            "summary": _text(item.get("summary") or item.get("content"), 1200),
            "content": _text(item.get("summary") or item.get("content")),
            "channels": _text(item.get("topic")),
            "symbols": _text(item.get("symbols")),
            "url": _text(item.get("url")),
            "language": "en",
            "sentiment": "",
            "raw": _text(item),
        })
    if not rows:
        return _empty_news()
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).select(NEWS_COLUMNS)


def _fetch_alphavantage(start: date, end: date, on_progress: Callable[[int, int], None] | None) -> pl.DataFrame:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not api_key:
        return _empty_news()

    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=30), end)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)

    frames: list[pl.DataFrame] = []
    for index, (window_start, window_end) in enumerate(windows, start=1):
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": api_key,
            "time_from": window_start.strftime("%Y%m%dT0000"),
            "time_to": window_end.strftime("%Y%m%dT2359"),
            "sort": "EARLIEST",
            "limit": 1000,
        }
        try:
            response = requests.get(
                "https://www.alphavantage.co/query",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("Information") or payload.get("Note"):
                logger.warning("Alpha Vantage 新闻接口提示: %s", payload.get("Information") or payload.get("Note"))
            frames.append(_normalize_alphavantage_feed(payload.get("feed") or []))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alpha Vantage 新闻窗口 %s~%s 失败: %s", window_start, window_end, exc)
        if on_progress:
            on_progress(index, len(windows))
        if index < len(windows):
            time.sleep(max(0.0, float(os.getenv("NEWS_SYNC_DELAY_S", "0.25"))))
    return _concat_news(frames)


def _fetch_recent_rss() -> pl.DataFrame:
    try:
        from app.services.aiagents_sources import overseas_market_data
        payload = overseas_market_data.get_overseas_news(force_refresh=True, limit_per_feed=20)
        return _normalize_rss_items(payload.get("items") or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("海外 RSS 新闻归档失败: %s", exc)
        return _empty_news()


def _concat_news(frames: list[pl.DataFrame]) -> pl.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.is_empty()]
    if not frames:
        return _empty_news()
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .select(NEWS_COLUMNS)
        .unique(subset=["id"], keep="last")
        .sort(["date", "published_at"])
    )


def _resolve_tushare_news(start: datetime, end: datetime, on_progress: Callable | None) -> pl.DataFrame:
    try:
        from app.plugins.tushare.provider import TushareProvider

        # 网关/账户的新闻源权限并不总是一致。保留用户显式配置的优先级，
        # 但在指定源返回空时继续尝试常见可用源，避免 10jqka 无数据就直接判定
        # 国内新闻整体不可用。实际返回的来源仍由 TushareProvider 记录。
        sources = _split_env("TUSHARE_NEWS_SRC", "cls")
        fallback_sources = _split_env("TUSHARE_NEWS_FALLBACK_SRC", "cls,sina,10jqka")
        seen = {source.lower() for source in sources}
        for source in fallback_sources:
            if source.lower() not in seen:
                sources.append(source)
                seen.add(source.lower())
        provider = TushareProvider()
        try:
            return provider.get_news(sources, start, end, on_chunk_done=on_progress)
        finally:
            provider.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tushare 国内新闻同步失败: %s", exc)
        return _empty_news()


def _first_present(item: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = item.get(field)
        if value not in (None, ""):
            return value
    return None


def _aiagents_news_symbols() -> list[str]:
    """选择少量受关注标的抓取近期新闻，避免对公开新闻源全A高频请求。"""
    values = _split_env("AIAGENTS_NEWS_SYMBOLS")
    if not values:
        try:
            from app.services import preferences
            values.extend(preferences.get_realtime_watchlist_symbols())
        except Exception as exc:  # noqa: BLE001
            logger.debug("读取实时自选股作为新闻回退范围失败: %s", exc)
        try:
            from app.tickflow.pools import get_pool
            values.extend(get_pool("watchlist"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("读取自选池作为新闻回退范围失败: %s", exc)

    max_symbols = max(1, min(100, int(os.getenv("AIAGENTS_NEWS_MAX_SYMBOLS", "30"))))
    unique: list[str] = []
    for value in values:
        symbol = str(value or "").strip().upper()
        if symbol and symbol not in unique:
            unique.append(symbol)
        if len(unique) >= max_symbols:
            break
    return unique


def _normalize_aiagents_domestic_items(symbol: str, items: list[dict[str, Any]]) -> pl.DataFrame:
    """把旧 AI Agents 的东方财富/新浪/财联社新闻归一为本地新闻归档格式。"""
    rows: list[dict[str, Any]] = []
    compact_symbol = "".join(re.findall(r"\d", symbol))[-6:].zfill(6)
    for item in items:
        if not isinstance(item, dict):
            continue
        title = _text(_first_present(item, ("title", "标题", "新闻标题", "公告标题")))
        content = _text(_first_present(item, ("content", "内容", "摘要", "summary", "新闻内容")), 5000)
        if not title and not content:
            continue
        published = _parse_datetime(
            _first_present(item, ("published_at", "发布时间", "发布时间", "时间", "日期", "date"))
        ) or datetime.now()
        source = _text(item.get("source"), 200) or "公开财经源"
        url = _text(_first_present(item, ("url", "链接", "URL")))
        rows.append({
            "id": _fingerprint("AI Agents", source, published.isoformat(), title, content[:600], url),
            "date": published.date(),
            "published_at": published.strftime("%Y-%m-%d %H:%M:%S"),
            "market": "domestic",
            "market_scope": "A股/国内财经",
            "source": f"AI Agents/{source}",
            "title": title or content[:80],
            "summary": content[:1200],
            "content": content,
            "channels": _text(_first_present(item, ("频道", "栏目", "category", "频道名称"))),
            "symbols": ",".join(value for value in (symbol, compact_symbol) if value),
            "url": url,
            "language": "zh",
            "sentiment": "",
            "raw": _text(item),
        })
    if not rows:
        return _empty_news()
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).select(NEWS_COLUMNS)


def _fetch_aiagents_domestic_recent(
    on_progress: Callable[[int, int], None] | None,
) -> pl.DataFrame:
    """用旧项目已有的多源新闻能力补充近期新闻，不伪装成两年历史回填。"""
    symbols = _aiagents_news_symbols()
    if not symbols:
        logger.info("AI Agents 新闻回退跳过：未配置 AIAGENTS_NEWS_SYMBOLS 且自选股为空")
        return _empty_news()
    try:
        from app.services.aiagents_sources import DomesticNewsFetcher

        fetcher = DomesticNewsFetcher()
    except Exception as exc:  # noqa: BLE001
        logger.warning("AI Agents 国内新闻抓取器初始化失败: %s", exc)
        return _empty_news()

    frames: list[pl.DataFrame] = []
    delay = max(0.0, float(os.getenv("AIAGENTS_NEWS_DELAY_S", "0.2")))
    for index, symbol in enumerate(symbols, start=1):
        try:
            payload = fetcher.get_stock_news(symbol)
            news_data = payload.get("news_data") if isinstance(payload, dict) else None
            items = news_data.get("items") if isinstance(news_data, dict) else []
            frame = _normalize_aiagents_domestic_items(symbol, items or [])
            if not frame.is_empty():
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            logger.debug("AI Agents 国内新闻抓取失败(%s): %s", symbol, exc)
        if on_progress:
            on_progress(index, len(symbols))
        if index < len(symbols) and delay:
            time.sleep(delay)
    return _concat_news(frames)


def sync_news_history(
    data_dir: Path,
    start: date,
    end: date,
    repo,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """同步并写入国内/海外新闻，返回覆盖与待补原因。"""
    result: dict[str, Any] = {
        "domestic_rows": 0,
        "overseas_rows": 0,
        "sources": [],
        "coverage": {},
        "unavailable": [],
    }

    def emit(stage_pct: int, msg: str) -> None:
        if on_progress:
            on_progress("sync_news", stage_pct, msg)

    emit(0, f"同步国内新闻：{start} 至 {end}")
    domestic = _resolve_tushare_news(
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.max.time()),
        lambda current, total: emit(int(35 * current / max(total, 1)), f"国内新闻源 {current}/{total}"),
    )
    if not domestic.is_empty():
        repo.append_news(domestic)
        result["domestic_rows"] = domestic.height
        result["sources"].append("Tushare 新闻")
        result["coverage"]["domestic"] = {"start": str(domestic["date"].min()), "end": str(domestic["date"].max()), "mode": "history"}
    else:
        emit(35, "Tushare 新闻为空，回退 AI Agents 多源国内新闻")
        domestic = _fetch_aiagents_domestic_recent(
            lambda current, total: emit(
                35 + int(5 * current / max(total, 1)),
                f"AI Agents 国内新闻 {current}/{total}",
            )
        )
        if not domestic.is_empty():
            repo.append_news(domestic)
            result["domestic_rows"] = domestic.height
            result["sources"].append("AI Agents 多源国内新闻（近期）")
            result["coverage"]["domestic"] = {
                "start": str(domestic["date"].min()),
                "end": str(domestic["date"].max()),
                "mode": "recent_fallback",
            }
            result["unavailable"].append({
                "dataset": "news_domestic_history",
                "reason": "Tushare 新闻接口未返回数据；已用东方财富/新浪/财联社补充近期新闻，但公开抓取源不能可靠回填两年历史。",
            })
        else:
            result["unavailable"].append({
                "dataset": "news_domestic",
                "reason": "Tushare 新闻接口未返回数据；AI Agents 的东方财富/新浪/财联社近期回退也未返回内容。请确认 TUSHARE_NEWS_SRC、news 权限、akshare 依赖和网络状态。",
            })
    emit(40, f"国内新闻完成，写入 {result['domestic_rows']:,} 行")

    emit(45, f"同步海外新闻：{start} 至 {end}")
    overseas = _fetch_alphavantage(
        start,
        end,
        lambda current, total: emit(45 + int(35 * current / max(total, 1)), f"海外历史新闻 {current}/{total}"),
    )
    overseas_history = not overseas.is_empty()
    if overseas.is_empty() and not os.getenv("ALPHAVANTAGE_API_KEY", "").strip():
        overseas = _fetch_recent_rss()
    if not overseas.is_empty():
        repo.append_news(overseas)
        result["overseas_rows"] = overseas.height
        source = "Alpha Vantage 新闻" if overseas_history else "公开 RSS（仅近期）"
        result["sources"].append(source)
        result["coverage"]["overseas"] = {
            "start": str(overseas["date"].min()),
            "end": str(overseas["date"].max()),
            "mode": "history" if overseas_history else "recent",
        }
        if not overseas_history:
            result["unavailable"].append({
                "dataset": "news_overseas_history",
                "reason": "未配置 ALPHAVANTAGE_API_KEY，海外新闻仅归档当前 RSS 返回的近期内容，未覆盖两年。",
            })
    else:
        result["unavailable"].append({
            "dataset": "news_overseas",
            "reason": "海外历史新闻未返回数据；请配置 ALPHAVANTAGE_API_KEY，或检查 RSS 网络访问。",
        })
    emit(90, f"海外新闻完成，写入 {result['overseas_rows']:,} 行")

    repo.rebuild_views()
    emit(100, "国内外新闻已落库")
    return result


def read_local_news(
    data_dir: Path | None = None,
    symbol: str = "",
    market: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    root = Path(data_dir or settings.data_dir) / "news"
    files = list(root.rglob("*.parquet")) if root.exists() else []
    if not files:
        return []
    try:
        paths = [str(path) for path in files]
        try:
            frame = pl.scan_parquet(paths, hive_partitioning=True)
        except TypeError:
            frame = pl.scan_parquet(paths)
        if market:
            frame = frame.filter(pl.col("market") == market)
        if symbol:
            normalized = str(symbol).split(".", 1)[0]
            frame = frame.filter(
                pl.col("symbols").fill_null("").str.contains(normalized, literal=True)
                | pl.col("symbols").fill_null("").eq("")
            )
        return frame.sort("published_at", descending=True).limit(limit).collect().to_dicts()
    except Exception as exc:  # noqa: BLE001
        logger.debug("读取本地新闻失败: %s", exc)
        return []
