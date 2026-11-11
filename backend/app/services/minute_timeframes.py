"""Session-aligned bars from the canonical China-time, end-labelled minute store."""
from __future__ import annotations

from datetime import datetime, time, timedelta

import polars as pl

PERIODS = (5, 15, 30, 60)
PRICE = ("open", "high", "low", "close")


def aggregate_minutes(frame: pl.DataFrame, period: int, as_of: datetime) -> pl.DataFrame:
    if period not in PERIODS:
        raise ValueError("period must be 5, 15, 30 or 60")
    if frame.is_empty():
        return pl.DataFrame()
    required = {"symbol", "datetime", *PRICE, "volume", "amount"}
    if not required.issubset(frame.columns):
        raise ValueError("Minute OHLC, volume or amount fields are missing")
    stamp_type = frame.schema["datetime"]
    if not isinstance(stamp_type, pl.Datetime):
        raise ValueError("Minute timestamps must be normalized Datetime")
    if stamp_type.time_zone:
        frame = frame.with_columns(pl.col("datetime").dt.convert_time_zone("Asia/Shanghai").dt.replace_time_zone(None))
    if as_of.tzinfo:
        from zoneinfo import ZoneInfo
        as_of = as_of.astimezone(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
    frame = frame.filter(pl.col("datetime") <= as_of).sort(["symbol", "datetime"])
    clock = pl.col("datetime").dt.hour().cast(pl.Int32) * 60 + pl.col("datetime").dt.minute()
    frame = frame.with_columns(clock.alias("_clock")).filter(
        clock.is_between(570, 690) | clock.is_between(781, 900))
    if frame.is_empty():
        return pl.DataFrame()
    valid = pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in PRICE])
    valid &= (pl.col("high") >= pl.max_horizontal(PRICE)) & (pl.col("low") <= pl.min_horizontal(PRICE))
    valid &= pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) >= 0) for c in ("volume", "amount")])
    valid &= (pl.col("datetime").dt.second() == 0) & (pl.col("datetime").dt.microsecond() == 0)
    # A separate 09:30 auction belongs to the first morning bar, not an extra candle.
    frame = frame.with_columns(
        pl.when(pl.col("_clock") <= 690).then(570).otherwise(780).alias("_session"),
        valid.fill_null(False).alias("_valid"),
    ).with_columns(
        ((pl.col("_clock") - pl.col("_session") - 1).clip(lower_bound=0) // period).alias("_bin")
    ).with_columns(
        (pl.col("datetime").dt.date().cast(pl.Datetime("us"))
         + pl.duration(minutes=pl.col("_session") + (pl.col("_bin") + 1) * period)).alias("_end"))
    bars = frame.group_by("symbol", "_end", maintain_order=True).agg(
        pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(),
        pl.col("volume").sum(), pl.col("amount").sum(),
        pl.col("datetime").max().alias("observed_at"),
        (pl.col("_clock") != 570).sum().alias("source_minutes"),
        pl.col("_clock").filter(pl.col("_clock") != 570).min().alias("_first"),
        pl.col("_valid").all().alias("_valid"),
        (pl.col("datetime").n_unique() == pl.len()).alias("_unique"),
    ).rename({"_end": "datetime"})
    end_clock = pl.col("datetime").dt.hour().cast(pl.Int32) * 60 + pl.col("datetime").dt.minute()
    observed_clock = pl.col("observed_at").dt.hour().cast(pl.Int32) * 60 + pl.col("observed_at").dt.minute()
    bars = bars.with_columns(
        (pl.col("_valid") & pl.col("_unique") & (pl.col("source_minutes") == period)
         & (pl.col("datetime") <= as_of)).alias("complete"),
        ((pl.col("_first") == end_clock - period + 1)
         & (pl.col("source_minutes") == observed_clock - pl.col("_first") + 1)).alias("_prefix"),
    ).with_columns(
        pl.when(~pl.col("_valid") | ~pl.col("_unique")).then(pl.lit("invalid"))
        .when(pl.col("complete")).then(pl.lit("complete"))
        .when((pl.col("datetime") > as_of) & pl.col("_prefix")).then(pl.lit("forming"))
        .otherwise(pl.lit("gap")).alias("quality"))
    return bars.drop("_valid", "_unique", "_first", "_prefix").sort(["symbol", "datetime"])


def five_minute_confirmation(bars: pl.DataFrame, now: datetime) -> dict:
    """Conservative execution gate, not a prediction or an order instruction."""
    if now.tzinfo:
        from app.market_time import CN_TZ
        now = now.astimezone(CN_TZ).replace(tzinfo=None)
    result = {"status": "missing", "passed": False, "bar_end": None, "label": "缺少5分钟数据"}
    if now.weekday() >= 5 or not (time(9, 30) <= now.time() < time(11, 30) or time(13) <= now.time() < time(15)):
        return {**result, "status": "closed", "label": "等待交易时段5分钟确认"}
    if bars.is_empty():
        return result
    prior = bars.filter((pl.col("datetime").dt.date() == now.date()) & (pl.col("datetime") <= now)).sort("datetime").tail(3)
    if prior.height < 3:
        return result
    end = prior["datetime"][-1]
    result["bar_end"] = end.isoformat()
    if now - end > timedelta(minutes=6):
        return {**result, "status": "stale", "label": "5分钟数据滞后"}
    if not prior["complete"].all() or any(b - a != timedelta(minutes=5) for a, b in zip(prior["datetime"], prior["datetime"].to_list()[1:])):
        return {**result, "label": "5分钟数据不连续"}
    latest = prior.row(-1, named=True)
    earlier = prior.head(2)
    price_ok = latest["close"] > earlier["high"].max() and latest["close"] > latest["open"]
    average_volume = earlier["volume"].mean()
    volume_ok = average_volume > 0 and latest["volume"] >= average_volume
    passed = price_ok and volume_ok
    return {**result, "passed": passed, "status": "confirmed" if passed else "waiting",
            "label": "5分钟量价确认通过" if passed else "等待5分钟放量突破",
            "price_pass": price_ok, "volume_pass": volume_ok}


def load_five_minute_confirmations(repo, symbols: list[str], now: datetime) -> dict:
    if not symbols:
        return {}
    empty = five_minute_confirmation(pl.DataFrame(), now)
    if empty["status"] == "closed":
        return {symbol: dict(empty) for symbol in symbols}
    frame = repo.get_minute_range(symbols, now.date(), now.date())
    bars = aggregate_minutes(frame, 5, now)
    if bars.is_empty():
        return {}
    return {key[0]: five_minute_confirmation(part, now)
            for key, part in bars.partition_by("symbol", as_dict=True).items()}


def chart_minutes(repo, symbol: str, asset_type: str, period: int, days: int, now: datetime) -> dict:
    if asset_type == "index":
        return {"rows": [], "warning": "指数分钟K尚未本地归档", "excluded_bars": 0}
    frame = repo.get_minute_range([symbol], now.date() - timedelta(days=days * 3 + 20), now.date(), asset_type)
    bars = aggregate_minutes(frame, period, now)
    if bars.is_empty():
        return {"rows": [], "warning": "本地暂无分钟K，请先同步分钟数据", "excluded_bars": 0}
    dates = sorted(bars["datetime"].dt.date().unique())[-days:]
    bars = bars.filter(pl.col("datetime").dt.date().is_in(dates))
    excluded = bars.filter(~pl.col("quality").is_in(["complete", "forming"])).height
    # Indicators must not bridge missing bars; recompute within each uninterrupted segment.
    times = bars["datetime"].to_list()
    breaks = []
    for i, stamp in enumerate(times):
        prior = times[i - 1] if i else None
        expected = prior + timedelta(minutes=period) if prior else stamp
        if prior and prior.time() == time(11, 30):
            expected = datetime.combine(prior.date(), time(13)) + timedelta(minutes=period)
        breaks.append(bool(prior and stamp.date() == prior.date() and stamp != expected))
    bars = bars.with_columns(pl.Series("_break", breaks)).with_columns(
        ((~pl.col("complete")) | pl.col("_break")).cast(pl.Int64).cum_sum().alias("_segment")
    ).drop("_break")
    bars = bars.filter(pl.col("quality").is_in(["complete", "forming"]))
    from app.indicators.pipeline import compute_indicators
    needed = {"ma5", "ma10", "ma20", "ma60", "macd_dif", "macd_dea", "macd_hist",
              "rsi_6", "rsi_14", "rsi_24", "kdj_k", "kdj_d", "kdj_j", "boll_upper", "boll_lower"}
    pieces = [compute_indicators(part.rename({"datetime": "date"}), needed=needed)
              for part in bars.partition_by("_segment")]
    rows = pl.concat(pieces).drop("_segment").to_dicts() if pieces else []
    return {"rows": rows,
            "excluded_bars": excluded, "warning": f"已排除{excluded}根缺失或异常K线" if excluded else None,
            "forming_bars": bars.filter(pl.col("quality") == "forming").height}
