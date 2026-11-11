"""Bounded, resumable repair of missing daily market partitions."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import date, datetime, time
from pathlib import Path

import polars as pl

from app.parquet import atomic_write_parquet


def validate_partition(
    frame: pl.DataFrame, day: date, date_col: str, *, ohlc=False
) -> pl.DataFrame:
    if frame.is_empty() or date_col not in frame.columns:
        raise ValueError("empty response or missing date field")
    text = pl.col(date_col).cast(pl.String).str.replace_all("-", "").str.slice(0, 8)
    if frame.filter(text.is_null() | (text != day.strftime("%Y%m%d"))).height:
        raise ValueError("response dates do not match requested trading day")
    if ohlc:
        required = {"open", "high", "low", "close", "volume", "amount", "symbol"}
        if not required <= set(frame.columns):
            raise ValueError("incomplete OHLCV schema")
        p = [pl.col(c).cast(pl.Float64, strict=False) for c in ("open", "high", "low", "close")]
        bad = pl.any_horizontal([c.is_null() | ~c.is_finite() | (c <= 0) for c in p])
        bad |= (p[1] < pl.max_horizontal(p)) | (p[2] > pl.min_horizontal(p))
        for name in ("volume", "amount"):
            c = pl.col(name).cast(pl.Float64, strict=False)
            bad |= c.is_null() | ~c.is_finite() | (c < 0)
        frame = frame.filter(~bad)
        if frame.is_empty():
            raise ValueError("no valid tradable OHLCV rows")
        if frame["symbol"].n_unique() != frame.height:
            raise ValueError("ambiguous duplicate daily symbols")
    return frame


def _dates(base: Path) -> set[date]:
    result = set()
    for path in base.glob("date=*"):
        if not list(path.glob("*.parquet")):
            continue
        try:
            result.add(date.fromisoformat(path.name[5:]))
        except ValueError:
            continue
    return result


def append_missing_stocks(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    def keyed(frame):
        return frame.with_columns(
            pl.col("symbol").cast(pl.String).str.extract(r"(\d{6})", 1).alias("_code")
        )

    old = keyed(existing)
    new = keyed(incoming).filter(pl.col("_code").is_not_null())
    additions = new.join(old.select("_code").unique(), on="_code", how="anti").drop("_code")
    return pl.concat([existing, additions], how="diagonal_relaxed")


def daily_from_complete_minutes(frame: pl.DataFrame, day: date, symbols: list[str]) -> pl.DataFrame:
    from app.services.minute_repair import keyed, regular, PRICE

    frame = keyed(frame)
    allowed = {str(s).split(".")[0] for s in symbols}
    frame = regular(frame, day, include_opening=True).filter(pl.col("_code").is_in(allowed))
    frame = frame.sort("datetime")
    valid = pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in PRICE])
    valid &= (pl.col("high") >= pl.max_horizontal(PRICE)) & (
        pl.col("low") <= pl.min_horizontal(PRICE)
    )
    valid &= pl.all_horizontal(
        [pl.col(c).is_finite() & (pl.col(c) >= 0) for c in ("volume", "amount")]
    )
    grouped = frame.group_by("_code").agg(
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
        pl.col("amount").sum(),
        valid.fill_null(False).all().alias("_valid"),
        (pl.len() == pl.col("datetime").n_unique()).alias("_unique"),
    )
    counts = (
        regular(frame, day).group_by("_code").agg(pl.col("datetime").n_unique().alias("_count"))
    )
    grouped = grouped.join(counts, on="_code")
    vwap = pl.col("amount") / (pl.col("volume") * 100)
    tolerance = pl.max_horizontal(pl.lit(0.02), pl.col("high") * 0.002)
    grouped = grouped.filter(
        (pl.col("_count") == 240)
        & pl.col("_valid")
        & pl.col("_unique")
        & (pl.col("volume") > 0)
        & vwap.is_between(pl.col("low") - tolerance, pl.col("high") + tolerance)
    )
    return (
        grouped.rename({"_code": "symbol"})
        .with_columns(
            pl.lit(day).alias("date"),
            (pl.col("volume") * 100).alias("volume"),
            pl.lit("local_minute_aggregate").alias("data_source"),
        )
        .select("symbol", "date", *PRICE, "volume", "amount", "data_source")
    )


def repair_market_gaps(
    repo,
    start: date,
    end: date,
    *,
    max_requests=120,
    datasets: list[str] | None = None,
    on_progress=None,
    use_local_minutes=False,
) -> dict:
    from app.market_time import cn_now
    from app.services.tushare_ext_sync import (
        EXT_SPECS,
        _call_tushare,
        _resolve_tushare_provider,
        _temporary_env,
        _temporary_promax_timeout,
        _upsert_config,
    )

    if start > end or end >= cn_now().date():
        raise ValueError("repair accepts completed historical sessions only")
    emit = on_progress or (lambda *args, **kwargs: None)
    root = repo.store.data_dir
    run_id = "market_repair_" + uuid.uuid4().hex[:12]
    report_dir = root / "import_reports" / run_id
    report_dir.mkdir(parents=True)
    result = {
        "run_id": run_id,
        "start": str(start),
        "end": str(end),
        "requests": 0,
        "rows_added": 0,
        "partitions_added": 0,
        "partitions_repaired": 0,
        "datasets": [],
        "writes": [],
    }
    provider, name, error, mode = _resolve_tushare_provider()
    if provider is None and not use_local_minutes:
        raise RuntimeError(error)
    result["provider"] = "local_minute_aggregate" if use_local_minutes else name
    calendar = set()
    for name in ("kline_daily", "kline_etf_daily", "kline_index_daily", "kline_minute"):
        calendar |= _dates(root / name)
    calendar = {d for d in calendar if start <= d <= end}
    result["calendar_basis"] = "local cross-dataset union, not independent exchange calendar"
    inst = pl.read_parquet(root / "instruments" / "instruments.parquet")
    symbols = inst["symbol"].drop_nulls().unique().to_list()
    specs = EXT_SPECS
    wanted = datasets or ["daily", "moneyflow", "auction", *specs]
    known = {"daily", "moneyflow", "auction", *specs}
    if set(wanted) - known:
        raise ValueError("unknown repair datasets")
    if use_local_minutes and wanted != ["daily"]:
        raise ValueError("local minute aggregation can only fill daily gaps")

    def save():
        (report_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    with _temporary_env("TUSHARE_MODE", mode):
        api = None if use_local_minutes else provider._ensure_api()
        with _temporary_promax_timeout(api, 12):
            for dataset in wanted:
                spec = specs.get(dataset)
                base = root / {
                    "daily": "kline_daily",
                    "moneyflow": "moneyflow",
                    "auction": "auction_open",
                }.get(dataset, f"ext_data/{spec.config_id}/timeseries" if spec else "")
                missing = sorted(calendar - _dates(base), reverse=True)
                sparse = set()
                if dataset == "daily":
                    for day in calendar & _dates(base):
                        path = base / f"date={day}" / "part.parquet"
                        if (
                            path.exists()
                            and pl.read_parquet(path, columns=["symbol"]).height < 4500
                        ):
                            sparse.add(day)
                    missing = sorted(set(missing) | sparse, reverse=True)
                entry = {
                    "dataset": dataset,
                    "missing_before": len(missing),
                    "sparse_partitions": len(sparse),
                    "added_days": [],
                    "errors": [],
                }
                result["datasets"].append(entry)
                failures = 0
                for day in missing:
                    if result["requests"] >= max_requests:
                        break
                    result["requests"] += 1
                    emit(
                        "repair_market",
                        min(95, int(result["requests"] / max_requests * 95)),
                        f"增量补缺 {dataset} {day} ({result['requests']}/{max_requests})",
                    )
                    try:
                        a, b = datetime.combine(day, time.min), datetime.combine(day, time.max)
                        if use_local_minutes:
                            source = root / "kline_minute" / f"date={day}" / "part.parquet"
                            frame = daily_from_complete_minutes(
                                pl.read_parquet(source), day, symbols
                            )
                        elif dataset == "daily":
                            frame = provider._daily_by_trade_dates(
                                [],
                                day.strftime("%Y%m%d"),
                                day.strftime("%Y%m%d"),
                                trade_dates=[day],
                            )
                        elif dataset == "moneyflow":
                            frame = provider.get_moneyflow(symbols, a, b, trade_dates=[day])
                        elif dataset == "auction":
                            frame = provider.get_auction(symbols, a, b, trade_dates=[day])
                        else:
                            params = {spec.date_col: day.strftime("%Y%m%d")}
                            frame = _call_tushare(getattr(api, spec.api), spec, params)
                        if frame is None:
                            raise ValueError(
                                "empty response; gap retained (event table may have no events)"
                            )
                        frame = validate_partition(
                            frame, day, spec.date_col if spec else "date", ohlc=dataset == "daily"
                        )
                        if dataset in {"moneyflow", "auction", "top_list", "top_inst", "sw_daily", "ths_daily", "dc_daily", "cyq_perf", "stk_limit", "margin", "moneyflow_dc", "moneyflow_ths", "hsgt_top10", "ggt_top10"}:
                            frame = frame.unique(maintain_order=True)
                        # A partial market response must not turn a whole missing session into 'complete'.
                        if dataset == "daily" and frame.height < min(4500, int(len(symbols) * 0.8)):
                            raise ValueError(
                                f"sparse market response: {frame.height}/{len(symbols)} stocks"
                            )
                        target = base / f"date={day}" / "part.parquet"
                        before = 0
                        backup = None
                        with repo._write_lock:
                            if target.exists():
                                if day not in sparse:
                                    raise ValueError(
                                        "partition changed concurrently; not overwritten"
                                    )
                                existing = pl.read_parquet(target)
                                before = existing.height
                                frame = append_missing_stocks(existing, frame)
                                if frame.height == before:
                                    raise ValueError(
                                        "source supplied no additional historical stocks"
                                    )
                                backup = (
                                    root
                                    / "import_backups"
                                    / run_id
                                    / f"date={day}"
                                    / "part.parquet"
                                )
                                backup.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(target, backup)
                            atomic_write_parquet(frame, target)
                            check = pl.read_parquet(target)
                            if not check.equals(frame):
                                if backup:
                                    atomic_write_parquet(pl.read_parquet(backup), target)
                                raise IOError("readback mismatch")
                        if spec:
                            _upsert_config(root, spec)
                        result["writes"].append(
                            {
                                "path": str(target.relative_to(root)),
                                "rows": frame.height,
                                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                                "rows_before": before,
                                "backup": str(backup.relative_to(root)) if backup else None,
                                "operation": "append_missing_stocks" if before else "new_partition",
                            }
                        )
                        result["rows_added"] += frame.height - before
                        result["partitions_repaired" if before else "partitions_added"] += 1
                        entry["added_days"].append(str(day))
                        failures = 0
                    except Exception as exc:
                        # No raw URLs/credentials are persisted in the user-visible report.
                        entry["errors"].append(
                            {
                                "date": str(day),
                                "type": type(exc).__name__,
                                "reason": str(exc).split(" for url ")[0][:240],
                            }
                        )
                        failures += 1
                    save()
                    if failures >= 3 and not use_local_minutes:
                        entry["circuit_breaker"] = True
                        break
                entry["remaining_days"] = len(missing) - len(entry["added_days"])
                save()
    result["complete"] = all(not row["remaining_days"] for row in result["datasets"])
    result["api_requests"] = 0 if use_local_minutes else result["requests"]
    save()
    repo.rebuild_views()
    repo.clear_cache()
    repo.refresh_cache(background=True)
    emit(
        "done",
        100,
        f"新增 {result['partitions_added']} 个分区，修复 {result['partitions_repaired']} 个分区，"
        f"补齐 {result['rows_added']} 行；未通过校验的缺口保持未完成",
    )
    return result
