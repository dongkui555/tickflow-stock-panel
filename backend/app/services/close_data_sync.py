"""Small, durable post-close retries, independent of minute/history rebuilds."""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from app.market_time import cn_now

logger = logging.getLogger(__name__)
TABLES = {"index": "kline_index_daily", "etf": "kline_etf_daily",
          "moneyflow": "moneyflow", "auction_open": "auction_open"}
LABELS = {"index": "指数", "etf": "ETF", "moneyflow": "主力资金", "auction_open": "集合竞价"}
RETRY_MINUTES = 30
MAX_ATTEMPTS = 8
AUCTION_SLOW_RETRY_MINUTES = 180
ENHANCED_MIN_COVERAGE = max(0.0, min(1.0, float(os.getenv("CLOSE_DATA_MIN_COVERAGE", "0.99"))))
ERROR_MESSAGES = {
    "upstream_empty": "集合竞价接口返回成功但当日全市场为空。尚不能确定是未发布或网关缺数",
    "no_matching_rows": "接口未补到请求日期及缺失股票的记录。已有数据保留。不将缺失视为零成交",
    "network_timeout": "集合竞价接口连接或读取超时",
    "network_unavailable": "集合竞价接口网络连接失败",
    "permission_denied": "集合竞价接口明确返回未授权或拒绝访问",
    "rate_limited": "集合竞价接口触发限流。等待后重试",
    "gateway_rejected": "集合竞价网关返回HTTP错误",
    "invalid_response": "集合竞价响应日期、字段或分页校验未通过",
    "request_failed": "集合竞价请求失败。请核对接口配置及网关状态",
    "all_sources_failed": "集合竞价主、备接口均失败。请核对各数据源的失败原因",
}


def read_status(data_dir) -> dict:
    path = Path(data_dir) / "user_data" / "close_data_sync.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(data_dir, state):
    path = Path(data_dir) / "user_data" / "close_data_sync.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _days(data_dir, table, start, end):
    root = Path(data_dir) / table
    return sorted(date.fromisoformat(p.name[5:]) for p in root.glob("date=????-??-??")
                  if start.isoformat() <= p.name[5:] <= end.isoformat())


def _keys(data_dir, table, day, *, traded_only=False):
    files = list((Path(data_dir) / table / f"date={day}").glob("*.parquet"))
    if not files:
        return set()
    scan = pl.scan_parquet(files).filter(pl.col("date").cast(pl.Date) == day)
    if traded_only and "volume" in scan.collect_schema():
        # Zero-volume halted stocks need no flow record. Unknown volume is not
        # proof of a halt, so keep it in the expected universe.
        scan = scan.filter(pl.col("volume").is_null() | (pl.col("volume") != 0))
    frame = scan.select("symbol", "date").collect()
    return {str(s).split(".")[0] for s in frame["symbol"].drop_nulls().to_list()}


def coverage(data_dir, dataset, day):
    table = TABLES[dataset]
    actual = _keys(data_dir, table, day)
    if dataset in ("moneyflow", "auction_open"):
        expected = _keys(data_dir, "kline_daily", day, traded_only=True)
    elif dataset == "index":
        from app.services.index_const import CORE_INDEX_SYMBOLS
        expected = {symbol.split(".")[0] for symbol in CORE_INDEX_SYMBOLS}
    else:
        previous = _days(data_dir, table, day - timedelta(days=14), day - timedelta(days=1))
        expected = _keys(data_dir, table, previous[-1]) if previous else set()
    missing = sorted(expected - actual)
    strict_complete = bool(expected) and not missing
    ratio = len(actual & expected) / len(expected) if expected else 0.0
    # Money-flow and auction vendors legitimately omit halted, newly listed or
    # unsupported-market instruments. A near-complete market partition is fit
    # for scoring and must not retry forever; exact holes remain visible.
    usable = strict_complete or (
        dataset in ("moneyflow", "auction_open")
        and ratio >= ENHANCED_MIN_COVERAGE
    )
    return {"target_date": day.isoformat(), "actual_symbols": len(actual),
            "expected_symbols": len(expected), "missing_symbols": missing,
            "coverage_ratio": round(ratio, 6), "strict_complete": strict_complete,
            "complete": usable}


def pending_tasks(data_dir, now, schedule, trading_day):
    """Calendar gate + source publication window + per-dataset backoff."""
    if trading_day is not True or now.weekday() >= 5:
        return []
    due = now.replace(hour=schedule["hour"], minute=schedule["minute"], second=0, microsecond=0)
    end = now.date() if now >= due else now.date() - timedelta(days=1)
    if now.hour < 8:
        return []
    state = read_status(data_dir)
    pending = []
    for day in reversed(_days(data_dir, "kline_daily", now.date() - timedelta(days=7), end)[-3:]):
        for dataset in TABLES:
            # Official moneyflow is published at 19:00; gateways can lag further.
            if dataset == "moneyflow" and day == now.date() and now.hour < 19:
                continue
            item = state.get(day.isoformat(), {}).get(dataset, {})
            if item.get("status") == "succeeded":
                continue
            # Let the worker reconcile legacy failed status from local data
            # under the current coverage contract without another API call.
            if coverage(data_dir, dataset, day)["complete"]:
                pending.append((day, dataset))
                continue
            if dataset != "auction_open" and item.get("attempts", 0) >= MAX_ATTEMPTS:
                continue
            if item.get("retry_at") and datetime.fromisoformat(item["retry_at"]) > now:
                continue
            pending.append((day, dataset))
    return pending


def _fetch_missing(repo, capset, dataset, day, missing, emit):
    from app.data_providers import custom
    from app.services import index_sync, preferences

    start = datetime.combine(day, datetime.min.time())
    end = datetime.combine(day, datetime.max.time())
    if dataset in ("index", "etf"):
        instruments = repo.get_index_instruments() if dataset == "index" else repo.get_etf_instruments()
        symbols = [s for s in instruments["symbol"].to_list() if str(s).split(".")[0] in missing]
        # Bound a retry's work. Remaining indexes are checkpointed for the next run.
        if dataset == "index":
            from app.services.index_const import CORE_INDEX_SYMBOLS
            symbols = sorted(symbols, key=lambda s: (s not in CORE_INDEX_SYMBOLS, s))[:80]
        if not symbols:
            return 0
        fn = index_sync.sync_and_persist_index_daily if dataset == "index" else index_sync.sync_and_persist_etf_daily
        return fn(repo, capset, start_date=start, end_date=end, symbols_override=symbols,
                  on_chunk_done=lambda i, n: emit("close_data", 50, f"{LABELS[dataset]} {day} 批次 {i}/{n}"))

    name = preferences.get_daily_data_provider()
    feature = "moneyflow" if dataset == "moneyflow" else "auction"
    if not custom.provider_has_dataset(name, feature):
        raise ValueError(f"{name} 未配置 {feature} 接口")
    provider = custom.get_provider(name)
    method = provider.get_moneyflow if dataset == "moneyflow" else provider.get_auction
    kwargs = {"market": "o"} if dataset == "auction_open" else {}
    # Bulk date request remains one market request, not thousands of stock requests.
    import inspect
    if "trade_dates" in inspect.signature(method).parameters:
        kwargs["trade_dates"] = [day]
    if dataset == "auction_open" and "strict" in inspect.signature(method).parameters:
        kwargs["strict"] = True
    primary_error = None
    try:
        frame = method(sorted(missing), start_time=start, end_time=end, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if dataset != "auction_open":
            raise
        primary_error = exc
        frame = pl.DataFrame()
    # Historical auction remains a Promax responsibility. For the current day,
    # an authenticated iQuant ContextInfo collector can supply the 09:15-09:25
    # snapshot captured locally when the gateway is empty or unavailable.
    if dataset == "auction_open" and day == cn_now().date() and (frame is None or frame.is_empty()):
        try:
            if str(os.getenv("QMT_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}:
                from app.plugins.qmt.provider import QmtProvider

                qmt = QmtProvider()
                try:
                    frame = qmt.get_auction(
                        sorted(missing), start_time=start, end_time=end,
                        trade_dates=[day], strict=False,
                    )
                finally:
                    qmt.close()
        except Exception as exc:  # noqa: BLE001
            logger.info("QMT auction supplement unavailable for %s: %s", day, type(exc).__name__)
    if (frame is None or frame.is_empty()) and primary_error is not None:
        raise primary_error
    if frame is None or frame.is_empty():
        return 0
    frame = frame.filter((pl.col("date").cast(pl.Date) == day)
                         & pl.col("symbol").cast(pl.String).str.split(".").list.first().is_in(sorted(missing)))
    if frame.is_empty():
        return 0
    persist = repo.append_moneyflow if dataset == "moneyflow" else repo.append_auction_open
    persist(frame)
    return frame.height


def run_close_data_sync(repo, capset, tasks, on_progress=None, now=None):
    """Call only while holding the pipeline execution slot; never overwrite history."""
    emit = on_progress or (lambda *args, **kwargs: None)
    now = now or cn_now()
    state = read_status(repo.store.data_dir)
    results = []
    for day, dataset in tasks:
        item = state.setdefault(day.isoformat(), {}).setdefault(dataset, {})
        emit("close_data", 0, f"核对{LABELS[dataset]} {day} 实际日期与覆盖")
        before = coverage(repo.store.data_dir, dataset, day)
        rows, error, error_reason = 0, None, None
        error_sources = []
        if not before["complete"]:
            item["attempts"] = item.get("attempts", 0) + 1
            item.update(status="running", started_at=now.isoformat())
            _save(repo.store.data_dir, state)
            try:
                rows = _fetch_missing(repo, capset, dataset, day, set(before["missing_symbols"]), emit)
            except Exception as exc:
                from app.services.pipeline_jobs import JobCancelledError
                if isinstance(exc, JobCancelledError):
                    raise
                error = type(exc).__name__
                reason = getattr(exc, "reason", None)
                if dataset == "auction_open" and isinstance(reason, str) and reason in ERROR_MESSAGES:
                    error_reason = reason
                    error_sources = [
                        {'source': failure['source'], 'reason': failure['reason']}
                        for failure in getattr(exc, 'source_failures', [])
                        if isinstance(failure, dict)
                        and failure.get('source') in {'official', 'promax', 'datahub'}
                        and failure.get('reason') in ERROR_MESSAGES
                    ]
                logger.warning("close data %s %s failed: %s", dataset, day, error)
        after = coverage(repo.store.data_dir, dataset, day)
        # Progress across bounded index batches is not an upstream failure attempt.
        if rows and len(after["missing_symbols"]) < len(before["missing_symbols"]):
            item["attempts"] = 0
        complete = after["complete"]
        if dataset == "auction_open" and not complete and not rows and not error:
            error_reason = "no_matching_rows"
        error_message = ERROR_MESSAGES.get(error_reason)
        retry_minutes = RETRY_MINUTES
        if dataset == "auction_open" and item.get("attempts", 0) >= MAX_ATTEMPTS:
            retry_minutes = AUCTION_SLOW_RETRY_MINUTES
        item.update(after, status="succeeded" if complete else "retry", rows_written=rows,
                    checked_at=cn_now().isoformat(), error=error,
                    error_reason=error_reason, error_message=error_message,
                    error_sources=error_sources,
                    data_state="complete" if complete else "partial" if after["actual_symbols"] else "empty",
                    retry_at=(cn_now() + timedelta(minutes=retry_minutes)).isoformat())
        if not complete and item.get("attempts", 0) >= MAX_ATTEMPTS:
            item["status"] = "needs_attention"
        _save(repo.store.data_dir, state)
        emit("close_data", 90, f"{LABELS[dataset]} {day}：已覆盖 {after['actual_symbols']}，缺 {len(after['missing_symbols'])}，{item['status']}")
        if error_message:
            emit("close_data", 90, f"{error_message}。{retry_minutes}分钟后补查")
        results.append({"dataset": dataset, **item})
    from app.api.data import invalidate_storage_cache
    invalidate_storage_cache()
    repo.refresh_index_views()
    emit("done", 100, "盘后分项补差完成，未齐数据按分项状态继续重试")
    return {"scheduled_mode": "close_dataset_incremental", "datasets": results,
            "complete": all(r["complete"] for r in results)}
