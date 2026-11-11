"""Separate short daily work from historical derivation and minute gap filling."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import os

from app.services import maintenance_queue


def partition_versions(data_dir, table):
    versions = {}
    for folder in (data_dir / table).glob("date=????-??-??"):
        try:
            day = date.fromisoformat(folder.name[5:])
            versions[day] = [(p.name, p.stat().st_mtime_ns, p.stat().st_size)
                             for p in sorted(folder.glob("*.parquet"))]
        except (ValueError, OSError):
            continue
    return versions


def update_recent_enriched(repo, before, affected_symbols, today, emit):
    from app.indicators.pipeline import run_date_pipeline
    from app.jobs.daily_pipeline import _prune_partial_enriched_partitions, _prune_stale_price_partitions

    root = repo.store.data_dir
    after = partition_versions(root, "kline_daily")
    days = sorted(d for d in after if d <= today)
    enriched = partition_versions(root, "kline_daily_enriched")
    dirty = {d for d in days if before.get(d) != after[d]}
    # A corrected old close also affects the subsequent indicator warmup window.
    impacted = set()
    for i, d in enumerate(days):
        if d in dirty:
            impacted.update(days[i:i + 121])
    recent = set(days[-5:])
    missing = set(days) - set(enriched)
    # Keep the published partition available until its queued replacement commits.
    invalid = {
        date.fromisoformat(d) for d in (
            _prune_partial_enriched_partitions(root / "kline_daily", root / "kline_daily_enriched", prune=False)
            + _prune_stale_price_partitions(root / "kline_daily", root / "kline_daily_enriched",
                                             max_dates=None, prune=False)
        )
    } & set(days)
    targets = (impacted | missing | invalid | (recent if affected_symbols else set())) & recent
    historical = (impacted | missing | invalid) - recent
    maintenance_queue.enqueue(root, "enriched_dates", historical)
    written = 0
    for i, day in enumerate(sorted(targets)):
        emit("compute_enriched", 65 + int(23 * i / max(1, len(targets))),
             f"当期指标 {i + 1}/{len(targets)}: {day}", stage_pct=int(100 * i / len(targets)))
        rows = run_date_pipeline(root, day)
        if not rows:
            raise RuntimeError(f"当期指标未生成: {day}")
        written += rows
    emit("compute_enriched", 88,
         f"当期指标更新 {len(targets)} 天; 历史补算 {len(historical)} 天已排队", stage_pct=100)
    return written, sorted(targets)


def queue_regime_history(repo, recent_dates, today):
    from app.services import regime_builder

    root = repo.store.data_dir
    available = regime_builder.enriched_date_set(repo)
    old = regime_builder.load_regime_history(root)
    existing = set(old["date"].to_list()) if not old.is_empty() else set()
    missing = available - existing
    stale = set(regime_builder.detect_stale_dates(root, repo))
    recent = set(recent_dates)
    maintenance_queue.enqueue(root, "regime", {d for d in stale - recent if d <= today},
                              reason="source_change_unverified")
    dates = {d for d in missing - recent if d <= today}
    return maintenance_queue.enqueue(root, "regime", dates, reason="missing_result")


def minute_validation_report(repo, items, start, end):
    import polars as pl

    from app.services import kline_sync, minute_failover
    from app.services.minute_repair import (
        _has_explicit_daily_identity,
        checked_additions,
        daily_reference,
        keyed,
        regular,
    )
    from app.services.market_status import historical_suspension_status, market_day_status
    from app.services.security_lifecycle import load_lifecycle_index, validate_security_lifecycle

    root = repo.store.data_dir
    lifecycle_index = load_lifecycle_index(root)
    # Minute storage and the durable gap queue use the six-digit security code
    # as their internal key.  Keep accepting market-qualified inputs at the
    # task boundary, but do not let ``920925.BJ`` miss the matching daily row
    # for ``920925`` during validation.
    requested = {
        kline_sync._normalize_sync_symbol(s).split(".", 1)[0]
        for s in items
    }
    listings = {}
    instrument_path = root / "instruments" / "instruments.parquet"
    if instrument_path.exists():
        instruments = pl.read_parquet(instrument_path)
        if {"symbol", "listing_date"}.issubset(instruments.columns):
            parsed = keyed(instruments).with_columns(pl.coalesce(
                pl.col("listing_date").cast(pl.String).str.to_date("%Y%m%d", strict=False),
                pl.col("listing_date").cast(pl.String).str.to_date("%Y-%m-%d", strict=False),
            ).alias("_listed")).filter(pl.col("_listed").is_not_null())
            for key, part in parsed.partition_by("_code", as_dict=True).items():
                if part["_listed"].n_unique() == 1:
                    listings[key[0]] = part["_listed"][0]
    dates = {d for d in kline_sync._partition_dates(root / "kline_daily") if start.date() <= d <= end.date()}
    dates.add(end.date())
    failures = []
    statuses = []
    for day in sorted(dates):
        full = datetime.combine(day, datetime.min.time()).replace(hour=15) <= end
        ref_path = root / "kline_daily" / f"date={day}" / "part.parquet"
        daily = pl.read_parquet(ref_path) if ref_path.exists() else pl.DataFrame()
        refs = daily_reference(daily, data_dir=root) if not daily.is_empty() else {}
        no_trades = set(keyed(daily).filter(
            (pl.col("volume") == 0) & (pl.col("amount") == 0))["_code"]) if not daily.is_empty() else set()
        daily_codes = set(refs) | no_trades
        market_status = market_day_status(root, day)
        path = root / "kline_minute" / f"date={day}" / "part.parquet"
        frame = keyed(pl.read_parquet(path)).filter(pl.col("_code").is_in(requested)) if path.exists() else pl.DataFrame()
        groups = frame.partition_by("_code", as_dict=True) if not frame.is_empty() else {}
        begin = datetime.combine(day, datetime.min.time()).replace(hour=9, minute=31)
        cutoff = min(end, begin.replace(hour=15, minute=0))
        for code in sorted(requested):
            if market_status["status"] == "market_closed":
                statuses.append({
                    "symbol": code, "date": str(day), "status": "market_closed",
                    "expected_minutes": 0, "actual_minutes": 0,
                    "reason": market_status["reason"],
                })
                continue
            if lifecycle_index:
                lifecycle = validate_security_lifecycle(
                    code, day, root, index=lifecycle_index,
                )
                if not lifecycle.ok:
                    # A live-only lifecycle dimension can omit a valid
                    # historical symbol. An exact, market-qualified daily
                    # reference is sufficient identity evidence for this
                    # stock-day; it does not relax market conflicts or
                    # invent an exchange for a bare row.
                    if (
                        lifecycle.reason == "identity_history_unknown"
                        and (
                            _has_explicit_daily_identity(code, refs.get(code))
                            # No daily row means there is no date-scoped
                            # reference to validate yet. Let the normal
                            # daily_reference_missing branch classify it.
                            or code not in refs
                        )
                    ):
                        pass
                    else:
                        lifecycle_status_row = {
                            "symbol": code, "date": str(day), "status": {
                                "not_listed_yet": "pre_listing",
                                "delisted_on_date": "delisted",
                                "code_changed": "code_changed",
                                "market_identity_conflict": "identity_conflict",
                                "identity_history_unknown": "identity_unknown",
                                "lifecycle_date_missing": "identity_unknown",
                            }.get(lifecycle.reason, "identity_unknown"),
                            "expected_minutes": 0, "actual_minutes": 0,
                            "reason": lifecycle.reason,
                            "lifecycle": lifecycle.as_dict(),
                        }
                        statuses.append(lifecycle_status_row)
                        failures.append({
                            "symbol": code, "date": str(day), "reason": lifecycle.reason,
                            "lifecycle": lifecycle.as_dict(),
                        })
                        continue
            if code in listings and day < listings[code]:
                statuses.append({"symbol": code, "date": str(day), "status": "pre_listing",
                                 "expected_minutes": 0, "actual_minutes": 0,
                                 "reason": "上市前无行情"})
                continue
            suspension = historical_suspension_status(root, code, day)
            if suspension["status"] == "suspended":
                if code in no_trades or code not in daily_codes:
                    statuses.append({
                        "symbol": code, "date": str(day), "status": "suspended",
                        "expected_minutes": 0, "actual_minutes": 0,
                        "reason": suspension["reason"],
                    })
                    continue
                statuses.append({
                    "symbol": code, "date": str(day),
                    "status": "suspension_data_conflict",
                    "expected_minutes": 0, "actual_minutes": 0,
                    "reason": "历史停牌记录与日K成交冲突",
                })
                failures.append({
                    "symbol": code, "date": str(day),
                    "reason": "suspension_data_conflict",
                    "suspension": suspension,
                })
                continue
            # A daily record with no trades is not a missing 240-minute session.
            if full and code in no_trades:
                statuses.append({"symbol": code, "date": str(day), "status": "no_trade",
                                 "expected_minutes": 0, "actual_minutes": 0,
                                 "reason": "日K成交量和成交额均为0"})
                continue
            part = groups.get((code,), pl.DataFrame())
            if not part.is_empty():
                part = regular(part.drop("_code"), day, include_opening=True).filter(pl.col("datetime") <= cutoff)
            if full and code not in refs:
                reason = "daily_reference_missing"
            elif full:
                _, reason = checked_additions(
                    part, part.clear(), refs[code], day, source="local"
                )
            else:
                reason = minute_failover._intraday_reason(part, day, begin, cutoff)
            if reason in {"accepted", "already_complete"}:
                statuses.append({"symbol": code, "date": str(day), "status": "complete",
                                 "expected_minutes": 240 if full else int(regular(part, day).height),
                                 "actual_minutes": int(regular(part, day).height),
                                 "reason": "本地分钟完整且通过日K量价校验"})
                continue
            actual_minutes = int(regular(part, day).height) if not part.is_empty() else 0
            statuses.append({"symbol": code, "date": str(day), "status": "unknown",
                             "quality_status": "partial" if actual_minutes else reason,
                             "expected_minutes": 240 if full else int(len(minute_failover._expected_minutes(day, begin, cutoff))),
                             "actual_minutes": actual_minutes, "reason": reason})
            failures.append({"symbol": code, "date": str(day), "reason": reason,
                             **minute_failover._evidence(part, day, begin, cutoff, refs.get(code))})
    return failures, statuses


def minute_validation_failures(repo, items, start, end):
    """Compatibility wrapper returning only non-terminal validation failures."""
    return minute_validation_report(repo, items, start, end)[0]


def annotate_minute_failures(failures, issues):
    from app.services.minute_failover import SOURCE_FAILURES, _safe_detail

    for failure in failures:
        matching = [i for i in issues if i.get("symbol", failure["symbol"]).split(".")[0] == failure["symbol"]
                    and i.get("date", failure["date"]) == failure["date"]]
        if not matching:
            continue
        failure["source_errors"] = [{"source": i.get("source", "primary"), **_safe_detail(i)} for i in matching]
        # A local hole is a separate observation, not proof of an empty upstream response.
        if failure["reason"] not in {"source_empty", "source_incomplete"}:
            continue
        source_errors = [i for i in matching if i.get("reason") in SOURCE_FAILURES | {"not_requested"}]
        if source_errors and matching[-1].get("reason") != "accepted":
            failure["data_reason"] = failure["reason"]
            primary = next((i for i in source_errors if i.get("source") == "promax"), source_errors[-1])
            failure["reason"] = primary.get("cause", primary["reason"])


def _minute_retry_policy(day, now, attempts):
    """Allow only the documented current-day retry windows."""
    max_attempts = max(1, int(os.getenv("MINUTE_CURRENT_DAY_MAX_RETRIES", "3")))
    attempts = max(0, int(attempts or 0))
    if day < now.date():
        return attempts < max_attempts, 0, attempts >= max_attempts
    if day > now.date():
        return False, 3600, False
    close = now.replace(hour=15, minute=10, second=0, microsecond=0)
    late = now.replace(hour=22, minute=0, second=0, microsecond=0)
    if now < close:
        return False, max(60, int((close - now).total_seconds())), False
    if now < late:
        if attempts < 1:
            return True, 0, False
        return False, max(60, int((late - now).total_seconds())), False
    if attempts < 2:
        return True, 0, False
    next_morning = datetime.combine(day + timedelta(days=1), datetime.min.time()).replace(
        hour=8, minute=30)
    if attempts < max_attempts:
        return False, max(60, int((next_morning - now).total_seconds())), False
    return False, 0, True


def _all_sources_unsupported(failure):
    errors = failure.get("source_errors") or []
    sources = {str(item.get("source") or "") for item in errors if item.get("source")}
    reasons = {str(item.get("reason") or item.get("cause") or "") for item in errors}
    return len(sources) >= 2 and bool(reasons) and reasons <= {"source_unsupported"}


def _cross_source_local_evidence(repo, symbol: str, day: date) -> dict:
    """Build the local side of the third-party stock-day decision."""
    import polars as pl

    from app.services import kline_sync
    from app.services.minute_repair import _daily_aggregate_exprs, daily_reference, keyed, regular
    from app.services.security_identity import exchange_for_code

    root = repo.store.data_dir
    code = "".join(ch for ch in str(symbol or "") if ch.isdigit())[-6:].zfill(6)
    daily_path = root / "kline_daily" / f"date={day}" / "part.parquet"
    if not daily_path.exists():
        return {"symbol": str(symbol).upper(), "local_daily": None, "minute_aggregate": None}
    daily = pl.read_parquet(daily_path)
    # Cross-source validation compares the published full-day baseline.  The
    # regular-session evidence used by minute import is deliberately not
    # applied here because it excludes a different trading-time scope.
    refs = daily_reference(daily)
    reference = refs.get(code)
    if reference is None:
        return {"symbol": str(symbol).upper(), "local_daily": None, "minute_aggregate": None}
    requested = kline_sync._normalize_sync_symbol(symbol)
    canonical = str(reference.get("symbol") or requested).upper()
    if "." not in canonical:
        exchange = exchange_for_code(code, "stock")
        canonical = f"{code}.{exchange}" if exchange else requested
    local_daily = {
        **reference,
        "symbol": canonical,
        "date": day.isoformat(),
        "volume_unit": str(reference.get("volume_unit") or "unknown").lower(),
        "volume_unit_basis": "explicit" if reference.get("volume_unit") else "unresolved",
    }

    minute_path = root / "kline_minute" / f"date={day}" / "part.parquet"
    if not minute_path.exists():
        return {"symbol": canonical, "local_daily": local_daily, "minute_aggregate": None}
    frame = pl.read_parquet(minute_path)
    code_frame = frame.filter(
        pl.col("symbol").cast(pl.String).str.extract(r"(\d{6})", 1) == code
    )
    raw_symbols = {
        str(value).strip().upper()
        for value in code_frame.get_column("symbol").cast(pl.String).unique().to_list()
        if str(value).strip()
    }
    invalid_identities = {
        value for value in raw_symbols
        if "." in value and value != canonical
    }
    if invalid_identities:
        return {
            "symbol": canonical,
            "local_daily": local_daily,
            "minute_aggregate": None,
            "identity_conflict": sorted(invalid_identities),
        }
    normalized_symbols = {
        str(value).strip().upper()
        for value in code_frame.get_column("symbol").cast(pl.String).unique().to_list()
    }
    # Prefer one explicit market-qualified identity.  A bare-code alias is
    # only a fallback when the canonical row is absent; never concatenate both
    # representations into one stock-day aggregate.
    selected_identity = canonical if canonical in normalized_symbols else code
    frame = code_frame.filter(
        pl.col("symbol").cast(pl.String).str.to_uppercase() == selected_identity
    )
    tagged = keyed(frame)
    part = tagged.filter(pl.col("_code") == code).drop("_code")
    if part.is_empty():
        return {"symbol": canonical, "local_daily": local_daily, "minute_aggregate": None}
    part = regular(part, day, include_opening=True, include_bj_after_hours=True).sort("datetime")
    if part.is_empty():
        return {"symbol": canonical, "local_daily": local_daily, "minute_aggregate": None}
    aggregate = part.select(_daily_aggregate_exprs()).row(0, named=True)
    minute_aggregate = {
        "symbol": canonical,
        "date": day.isoformat(),
        **{field: aggregate.get(field) for field in ("open", "high", "low", "close", "volume", "amount")},
        "rows": int(regular(part, day).height),
        # The normalized local minute store uses lots, including QMT imports.
        "volume_unit": "lots",
    }
    return {"symbol": canonical, "local_daily": local_daily,
            "minute_aggregate": minute_aggregate}


def _cross_source_verify_failures(repo, failures, *, max_items: int = 50) -> list[dict]:
    """Run the bounded Promax/Tushare -> Eastmoney verification chain."""
    from app.services.minute_cross_source import collect_daily_snapshots, verify_stock_day

    skip_reasons = {
        "daily_reference_missing", "market_closed", "suspended", "pre_listing", "delisted", "code_changed",
        "identity_unknown", "identity_conflict", "no_trade",
    }
    selected = [item for item in failures if item.get("reason") not in skip_reasons][:max_items]
    if not selected:
        return []
    provider = None
    fuyao_client = None
    try:
        from app.plugins.tushare.provider import TushareProvider
        provider = TushareProvider()
    except Exception:
        provider = None
    try:
        from app.plugins.fuyao.client import FuyaoClient
        from app.plugins.fuyao.provider import get_api_key
        api_key = get_api_key()
        if api_key:
            fuyao_client = FuyaoClient(api_key=api_key, timeout=15.0)
    except Exception:
        fuyao_client = None
    results: list[dict] = []
    try:
        for failure in selected:
            day = date.fromisoformat(str(failure["date"])[:10])
            context = _cross_source_local_evidence(repo, failure["symbol"], day)
            symbol = context["symbol"]
            snapshots, source_errors = collect_daily_snapshots(
                symbol, day, promax_provider=provider, fuyao_client=fuyao_client,
            )
            result = verify_stock_day(
                symbol=symbol,
                trade_date=day,
                local_daily=context.get("local_daily"),
                minute_aggregate=context.get("minute_aggregate"),
                external_snapshots=snapshots,
                source_errors=[
                    *source_errors,
                    *([{"source": "local", "reason": "identity_conflict", "retryable": False}]
                      if context.get("identity_conflict") else []),
                ],
                existing_values_conflict=failure.get("reason") in {
                    "existing_values_conflict", "existing_symbol_alias_conflict",
                },
            )
            result["actual_minutes"] = int((context.get("minute_aggregate") or {}).get("rows") or 0)
            attempted = sorted({str(item.get("source") or "") for item in source_errors if item.get("source")})
            attempted.extend(str(item.get("source") or "") for item in snapshots if item.get("source"))
            result["source_chain"] = ",".join(dict.fromkeys(item for item in attempted if item))
            result["evidence"] = {
                "local_daily": {
                    key: value for key, value in (context.get("local_daily") or {}).items()
                    if key in {"symbol", "date", "open", "high", "low", "close", "volume", "amount", "volume_unit", "volume_unit_basis"}
                },
                "minute_aggregate": {
                    key: value for key, value in (context.get("minute_aggregate") or {}).items()
                    if key in {"symbol", "date", "open", "high", "low", "close", "volume", "amount", "volume_unit", "rows"}
                },
                "external_snapshots": [
                    {key: value for key, value in item.items()
                     if key in {"source", "symbol", "date", "open", "high", "low", "close", "volume", "amount", "volume_unit"}}
                    for item in snapshots
                ],
            }
            result["failure_reason"] = failure.get("reason")
            results.append(result)
    finally:
        close = getattr(provider, "close", None) if provider is not None else None
        if callable(close):
            close()
        close = getattr(fuyao_client, "close", None) if fuyao_client is not None else None
        if callable(close):
            close()
    return results


def execute(repo, capset, task, emit):
    from app.indicators.pipeline import run_date_pipeline, run_pipeline
    from app.jobs.daily_pipeline import _invalidate, _refresh_single_view
    from app.services import kline_sync, market_mainline, regime_builder

    root = repo.store.data_dir
    items = task["items"]
    kind = task["kind"]
    emit(kind, 0, f"{task['label']}: 本批 {len(items)} 项", stage_pct=0)
    if kind == 'index_daily':
        from app.services import index_incremental
        try:
            result = index_incremental.sync(
                repo, capset, symbols=items,
                start_date=date.fromisoformat(task['options']['start_date']),
                end_date=date.fromisoformat(task['options']['end_date']),
                max_symbols=len(items), queue_pending=False,
                on_progress=lambda cur, total, text: emit(
                    'sync_index', int(90 * cur / max(total, 1)), text,
                    stage_pct=int(100 * cur / max(total, 1))),
            )
        finally:
            for category in ('index_daily', 'index_enriched'):
                _invalidate(category)
        emit('sync_index', 100, index_incremental.message(result), stage_pct=100)
        return {'index_sync': result, 'rows': result['written_rows'],
                'completed_items': result['completed_symbols'],
                'retry_after_seconds': result['retry_after_seconds'],
                'error': None if result['complete'] else index_incremental.message(result)}
    if kind == "minute":
        end = datetime.fromisoformat(task["options"]["as_of"])
        days = task["options"]["days"]
        from app.market_time import cn_now
        now = cn_now().replace(tzinfo=None)
        # An old midday request can be completed using the now-closed full session,
        # so its additions are validated against a complete daily reference before writing.
        if end.hour < 15 and now >= end.replace(hour=15, minute=10, second=0, microsecond=0):
            end = end.replace(hour=15, minute=0, second=0, microsecond=0)
            emit("sync_minute", 0, f"午盘续补转为 {end.date()} 全日校验, 仅补缺失记录", stage_pct=0)
        if end.hour >= 15 and not (root / "kline_daily" / f"date={end.date()}" / "part.parquet").exists():
            raise RuntimeError("目标交易日日K参考尚未就绪, 不将空结果视为分钟补齐")
        def progress(cur, total, label=""):
            pct = int(90 * cur / max(total, 1))
            emit("sync_minute", pct, f"分钟补差 {cur}/{total} {label}", stage_pct=pct, skip_log=True)
        source_issues = []
        written = 0
        cross_source_results = []
        cross_source_queue = {}
        cross_source_report = None
        # Closed-session maintenance is driven by a durable stock-day queue.
        # Intraday jobs still use the tail synchronizer because a partial day
        # cannot be judged against a final daily reference yet.
        target_date = date.fromisoformat(str(task["options"].get("target_date") or end.date())[:10])
        closed_job = end.hour >= 15 and target_date <= now.date()
        fetch_items = list(items)
        queue_result = {}
        repair_result = {}
        if closed_job:
            from app.services import minute_gap_queue, preferences

            queue_result = minute_gap_queue.ensure_date(root, items, target_date)
            pending_codes = set(minute_gap_queue.pending_batch(
                root, target_date, limit=max(1, len(items))))
            # The durable queue stores verified market-qualified identities,
            # while legacy maintenance tasks may still contain bare codes.
            lookup = minute_gap_queue._queue_lookup(root)
            fetch_items = []
            for symbol in items:
                identity, _ = minute_gap_queue._parse_queue_identity(root, symbol, lookup=lookup)
                if identity is not None and identity.symbol in pending_codes:
                    fetch_items.append(identity.symbol)
            progress(0, max(1, len(fetch_items)),
                     f"{target_date} 本地预检: 缺口 {len(fetch_items)} 只, "
                     f"跳过完整/停牌/上市前 {len(items) - len(fetch_items)} 只")
            if fetch_items:
                try:
                    from app.services.minute_repair import repair_minute_gaps

                    repair_result = repair_minute_gaps(
                        repo, start_date=target_date, end_date=target_date,
                        provider_name=preferences.get_minute_data_provider().strip().lower(),
                        progress=progress, workers=1, symbols=fetch_items,
                        stock_days=[(symbol, target_date) for symbol in fetch_items],
                        max_requests=len(fetch_items), single_day_requests=True,
                        repair_conflicts=task["options"].get("repair_conflicts") is True,
                        task_id=task["id"],
                    )
                    written = int(repair_result.get("written_rows", 0) or 0)
                    source_issues.extend(repair_result.get("source_errors", []))
                except Exception as exc:
                    if not getattr(exc, "reason", None):
                        raise
                    written = getattr(exc, "written_rows", 0)
                    from app.plugins.tushare.provider import minute_error_details
                    if getattr(exc, "issues", None):
                        source_issues.extend(exc.issues)
                    else:
                        source_issues.append(minute_error_details(exc))
            else:
                emit("sync_minute", 100,
                     f"{target_date} 本地校验无真实缺口, 不请求分钟源", stage_pct=100)
        else:
            try:
                written = kline_sync.sync_and_persist_minute(
                    items, repo, capset, days=days, as_of=end, on_chunk_done=progress,
                    on_source_issues=source_issues.extend)
            except Exception as exc:
                if not getattr(exc, "reason", None):
                    raise
                written = getattr(exc, "written_rows", 0)
                from app.plugins.tushare.provider import minute_error_details
                if getattr(exc, "issues", None):
                    if not source_issues:
                        source_issues.extend(exc.issues)
                else:
                    source_issues.append(minute_error_details(exc))
        # Successfully persisted segments remain usable after an upstream timeout.
        _invalidate("minute")
        validation_start = datetime.combine(target_date, datetime.min.time()) if closed_job else end - timedelta(days=days)
        validation_end = datetime.combine(target_date, datetime.min.time()).replace(hour=15) if closed_job else end
        failures, statuses = minute_validation_report(repo, items, validation_start, validation_end)
        annotate_minute_failures(failures, source_issues)
        status_by_key = {(item["symbol"], item["date"]): item for item in statuses}
        for failure in failures:
            item = status_by_key.get((failure["symbol"], failure["date"]))
            if item is None:
                continue
            item["reason"] = failure.get("reason")
            item["source_errors"] = failure.get("source_errors", [])
            if _all_sources_unsupported(failure):
                item["status"] = "source_unsupported"
        invalid = {r["symbol"] for r in failures}
        current_day = end.date()
        terminal_unknown = set()
        scheduled_codes = set()
        schedule_retry_after = 0
        schedule_retry = False
        from app.services import minute_failover, minute_status
        for failure in failures:
            if failure.get("date") != str(current_day):
                continue
            code = failure["symbol"]
            existing = minute_status.get(root, code, current_day) or {}
            attempts = int(existing.get("retry_attempts") or 0)
            source_terminal = _all_sources_unsupported(failure)
            # Missing or unusable daily OHLC/turnover cannot be repaired by
            # another minute provider. Keep the stock-day unknown until the
            # daily pipeline supplies a trustworthy reference.
            if closed_job and failure.get("reason") == "daily_reference_missing":
                terminal_unknown.add(code)
                continue
            retryable_empty = failure.get("reason") in {"source_empty", "source_pending", "not_requested"}
            retryable_empty = retryable_empty or failure.get("data_reason") == "source_empty"
            if source_terminal:
                terminal_unknown.add(code)
                continue
            if retryable_empty:
                allowed, delay, exhausted = _minute_retry_policy(current_day, now, attempts)
                if exhausted:
                    terminal_unknown.add(code)
                elif not allowed:
                    scheduled_codes.add(code)
                    schedule_retry = True
                    schedule_retry_after = max(schedule_retry_after, delay)
                else:
                    scheduled_codes.add(code)
                    schedule_retry = True
        for item in statuses:
            if item["date"] != str(current_day):
                continue
            code = item["symbol"]
            if code in terminal_unknown:
                item["status"] = "unknown"
                item["terminal"] = True
                item["reason"] = item.get("reason") or "当日数据在限定窗口内仍未返回"
            elif item.get("status") == "unknown":
                item["retry_attempts"] = int((minute_status.get(root, code, current_day) or {}).get(
                    "retry_attempts", 0)) + (1 if code in scheduled_codes else 0)
        minute_status.upsert(root, statuses)
        if closed_job:
            from app.services import minute_gap_queue

            minute_gap_queue.record_statuses(root, statuses, source_issues)
            max_cross_source = max(1, int(os.getenv("MINUTE_CROSS_SOURCE_MAX_ITEMS", "50")))
            if failures:
                emit("sync_minute", 92,
                     f"第三方交叉核验 {min(len(failures), max_cross_source)}/{len(failures)} 个股票日",
                     stage_pct=96, skip_log=True)
                cross_source_results = _cross_source_verify_failures(
                    repo, failures, max_items=max_cross_source,
                )
                from app.services.minute_cross_source import persist_verification_report
                cross_source_report = persist_verification_report(
                    root, cross_source_results, task_kind="minute",
                )
                if cross_source_results:
                    cross_source_queue = minute_gap_queue.record_cross_source_results(
                        root, cross_source_results,
                    )
        complete = [s for s in items if (
            kline_sync._normalize_sync_symbol(s).split(".", 1)[0] not in invalid
            or kline_sync._normalize_sync_symbol(s).split(".", 1)[0] in terminal_unknown
        )]
        skipped_items = [
            s for s in items
            if kline_sync._normalize_sync_symbol(s).split(".", 1)[0] in terminal_unknown
        ]
        skipped_items += [
            s for s in items
            if any(item["symbol"] == kline_sync._normalize_sync_symbol(s).split(".", 1)[0]
                   and item["status"] in {
                       "no_trade", "pre_listing", "delisted", "code_changed",
                       "identity_unknown", "identity_conflict",
                   }
                  for item in statuses)
            and kline_sync._normalize_sync_symbol(s).split(".", 1)[0] not in invalid
        ]
        from collections import Counter
        labels = {"source_pending": "主源数据准备中", "not_requested": "源冷却中, 尚未请求",
                  "rate_limited": "分钟源限流", "request_timeout": "分钟源请求超时",
                  "transport_error": "分钟源连接失败", "upstream_unavailable": "分钟源暂不可用",
                  "backup_unavailable": "备用源未通过分钟可用性探测",
                  "permission_denied": "分钟源权限不足", "endpoint_not_found": "分钟接口不存在",
                  "source_empty": "无分钟数据", "source_incomplete": "分钟缺口",
                  "source_invalid": "分钟量价异常", "daily_price_conflict": "与日K价格冲突",
                  "daily_turnover_conflict": "与日K量额冲突", "minute_turnover_unit_conflict": "量额单位冲突",
                  "daily_reference_missing": "缺日K参考"}
        counts = Counter(r["reason"] for r in failures)
        detail = "; ".join(f"{labels.get(k, k)} {v} 个股票日" for k, v in counts.items())
        state = minute_failover.retry_state(root, include_local=end.hour >= 15) if source_issues else {}
        source_retry = bool(state.get("source_retry")) or schedule_retry
        retry_after = max(state.get("retry_after_seconds", 0), schedule_retry_after)
        if source_retry:
            retry_label = "按当日窗口重试" if schedule_retry_after else "主备源冷却"
            detail += f"; {retry_label}, {retry_after} 秒后重试同一增量"
        terminal_validation_reasons = {
            "daily_reference_missing", "identity_history_unknown",
            "not_listed_yet", "delisted_on_date", "code_changed",
            "market_identity_conflict", "lifecycle_date_missing",
            "pre_listing", "delisted", "identity_unknown", "identity_conflict",
            "no_trade", "suspension_data_conflict",
        }
        non_terminal_failures = [
            failure for failure in failures
            if failure.get("symbol") not in terminal_unknown
            and failure.get("reason") not in terminal_validation_reasons
        ]
        emit("sync_minute", 100, f"本批新增 {written} 行; 校验完成 {len(complete)}/{len(items)} 只", stage_pct=100)
        return {"minute_rows": written, "completed_items": complete,
                "skipped_items": skipped_items, "unknown_items": [s for s in items
                    if kline_sync._normalize_sync_symbol(s) in terminal_unknown],
                "validation_failures": failures, "validated_until": end.isoformat(),
                "source_errors": source_issues, "source_retry": source_retry,
                "retry_after_seconds": retry_after,
                "gap_queue": queue_result if closed_job else {},
                "repair": repair_result if closed_job else {},
                "authorization_required": repair_result.get("approval_required", []) if closed_job else [],
                "identity_pending": repair_result.get("identity_pending", []) if closed_job else [],
                "cross_source_verification": {
                    "checked": len(cross_source_results),
                    "report_dir": cross_source_report,
                    "queue": cross_source_queue,
                    "results": cross_source_results,
                } if closed_job else {},
                # Missing daily references and unresolved lifecycle history are
                # durable quarantine states, not batch-fatal errors. They stay
                # visible in the queue and can be resumed after the daily
                # source is repaired without blocking other stock-days.
                "error": detail if non_terminal_failures else None}

    if kind == "adjustment":
        from app.indicators.pipeline import enriched_publication_incomplete
        if enriched_publication_incomplete(root, "stock"):
            raise RuntimeError("历史发布尚未完成, 请先修复发布; 不自动扩大到全市场重建")
        from app.services.regime_dependencies import AdjustmentScope
        scope = AdjustmentScope(repo, items)
        written = run_pipeline(data_dir=root, symbols=items,
            on_before_symbol_publish=scope.before_publish, on_batch_done=lambda cur, total: emit(
            "rebuild_adjustment", int(95 * cur / max(total, 1)),
            f"复权历史重算 {cur}/{total} 批", stage_pct=int(100 * cur / max(total, 1))))
        scope.committed()
    elif kind == "enriched_dates":
        maintenance_queue.enqueue(root, "regime", items, reason="input_changed")
        written = sum(run_date_pipeline(root, date.fromisoformat(d)) for d in items)
        if not written:
            raise RuntimeError("历史指标未生成")
    elif kind == "regime":
        from app.services.regime_queue_audit import requires_review
        options = task.get("options", {})
        if requires_review(options):
            raise RuntimeError("regime_change_evidence_required")
        dates = [date.fromisoformat(d) for d in items]
        skipped_items = []
        if options.get("reason") == "missing_result":
            old = regime_builder.load_regime_history(root)
            known = set(old["date"].to_list()) if not old.is_empty() else set()
            needed = set(dates) - known
            dates = sorted(set(dates) & needed)
            skipped_items = [d for d in items if date.fromisoformat(d) not in dates]
            if not dates:
                emit("regime", 100, f"本批 {len(items)} 日输入未变化, 跳过重复计算", stage_pct=100)
                return {"rows": 0, "skipped_dates": len(items), "skipped_items": skipped_items}
        rows = regime_builder.compute_regime_incremental(
            repo, root, target_dates=dates,
            on_progress=lambda cur, total, label: emit("compute_regime", int(50 * cur / max(total, 1)),
                f"历史市场环境 {cur}/{total} 日 [{label}]", stage_pct=int(100 * cur / max(total, 1))))
        if set(rows["date"].to_list() if not rows.is_empty() else []) != set(dates):
            raise RuntimeError("历史市场环境输入不完整")
        for i, category in enumerate(("concept", "industry")):
            label = "概念主线" if category == "concept" else "行业主线"
            market_mainline.compute_mainline_incremental(repo, root, kind=category, target_dates=dates,
                on_progress=lambda cur, total, window, base=50 + 25 * i, name=label: emit(
                    "compute_regime", base + int(25 * cur / max(total, 1)),
                    f"{name} {cur}/{total} 日 [{window}]", stage_pct=int(100 * cur / max(total, 1))))
        from app.api.regime import invalidate_regime_cache
        invalidate_regime_cache()
        written = rows.height
    else:
        raise ValueError("Unknown maintenance task")
    if kind != "regime":
        _refresh_single_view(repo, "kline_enriched")
        _invalidate("enriched")
        repo.refresh_cache()
    message = f"{task['label']}: 本批完成 {len(items)} 项"
    result = {"rows": written}
    if kind == "regime":
        result["skipped_items"] = skipped_items
    if kind == "adjustment":
        message += f"; 市场环境实际变化 {len(scope.changed)} 日已排队, 未变化日期不重算"
        result["regime_changed_dates"] = [str(d) for d in sorted(scope.changed)]
    emit(kind, 100, message, stage_pct=100)
    return result
