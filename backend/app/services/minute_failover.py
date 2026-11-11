"""Promax-first minute retrieval with independently checked TDX fallback."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import uuid
from datetime import datetime, time, timedelta
from pathlib import Path

import polars as pl

from app.config import settings
from app.market_time import cn_now

logger = logging.getLogger(__name__)
# The TDX bridge pages recent bars, not arbitrary two-year history.
BACKUP_LOOKBACK_DAYS = 90
SOURCE_FAILURES = {"source_pending", "rate_limited", "upstream_unavailable", "request_timeout",
                   "transport_error", "backup_unavailable", "permission_denied", "endpoint_not_found"}
_state_lock = threading.RLock()


def _safe_detail(detail):
    return {k: detail[k] for k in ("reason", "cause", "retryable", "http_status", "retry_after_seconds")
            if k in detail}


def _source_states(root):
    path = Path(root) / "user_data/minute_source_state.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        logger.warning("Minute source cooldown state could not be read")
        return {}


def _source_coverage(root):
    path = Path(root) / "user_data/minute_source_coverage.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError, TypeError):
        value = {}
    return value if isinstance(value, dict) else {}


def source_coverage(root):
    """Return per-source, per-symbol coverage decisions for diagnostics."""
    with _state_lock:
        return _source_coverage(root)


def source_unsupported(root, source, symbol) -> bool:
    code = re.search(r"\d{6}", str(symbol or ""))
    if not code:
        return False
    item = _source_coverage(root).get(f"{source}|{code.group()}") or {}
    return bool(item.get("unsupported"))


def _save_source_coverage(root, state):
    path = Path(root) / "user_data/minute_source_coverage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def _trading_days(root):
    days = set()
    for folder in (Path(root) / "kline_daily").glob("date=????-??-??"):
        try:
            days.add(datetime.strptime(folder.name[5:], "%Y-%m-%d").date())
        except (ValueError, OSError):
            continue
    return sorted(days)


def _record_source_coverage(root, issues):
    """Mark a source unsupported only after two consecutive historical empties."""
    observations = {}
    for item in issues or []:
        source = str(item.get("source") or "").lower()
        reason = str(item.get("reason") or "")
        symbol = re.search(r"\d{6}", str(item.get("symbol") or ""))
        day = str(item.get("date") or "")[:10]
        if (source not in {"promax", "tdx", "stockdb"} or not symbol or len(day) != 10
                or reason not in {"source_empty", "accepted"}):
            continue
        key = f"{source}|{symbol.group()}"
        value = observations.setdefault(key, {"empty_days": set(), "accepted_days": set()})
        if reason == "source_empty":
            value["empty_days"].add(day)
        elif reason == "accepted":
            value["accepted_days"].add(day)
    if not observations:
        return
    days = _trading_days(root)
    index = {day.isoformat(): i for i, day in enumerate(days)}
    with _state_lock:
        state = _source_coverage(root)
        for key, value in observations.items():
            entry = dict(state.get(key) or {})
            accepted_day = max(value["accepted_days"], default=entry.get("last_accepted_date"))
            if accepted_day:
                entry["last_accepted_date"] = max(accepted_day, entry.get("last_accepted_date") or accepted_day)
                entry.pop("unsupported", None)
                entry.pop("unsupported_reason", None)
            empty_days = set(entry.get("empty_days") or []) | value["empty_days"]
            empty_days = {day for day in empty_days if not accepted_day or day > entry["last_accepted_date"]}
            if empty_days:
                ordered = sorted(empty_days, key=lambda item: index.get(item, -1))
                entry["empty_days"] = ordered[-8:]
                consecutive = 0
                last_position = None
                for current in reversed(ordered):
                    position = index.get(current)
                    if position is None:
                        break
                    if last_position is not None and position != last_position - 1:
                        break
                    consecutive += 1
                    last_position = position
                entry["consecutive_empty"] = consecutive
                if consecutive >= 2:
                    entry["unsupported"] = True
                    entry["unsupported_reason"] = "two_consecutive_historical_empty_days"
            else:
                entry.pop("unsupported", None)
                entry.pop("unsupported_reason", None)
                entry["consecutive_empty"] = 0
                entry["empty_days"] = []
            entry["source"], entry["symbol"] = key.split("|", 1)
            entry["updated_at"] = _wallclock(cn_now()).isoformat(timespec="seconds")
            state[key] = entry
        _save_source_coverage(root, state)


def _save_source(root, name, failure=None):
    now = _wallclock(cn_now())
    with _state_lock:
        states = _source_states(root)
        if failure is None:
            if name not in states:
                return
            states.pop(name)
        else:
            attempts = states.get(name, {}).get("attempts", 0) + 1
            delay = max(60 * 2 ** min(attempts - 1, 5), failure.get("retry_after_seconds", 0))
            if failure.get("cause") == "stockdb_snapshot_stale":
                delay = max(60, failure.get("retry_after_seconds", 600))
            if failure.get("reason") in {"permission_denied", "endpoint_not_found"}:
                delay = max(delay, 3600)
            states[name] = {**_safe_detail(failure), "attempts": attempts,
                            "retry_at": (now + timedelta(seconds=delay)).isoformat()}
        path = Path(root) / "user_data/minute_source_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(states, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)


def _record_stockdb_freshness(root, stats, now):
    """Persist observed publication time; updater success alone is not data freshness."""
    day = stats.get("snapshot_target_date")
    if not day:
        return
    path = Path(root) / "user_data/stockdb_freshness.json"
    with _state_lock:
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError):
            state = {}
        history = state.setdefault("history", {})
        entry = history.setdefault(day, {"first_checked_at": now.isoformat()})
        entry.update({
            "last_checked_at": now.isoformat(),
            "sample_size": stats.get("snapshot_sample_size", 0),
            "available_samples": stats.get("snapshot_available_samples", 0),
            "coverage": stats.get("snapshot_coverage"),
            "ready": stats.get("snapshot_ready"),
        })
        if stats.get("snapshot_ready") and not entry.get("first_ready_at"):
            entry["first_ready_at"] = now.isoformat()
        state["latest"] = {"date": day, **entry}
        for old_day in sorted(history)[:-30]:
            history.pop(old_day, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)


def _deferred(root, name):
    with _state_lock:
        state = _source_states(root).get(name)
    if not state:
        return None
    # Snapshot publication is date-scoped. A missing latest day must not disable
    # direct reads for older dates that are already present in the local database.
    if name == "stockdb" and state.get("cause") == "stockdb_snapshot_stale":
        return None
    delay = math.ceil((datetime.fromisoformat(state["retry_at"]) - _wallclock(cn_now())).total_seconds())
    if delay <= 0:
        return None
    return {**_safe_detail(state), "reason": "not_requested", "cause": state.get("cause", state["reason"]),
            "retry_after_seconds": delay, "source": name}


def retry_state(root, *, include_local=False):
    """Only defer the increment when neither route can currently be attempted."""
    names = ["promax", "tdx"]
    if include_local and settings.stockdb_enabled and settings.stockdb_minute_fallback_enabled:
        names.append("stockdb")
    failures = [_deferred(root, name) for name in names]
    blocked = all(failures)
    return {"retry_after_seconds": min(r["retry_after_seconds"] for r in failures) if blocked else 0,
            "source_errors": [r for r in failures if r], "source_retry": blocked}


class MinuteSourcesUnavailableError(RuntimeError):
    def __init__(self, issues=None):
        self.issues = issues or []
        terminal = {}
        for item in self.issues:
            terminal[(item.get("symbol"), item.get("date"))] = item
        failures = [i for i in terminal.values() if i.get("reason") != "accepted"]
        network = SOURCE_FAILURES | {"not_requested", "source_unsupported"}
        self.reason = ("upstream_unavailable" if not failures or all(
            i.get("reason") in network and i.get("primary_reason", i.get("reason")) in network
            for i in failures) else "minute_validation_failed")
        self.retryable = not failures or any(i.get("reason") not in {
            "existing_values_conflict", "existing_symbol_alias_conflict"} for i in failures)
        self.retry_after_seconds = max((i.get("retry_after_seconds", 0) for i in self.issues), default=0)
        detail = "; ".join(
            f"{i.get('symbol', '')} {i.get('date', '')} {i.get('source', '')}: {i.get('reason', '')}"
            for i in self.issues[:6])
        title = "分钟源暂不可用" if self.reason == "upstream_unavailable" else "分钟数据未通过校验, 异常股票待补"
        super().__init__(f"{title}: {detail}; 详见 minute_failover 记录")


def enabled(name, provider):
    mode = getattr(provider, "_connection_mode", None)
    # QMT is an authenticated local primary when explicitly selected.  Its
    # empty/partial result must still enter the same verified Promax -> TDX ->
    # StockDB chain instead of silently becoming a terminal empty response.
    if name == "qmt":
        return True
    return (name == "tushare" and settings.minute_fallback_provider.strip().lower() == "tdx"
            and callable(mode) and mode() == "promax")


def resolve_backup():
    from app.data_providers import custom

    try:
        if custom.provider_has_dataset("tdx", "minute"):
            return custom.get_provider("tdx")
    except Exception:
        logger.warning("TDX minute fallback could not be resolved")
    return None


def resolve_qmt_backup():
    """Resolve the local QMT minute source for closed-day gap repair."""
    from app.data_providers import custom

    try:
        if custom.provider_has_dataset("qmt", "minute"):
            return custom.get_provider("qmt")
    except Exception:
        logger.warning("QMT minute fallback could not be resolved")
    return None


def resolve_stockdb_backup():
    if not (settings.stockdb_enabled and settings.stockdb_minute_fallback_enabled):
        return None
    from app.data_providers import custom

    try:
        if custom.provider_has_dataset("stockdb", "minute"):
            return custom.get_provider("stockdb")
    except Exception:
        logger.warning("Local StockDB minute fallback could not be resolved")
    return None


def resolve_promax_backup():
    """Resolve the configured Promax/Tushare minute adapter for QMT fallback."""
    from app.data_providers import custom

    for name in ("tushare", "promax"):
        try:
            if custom.provider_has_dataset(name, "minute"):
                return custom.get_provider(name)
        except Exception:
            logger.warning("Promax minute fallback could not be resolved: %s", name)
    return None


def _offline_fallback(symbols, start, end):
    """Read extracted per-symbol offline files as the last fallback.

    Archives are intentionally not unpacked here.  The normal import page
    handles bulk archives; this path is for already extracted CSV/Parquet
    files and is only invoked for unresolved stock-days.
    """
    configured_root = os.getenv("OFFLINE_DATA_DIR", "").strip()
    if not configured_root or not symbols:
        return pl.DataFrame(), []
    offline_root = Path(configured_root).expanduser()
    if not offline_root.exists():
        return pl.DataFrame(), []
    from app.services.history_import import _normalize_minute_history_df
    from app.services.minute_repair import keyed

    codes = {re.search(r"\d{6}", str(symbol)).group() for symbol in symbols
             if re.search(r"\d{6}", str(symbol))}
    frames, issues = [], []
    candidates = [path for path in offline_root.rglob("*")
                  if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".pq"}
                  and any(code in path.name for code in codes)]
    for path in candidates[:200]:
        try:
            frame, _ = _normalize_minute_history_df(path, None, {})
            if frame.is_empty():
                continue
            frame = keyed(frame).filter(
                pl.col("datetime").is_between(start, end)
                & pl.col("_code").is_in(codes)
            ).drop("_code")
            if not frame.is_empty():
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            issues.append({"source": "offline", "file": str(path.name),
                           "reason": "source_schema_invalid", "detail": str(exc),
                           "retryable": False})
    if frames:
        return pl.concat(frames, how="diagonal_relaxed").unique(
            subset=["symbol", "datetime"], keep="last"), issues
    return pl.DataFrame(), issues


def _qmt_first(provider, symbols, start, end, *, asset_type, freq,
               on_chunk_done=None, data_dir=None, allow_replacement=False):
    """Use QMT only for verified rows, then route unresolved stock-days onward."""
    root = Path(data_dir or settings.data_dir)
    primary, primary_issues = _read_minutes(provider, symbols, start, end, asset_type, freq)
    if asset_type != "stock" or freq != "1m":
        return primary, primary_issues

    # Offline QMT history can be intentionally imported before every minute is
    # available.  This is opt-in for historical recovery only: downstream
    # validation still marks incomplete stock-days as unknown/partial, so the
    # raw rows never masquerade as a complete 240-bar session.
    if os.getenv("QMT_IMPORT_PARTIAL", "0").strip().lower() in {"1", "true", "yes"}:
        if not primary.is_empty() and on_chunk_done:
            on_chunk_done(2, 2)
        return primary, primary_issues

    from app.services.minute_repair import checked_additions, daily_reference, keyed

    unresolved = []
    verified = []
    days = [start.date() + timedelta(days=i) for i in range((end.date() - start.date()).days + 1)]
    for symbol in symbols:
        code = re.search(r"\d{6}", str(symbol or ""))
        if not code:
            continue
        code = code.group()
        for day in days:
            daily_path = root / "kline_daily" / f"date={day}" / "part.parquet"
            if not daily_path.exists():
                continue
            daily = pl.read_parquet(daily_path)
            refs = daily_reference(daily, data_dir=root)
            daily_codes = keyed(daily)
            no_trade = code in set(daily_codes.filter(
                (pl.col("volume") == 0) & (pl.col("amount") == 0)
            )["_code"].drop_nulls())
            if no_trade:
                continue
            part = _part(primary, code, day)
            closed = end >= datetime.combine(day, time(15)) and day <= cn_now().date()
            if closed:
                reason = ("daily_reference_missing" if code not in refs else
                          checked_additions(part, part.clear(), refs[code], day, source="qmt")[1])
            else:
                reason = _intraday_reason(part, day, start, min(end, cn_now().replace(tzinfo=None)))
            if reason in {"accepted", "already_complete"}:
                if not part.is_empty():
                    verified.append(part)
                continue
            unresolved.append(symbol)
            primary_issues.append({"source": "qmt", "symbol": code, "date": str(day),
                                   "reason": reason, "primary_reason": reason,
                                   "retryable": reason not in {"daily_reference_missing"}})
    unresolved = list(dict.fromkeys(unresolved))
    if on_chunk_done:
        on_chunk_done(1, 2)
    if not unresolved:
        return (pl.concat(verified, how="diagonal_relaxed") if verified else pl.DataFrame()), primary_issues

    # QMT is a local primary.  Do not hold already verified rows hostage to a
    # remote backup request: Promax/TDX may be rate-limited or return 202 while
    # the local QMT rows are perfectly usable.  Persist the verified portion
    # now; unresolved stock-days stay in the gap queue and are handled by the
    # next backup pass (or an explicit Promax/TDX repair run).
    if verified:
        logger.info(
            "QMT 已验证 %d 个分钟片段，继续为 %d 个未通过股票日执行备用源校验",
            len(verified), len(unresolved),
        )

    promax = resolve_promax_backup()
    if promax is None:
        primary_issues.append({"source": "promax", "reason": "backup_unavailable", "retryable": True})

        class UnavailablePromax:
            def get_minute(self, *_args, **_kwargs):
                return pl.DataFrame()

        promax = UnavailablePromax()
    remote, remote_issues = _fetch_remote_first(
        promax, unresolved, start, end, asset_type=asset_type, freq=freq,
        on_chunk_done=None, data_dir=root, allow_replacement=allow_replacement,
    )
    offline_symbols = []
    for symbol in unresolved:
        code = re.search(r"\d{6}", str(symbol)).group()
        needs_offline = False
        for day in days:
            daily_path = root / "kline_daily" / f"date={day}" / "part.parquet"
            if not daily_path.exists():
                continue
            daily = pl.read_parquet(daily_path)
            refs = daily_reference(daily, data_dir=root)
            part = _part(remote, code, day)
            closed = end >= datetime.combine(day, time(15)) and day <= cn_now().date()
            if closed:
                reason = ("daily_reference_missing" if code not in refs else
                          checked_additions(part, part.clear(), refs[code], day, source="qmt")[1])
            else:
                reason = _intraday_reason(part, day, start, min(end, cn_now().replace(tzinfo=None)))
            if reason not in {"accepted", "already_complete"}:
                needs_offline = True
        if needs_offline:
            offline_symbols.append(symbol)
    offline, offline_issues = _offline_fallback(offline_symbols, start, end)
    if not remote.is_empty() and not offline.is_empty():
        remote_keys = keyed(remote).select(["_code", "datetime"]).unique()
        offline = keyed(offline).join(
            remote_keys, on=["_code", "datetime"], how="anti"
        ).drop("_code")
    pieces = [frame for frame in [*verified, remote, offline] if not frame.is_empty()]
    return (pl.concat(pieces, how="diagonal_relaxed") if pieces else pl.DataFrame(),
            [*primary_issues, *remote_issues, *offline_issues])


def _stockdb_fallback(root, unresolved, refs, existing_days, output, issues, allow_replacement):
    """Last-resort closed-day reads, with the same daily and old-value gates."""
    from app.services.minute_repair import checked_additions

    if not (settings.stockdb_enabled and settings.stockdb_minute_fallback_enabled):
        return
    terminal = {(item["symbol"], item["date"]): item for item in issues if "symbol" in item and "date" in item}
    pending = [(symbol, day, primary_reason) for symbol, missing in unresolved.items()
               for day, closed, _, primary_reason in missing if closed
               and terminal.get((symbol.split(".")[0], str(day)), {}).get("reason") != "accepted"]
    if not pending:
        return
    backup = resolve_stockdb_backup()
    deferred = _deferred(root, "stockdb")
    if backup is None and not deferred:
        deferred = {"reason": "backup_unavailable", "retryable": True}
        _save_source(root, "stockdb", deferred)
        deferred = _deferred(root, "stockdb")
    for index, (symbol, day, primary_reason) in enumerate(pending):
        code = symbol.split(".")[0]
        is_bj = code.startswith(("43", "83", "87", "88", "92"))
        begin, end = datetime.combine(day, time(9, 30)), datetime.combine(day, time(15, 30) if is_bj else time(15))
        budget = index >= settings.stockdb_max_stock_days
        if budget:
            frame, failures = pl.DataFrame(), [{"reason": "not_requested", "cause": "local_batch_limit", "retryable": True}]
        elif deferred:
            frame, failures = pl.DataFrame(), [deferred]
        else:
            frame, failures = _read_minutes(backup, [symbol], begin, end, "stock", "1m")
        part = _part(frame, code, day, include_bj_after_hours=is_bj)
        if day not in existing_days and not part.is_empty():
            path = root / "kline_minute" / f"date={day}" / "part.parquet"
            existing_days[day] = pl.read_parquet(path) if path.exists() else pl.DataFrame()
        old = _part(existing_days.get(day, pl.DataFrame()), code, day, include_bj_after_hours=is_bj)
        current = part.clear() if allow_replacement or old.is_empty() else old
        delta, reason = checked_additions(
            part, current, refs[day].get(code), day, source="stockdb"
        )
        if reason in {"accepted", "already_complete"}:
            output.append(pl.concat([current, delta], how="diagonal_relaxed").sort("datetime"))
            reason = "accepted"
            _save_source(root, "stockdb")
        detail = next((item for item in reversed(failures) if item.get("date", str(day)) == str(day)), {})
        if part.is_empty() and detail:
            reason = detail.get("reason", reason)
        issues.append({"symbol": code, "date": str(day), "source": "stockdb", "primary_reason": primary_reason,
                       "reason": reason, "rows": part.height if reason == "accepted" else 0,
                       "retryable": reason not in {"accepted", "existing_values_conflict", "existing_symbol_alias_conflict"},
                       **(_safe_detail(detail) if part.is_empty() else {}),
                       **_evidence(part, day, begin.replace(minute=31), end, refs[day].get(code))})
        if not deferred and reason in SOURCE_FAILURES:
            _save_source(root, "stockdb", {"reason": reason, **_safe_detail(detail)})
            deferred = _deferred(root, "stockdb")


def _wallclock(value):
    from app.services.kline_sync import CN_TZ

    return value.astimezone(CN_TZ).replace(tzinfo=None) if value.tzinfo else value


def _read_minutes(provider, symbols, start, end, asset_type, freq):
    from app.plugins.tushare.provider import minute_error_details
    from app.services.kline_sync import _enforce_minute_beijing_wallclock

    try:
        diagnostic = getattr(provider, "get_minute_diagnostics", None)
        if callable(diagnostic):
            frame, issues = diagnostic(symbols, start, end, asset_type=asset_type, freq=freq)
        else:
            frame, issues = provider.get_minute(symbols, start, end, asset_type=asset_type, freq=freq), []
        frame = frame if frame is not None else pl.DataFrame()
        from app.services.minute_repair import COLS
        if not frame.is_empty() and not set(COLS).issubset(frame.columns):
            return pl.DataFrame(), [{"reason": "source_schema_invalid", "retryable": False}]
        return _enforce_minute_beijing_wallclock(frame, source="minute_failover"), issues
    except Exception as exc:
        return pl.DataFrame(), [minute_error_details(exc)]


def _part(frame, code, day, *, include_bj_after_hours: bool = False):
    from app.services.minute_repair import COLS, keyed, regular

    if frame.is_empty():
        return frame
    if not set(COLS).issubset(frame.columns):
        return pl.DataFrame()
    return regular(
        keyed(frame).filter(pl.col("_code") == code).select(COLS),
        day,
        include_opening=True,
        include_bj_after_hours=include_bj_after_hours,
    ).sort("datetime")


def _expected_minutes(day, begin, end):
    stamps = [datetime.combine(day, time(9, 31)) + timedelta(minutes=i) for i in range(120)]
    stamps += [datetime.combine(day, time(13, 1)) + timedelta(minutes=i) for i in range(120)]
    return [d for d in stamps if begin <= d <= end]


def _intraday_reason(frame, day, begin, end):
    from app.services.minute_repair import PRICE, regular

    if frame.is_empty():
        return "source_empty"
    expected = set(_expected_minutes(day, begin, end))
    found = set(regular(frame, day)["datetime"])
    if not expected or not expected.issubset(found) or frame["datetime"].n_unique() != frame.height:
        return "source_incomplete"
    valid = pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in PRICE])
    valid &= (pl.col("high") >= pl.max_horizontal(PRICE)) & (pl.col("low") <= pl.min_horizontal(PRICE))
    valid &= pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) >= 0) for c in ("volume", "amount")])
    # Providers may expose volume in lots or shares.  The price/VWAP sanity
    # check must accept either representation; the daily reference gate later
    # still decides whether the aggregate reconciles with the session.
    tolerance = pl.max_horizontal(pl.lit(0.02), pl.col("high") * .002)
    valid &= ((pl.col("volume") == 0) & (pl.col("amount") == 0)) | (
        (pl.col("volume") > 0) & pl.any_horizontal([
            (pl.col("amount") / (pl.col("volume") * unit)).is_between(
                pl.col("low") - tolerance, pl.col("high") + tolerance,
            )
            for unit in (1.0, 100.0)
        ]))
    return "accepted" if frame.select(valid.fill_null(False).all()).item() else "source_invalid"


def _validated_prefix(frame, day, begin, end):
    """Only keep a contiguous verified prefix; never advance past a missing minute."""
    if frame.is_empty():
        return frame
    found = set(frame["datetime"])
    last = None
    for stamp in _expected_minutes(day, begin, end):
        if stamp not in found:
            break
        last = stamp
    if last is None:
        return frame.clear()
    prefix = frame.filter(pl.col("datetime").is_between(begin, last))
    return prefix if _intraday_reason(prefix, day, begin, last) == "accepted" else frame.clear()


def _evidence(frame, day, begin, end, reference=None):
    from app.services.minute_repair import regular

    found = set(regular(frame, day)["datetime"]) if not frame.is_empty() else set()
    missing = [d for d in _expected_minutes(day, begin, end) if d not in found]
    result = {"source_rows": frame.height, "missing_minutes": len(missing),
              "missing_times": [d.strftime("%H:%M") for d in missing[:8]]}
    if not frame.is_empty():
        result.update(first_time=str(frame["datetime"].min()), last_time=str(frame["datetime"].max()),
                      source_volume=float(frame["volume"].sum()), source_amount=float(frame["amount"].sum()))
    if reference:
        result.update(reference_volume=reference["volume"], reference_amount=reference["amount"])
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in result.items()}


def _preserve_old(frame, old):
    """Do not replace previously recorded bars when switching intraday sources."""
    from app.services.minute_repair import COLS, PRICE

    if old.is_empty():
        return frame, "accepted"
    if old["symbol"].n_unique() != 1:
        return frame.clear(), "existing_symbol_alias_conflict"
    overlap = frame.join(old, on="datetime", suffix="_old")
    for col in COLS[2:]:
        absolute, relative = (.001, 1e-6) if col in PRICE else (1., .002)
        valid = (pl.col(col) - pl.col(col + "_old")).abs() <= pl.max_horizontal(
            pl.lit(absolute), pl.col(col + "_old").abs() * relative)
        if not overlap.select(valid.fill_null(False).all()).item():
            return frame.clear(), "existing_values_conflict"
    old = old.join(frame.select("datetime"), on="datetime", how="semi")
    frame = frame.with_columns(pl.lit(old["symbol"][0] if old.height else frame["symbol"][0]).alias("symbol"))
    return pl.concat([old, frame.join(old.select("datetime"), on="datetime", how="anti")]).sort("datetime"), "accepted"


def _stockdb_primary(root, symbols, start, end, now, *, allow_replacement=False, on_chunk_done=None):
    """Read closed stock-days locally first and return only fully verified days."""
    from app.services.minute_repair import checked_additions, daily_reference, keyed

    codes = {
        re.search(r"\d{6}", value).group(): value
        for value in symbols
        if re.search(r"\d{6}", value)
    }
    first = max(start.date(), now.date() - timedelta(days=BACKUP_LOOKBACK_DAYS))
    days = [first + timedelta(days=i) for i in range(max(0, (end.date() - first).days + 1))]
    targets = []
    references = {}
    no_trades = {}
    for day in days:
        closed = ((day < now.date() or now.time() >= time(15, 35))
                  and end >= datetime.combine(day, time(15)))
        path = root / "kline_daily" / f"date={day}" / "part.parquet"
        if not closed or not path.exists():
            continue
        daily = pl.read_parquet(path)
        references[day] = daily_reference(daily, data_dir=root)
        keyed_daily = keyed(daily)
        no_trades[day] = set(keyed_daily.filter(
            (pl.col("volume") == 0) & (pl.col("amount") == 0))["_code"])
        targets.append(day)
    if not targets or not codes:
        return pl.DataFrame(), set(), [], {"requested_stock_days": 0, "accepted_stock_days": 0}, targets

    backup = resolve_stockdb_backup()
    deferred = _deferred(root, "stockdb")
    issues = []
    output = []
    accepted = set()
    requested = 0
    source_success = False
    stopped = False
    limit = max(1, int(settings.stockdb_max_stock_days))
    existing_days = {}
    source_probed = {day: 0 for day in targets}
    source_available = {day: 0 for day in targets}
    source_unavailable_reasons = {day: [] for day in targets}
    values = list(codes.values())
    for day in targets:
        if stopped:
            break
        existing_path = root / "kline_minute" / f"date={day}" / "part.parquet"
        existing_days[day] = pl.read_parquet(existing_path) if existing_path.exists() else pl.DataFrame()
        # Do not query StockDB again for a locally complete and daily-validated
        # stock-day.  This is the important distinction between an incremental
        # import and a full-market re-read: local validation decides the work
        # set before any StockDB/API request is made.
        pending = []
        for symbol in values:
            code = re.search(r"\d{6}", symbol).group()
            if code in no_trades[day]:
                accepted.add((code, day))
                continue
            old = _part(existing_days[day], code, day, include_bj_after_hours=code.startswith(("43", "83", "87", "88", "92")))
            if not old.is_empty():
                _, old_reason = checked_additions(
                    old, old.clear(), references[day].get(code), day, source="local"
                )
                if old_reason in {"accepted", "already_complete"}:
                    accepted.add((code, day))
                    continue
            pending.append(symbol)
        # Existing rows that pass the same contract are already usable local
        # data. Count them in the coverage snapshot so a mostly-complete day
        # is not mislabeled as an unpublished StockDB snapshot.
        source_probed[day] += len(values) - len(pending)
        source_available[day] += len(values) - len(pending)
        if not pending:
            continue
        for offset in range(0, len(pending), limit):
            chunk = pending[offset:offset + limit]
            requested += len(chunk)
            if deferred or backup is None:
                detail = deferred or {"reason": "backup_unavailable", "retryable": True}
                frame, failures = pl.DataFrame(), [detail]
                stopped = True
            else:
                request_end = time(15, 30) if any(
                    re.search(r"\d{6}", value).group().startswith(("43", "83", "87", "88", "92"))
                    for value in chunk if re.search(r"\d{6}", value)
                ) else time(15)
                frame, failures = _read_minutes(
                    backup, chunk, datetime.combine(day, time(9, 30)), datetime.combine(day, request_end),
                    "stock", "1m")
            failure_by_code = {
                str(item.get("symbol", "")).split(".")[0]: item for item in failures
                if item.get("symbol")
            }
            for symbol in chunk:
                code = re.search(r"\d{6}", symbol).group()
                if code in no_trades[day]:
                    accepted.add((code, day))
                    continue
                part = _part(
                    frame,
                    code,
                    day,
                    include_bj_after_hours=code.startswith(("43", "83", "87", "88", "92")),
                )
                source_probed[day] += 1
                if not part.is_empty():
                    source_available[day] += 1
                old = _part(
                    existing_days[day],
                    code,
                    day,
                    include_bj_after_hours=code.startswith(("43", "83", "87", "88", "92")),
                )
                current = old.clear() if allow_replacement or old.is_empty() else old
                delta, reason = checked_additions(
                    part, current, references[day].get(code), day, source="stockdb"
                )
                detail = failure_by_code.get(code, {})
                if reason in {"accepted", "already_complete"}:
                    accepted.add((code, day))
                    source_success = source_success or not part.is_empty()
                    if not delta.is_empty():
                        output.append(part)
                    continue
                if part.is_empty() and detail:
                    reason = detail.get("reason", reason)
                if part.is_empty():
                    source_unavailable_reasons[day].append(detail.get("reason", reason))
                issues.append({
                    "symbol": code, "date": str(day), "source": "stockdb", "route": "after_close_primary",
                    **(_safe_detail(detail) if detail else {}),
                    "reason": reason, "retryable": reason not in {
                        "existing_values_conflict", "existing_symbol_alias_conflict", "source_symbol_conflict"},
                    **_evidence(part, day, datetime.combine(day, time(9, 31)), datetime.combine(day, time(15)),
                                references[day].get(code)),
                })
            network = next((item for item in failures if item.get("batch_aborted")
                            or item.get("reason") in SOURCE_FAILURES), None)
            if network:
                _save_source(root, "stockdb", network)
                stopped = True
                break
            if on_chunk_done:
                on_chunk_done(min(offset + len(chunk), len(values)), len(values))
    if source_success:
        _save_source(root, "stockdb")
    latest = targets[-1]
    sample_size = source_probed[latest]
    available_samples = source_available[latest]
    coverage = available_samples / sample_size if sample_size else 0.0
    enough_samples = sample_size >= min(len(values), settings.stockdb_readiness_sample_size)
    only_unpublished = bool(source_unavailable_reasons[latest]) and all(
        reason in {"source_empty", "source_incomplete"}
        for reason in source_unavailable_reasons[latest])
    snapshot_ready = enough_samples and coverage >= settings.stockdb_readiness_min_coverage
    publication_pending = enough_samples and not snapshot_ready and only_unpublished
    stats = {"requested_stock_days": requested, "accepted_stock_days": len(accepted),
             "failed_stock_days": len(issues), "stopped_early": stopped,
             "snapshot_target_date": str(latest), "snapshot_sample_size": sample_size,
             "snapshot_available_samples": available_samples,
             "snapshot_coverage": round(coverage, 4), "snapshot_ready": snapshot_ready,
             "publication_pending": publication_pending}
    return (pl.concat(output, how="diagonal_relaxed") if output else pl.DataFrame(),
            accepted, issues, stats, targets)


def _without_stock_days(frame, accepted):
    if frame.is_empty() or not accepted:
        return frame
    from app.services.minute_repair import keyed

    blocked = [f"{code}|{day}" for code, day in accepted]
    tagged = keyed(frame).with_columns(
        (pl.col("_code") + pl.lit("|") + pl.col("datetime").dt.date().cast(pl.String)).alias("_stock_day"))
    return tagged.filter(~pl.col("_stock_day").is_in(blocked)).drop("_stock_day", "_code")


def fetch_minutes(provider, symbols, start, end, *, asset_type="stock", freq="1m",
                  on_chunk_done=None, data_dir=None, allow_replacement=False):
    root = Path(data_dir or settings.data_dir)
    now = _wallclock(cn_now())
    bounded_end = min(_wallclock(end) if end else now, now)
    bounded_start = _wallclock(start) if start else bounded_end - timedelta(days=5)
    if getattr(provider, "name", "") == "qmt":
        return _qmt_first(
            provider, symbols, bounded_start, bounded_end, asset_type=asset_type, freq=freq,
            on_chunk_done=on_chunk_done, data_dir=root, allow_replacement=allow_replacement,
        )
    # Closed-day repair uses QMT first when the local bridge exposes
    # historical minutes. Intraday refresh keeps its existing low-latency
    # online route; only the durable closed-day gap path enters this branch.
    closed_request = (
        asset_type == "stock"
        and freq == "1m"
        and bounded_end.time() >= time(15)
        and (bounded_end.date() < now.date() or now.time() >= time(15, 10))
    )
    if closed_request:
        qmt = resolve_qmt_backup()
        if qmt is not None:
            return _qmt_first(
                qmt, symbols, bounded_start, bounded_end, asset_type=asset_type, freq=freq,
                on_chunk_done=on_chunk_done, data_dir=root, allow_replacement=allow_replacement,
            )
    local_first = (settings.stockdb_enabled and settings.stockdb_minute_fallback_enabled
                   and settings.stockdb_after_close_primary_enabled
                   and asset_type == "stock" and freq == "1m")
    if not local_first:
        return _fetch_remote_first(
            provider, symbols, bounded_start, bounded_end, asset_type=asset_type, freq=freq,
            on_chunk_done=on_chunk_done, data_dir=root, allow_replacement=allow_replacement)

    local, accepted, local_issues, stats, target_days = _stockdb_primary(
        root, symbols, bounded_start, bounded_end, now, allow_replacement=allow_replacement,
        on_chunk_done=on_chunk_done)
    if not target_days:
        return _fetch_remote_first(
            provider, symbols, bounded_start, bounded_end, asset_type=asset_type, freq=freq,
            on_chunk_done=on_chunk_done, data_dir=root, allow_replacement=allow_replacement)
    _record_stockdb_freshness(root, stats, now)
    publication_pending = stats.get("publication_pending", False)
    if publication_pending:
        target_day = stats["snapshot_target_date"]
        retry_after = settings.stockdb_publication_retry_seconds
        for issue in local_issues:
            if issue.get("date") == target_day and issue.get("reason") in {"source_empty", "source_incomplete"}:
                issue.update(reason="source_pending", cause="stockdb_snapshot_stale",
                             retryable=True, retry_after_seconds=retry_after)
        _save_source(root, "stockdb", {
            "reason": "source_pending", "cause": "stockdb_snapshot_stale",
            "retryable": True, "retry_after_seconds": retry_after,
        })
    target_set = set(target_days)
    remote_symbols = []
    for value in symbols:
        match = re.search(r"\d{6}", value)
        if match and any((match.group(), day) not in accepted for day in target_set):
            remote_symbols.append(value)
    fallback_total = len(remote_symbols)
    if publication_pending or stats["accepted_stock_days"]:
        remote_symbols = remote_symbols[:settings.stockdb_online_fallback_symbols]
    stats["online_fallback_symbols"] = len(remote_symbols)
    stats["deferred_online_symbols"] = fallback_total - len(remote_symbols)
    remote, remote_issues = (pl.DataFrame(), [])
    if remote_symbols:
        remote, remote_issues = _fetch_remote_first(
            provider, remote_symbols, bounded_start, bounded_end, asset_type=asset_type, freq=freq,
            on_chunk_done=on_chunk_done, data_dir=root, allow_replacement=allow_replacement,
            allow_stockdb_tail=False, write_journal=False)
        remote = _without_stock_days(remote, accepted)
    pieces = [frame for frame in (local, remote) if not frame.is_empty()]
    issues = [*local_issues, *remote_issues]
    folder = root / "import_reports/minute_failover" / str(now.date())
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{uuid.uuid4().hex}.json").write_text(json.dumps({
        "requested_at": now.isoformat(), "route": ["qmt", "stockdb", "promax", "tdx", "offline"],
        "start": bounded_start.isoformat(), "end": bounded_end.isoformat(),
        "local_primary": stats, "details": issues,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if not pieces:
        return pl.DataFrame(), issues
    result = pl.concat(pieces, how="diagonal_relaxed")
    return result.filter(pl.col("datetime").is_between(bounded_start, bounded_end)).sort(["symbol", "datetime"]), issues


def _fetch_remote_first(provider, symbols, start, end, *, asset_type="stock", freq="1m",
                        on_chunk_done=None, data_dir=None, allow_replacement=False,
                        allow_stockdb_tail=True, write_journal=True):
    from app.services.minute_repair import checked_additions, daily_reference

    root = Path(data_dir or settings.data_dir)
    now = _wallclock(cn_now())
    end = min(_wallclock(end) if end else now, now)
    start = _wallclock(start) if start else end - timedelta(days=5)
    deferred = _deferred(root, "promax")
    skipped_primary = {
        re.search(r"\d{6}", value).group()
        for value in symbols
        if re.search(r"\d{6}", value) and source_unsupported(root, "promax", re.search(r"\d{6}", value).group())
    }
    request_symbols = [
        value for value in symbols
        if not (re.search(r"\d{6}", value) and re.search(r"\d{6}", value).group() in skipped_primary)
    ]
    primary, primary_issues = (pl.DataFrame(), [deferred]) if deferred else (
        _read_minutes(provider, request_symbols, start, end, asset_type, freq)
        if request_symbols else (pl.DataFrame(), []))
    if not deferred and request_symbols:
        aborted = any(i.get("batch_aborted") or i.get("reason") == "not_requested" for i in primary_issues)
        unavailable = primary.is_empty() and primary_issues and all(
            i.get("cause", i.get("reason")) in SOURCE_FAILURES for i in primary_issues)
        if aborted or unavailable:
            # HTTP 202 means this individual Promax request is being prepared.
            # It must not cool down the whole source: different symbols have
            # independent preparation keys and can still be submitted.
            actual = [i for i in primary_issues
                      if i.get("reason") in SOURCE_FAILURES
                      and i.get("reason") != "source_pending"]
            if actual:
                detail = max(actual, key=lambda i: i.get("retry_after_seconds", 0))
                _save_source(root, "promax", detail)
        elif not primary.is_empty():
            _save_source(root, "promax")
    if on_chunk_done:
        on_chunk_done(1, 2)
    # This contract is A-share 1m only; never substitute stock bars for ETF/index.
    if asset_type != "stock" or freq != "1m" or start > end:
        _record_source_coverage(root, primary_issues)
        return primary, primary_issues
    codes = {re.search(r"\d{6}", s).group(): s for s in symbols if re.search(r"\d{6}", s)}
    first = max(start.date(), now.date() - timedelta(days=BACKUP_LOOKBACK_DAYS))
    days = [first + timedelta(days=i) for i in range(max(0, (end.date() - first).days + 1))]
    refs = {}
    targets = {}
    for day in days:
        closed = (day < now.date() or now.time() >= time(15, 10)) and end >= datetime.combine(day, time(15))
        path = root / "kline_daily" / f"date={day}" / "part.parquet"
        if closed:
            if not path.exists():
                continue
            refs[day] = daily_reference(pl.read_parquet(path), data_dir=root)
            targets[day] = True
        elif day.weekday() < 5 and (day < now.date() or now.time() >= time(9, 32)):
            targets[day] = False
    output, issues, unresolved = [], [], {}
    # Preserve primary history outside the bounded backup window, but not foreign tickers.
    if not primary.is_empty():
        from app.services.minute_repair import keyed
        outside = keyed(primary).filter(pl.col("_code").is_in(codes)
                                         & ~pl.col("datetime").dt.date().is_in(list(targets))).drop("_code")
        if not outside.is_empty():
            output.append(outside)
    for code, symbol in codes.items():
        for day, closed in targets.items():
            part = _part(primary, code, day)
            cutoff = min(end, now.replace(second=0, microsecond=0) - timedelta(minutes=1), datetime.combine(day, time(15)))
            if closed:
                _, reason = checked_additions(
                    part, part.clear(), refs[day].get(code), day, source="promax"
                )
            else:
                if not _expected_minutes(day, start, cutoff):
                    continue
                part = part.filter(pl.col("datetime") <= cutoff) if not part.is_empty() else part
                reason = _intraday_reason(part, day, start, cutoff)
            if reason == "accepted":
                output.append(part)
                continue
            detail = next((r for r in reversed(primary_issues)
                           if r.get("symbol", code).split(".")[0] == code
                           and r.get("date", str(day)) == str(day)), {})
            if code in skipped_primary:
                detail = {"reason": "source_unsupported", "retryable": False}
            issue = {"symbol": code, "date": str(day), "source": "promax",
                     **_safe_detail(detail),
                     "reason": detail.get("reason", reason), "retryable": detail.get("retryable", True),
                     **_evidence(part, day, datetime.combine(day, time(9, 31)) if closed else start,
                                 datetime.combine(day, time(15)) if closed else cutoff, refs.get(day, {}).get(code))}
            issues.append(issue)
            if closed and code not in refs[day]:
                issue["reason"] = "daily_reference_missing"
                continue
            unresolved.setdefault(symbol, []).append((day, closed, cutoff, issue["reason"]))
    backup = resolve_backup() if unresolved else None
    backup_failure = _deferred(root, "tdx") if unresolved else None
    if unresolved and backup is None and not backup_failure:
        backup_failure = {"reason": "backup_unavailable", "retryable": True}
        _save_source(root, "tdx", backup_failure)
    probe = getattr(backup, "probe_minute", None)
    if unresolved and not backup_failure and callable(probe):
        begin = datetime.combine(min(d for missing in unresolved.values() for d, *_ in missing), time(9, 30))
        finish = max(datetime.combine(d, time(15)) if closed else cutoff
                     for missing in unresolved.values() for d, closed, cutoff, _ in missing)
        try:
            health = probe(begin, finish)
        except Exception as exc:
            from app.plugins.tushare.provider import minute_error_details
            health = {"usable": False, **minute_error_details(exc)}
        if health.get("usable"):
            _save_source(root, "tdx")
        else:
            backup_failure = {**_safe_detail(health), "reason": "backup_unavailable",
                              "cause": health.get("reason", "source_error_unclassified"), "retryable": True}
            _save_source(root, "tdx", backup_failure)
            backup_failure["retry_after_seconds"] = (_deferred(root, "tdx") or {}).get("retry_after_seconds", 60)
    existing_days = {}
    for index, (symbol, missing) in enumerate(unresolved.items(), 1):
        code = re.search(r"\d{6}", symbol).group()
        begin = datetime.combine(min(d for d, *_ in missing), time(9, 30))
        finish = max(datetime.combine(d, time(15)) if closed else cutoff for d, closed, cutoff, _ in missing)
        tdx_skipped = source_unsupported(root, "tdx", code)
        frame, failures = _read_minutes(backup, [symbol], begin, finish, "stock", "1m") if (
            backup and not backup_failure and not tdx_skipped
        ) else (pl.DataFrame(), [{"reason": "source_unsupported", "retryable": False}]
                if tdx_skipped else [backup_failure or {"reason": "backup_unavailable", "retryable": True}])
        for day, closed, cutoff, primary_reason in missing:
            part = _part(frame, code, day)
            if not closed and not part.is_empty():
                part = part.filter(pl.col("datetime") <= cutoff)
            evidence = _evidence(part, day, datetime.combine(day, time(9, 31)) if closed else max(start, begin),
                                 datetime.combine(day, time(15)) if closed else cutoff, refs.get(day, {}).get(code))
            accepted_rows = 0
            if part.is_empty():
                old = part.clear()
            elif day not in existing_days:
                path = root / "kline_minute" / f"date={day}" / "part.parquet"
                existing_days[day] = pl.read_parquet(path) if path.exists() else pl.DataFrame()
                old = _part(existing_days[day], code, day)
            else:
                old = _part(existing_days[day], code, day)
            if old.is_empty():
                old = part.clear()
            if closed:
                current = old.clear() if allow_replacement else old
                delta, reason = checked_additions(
                    part, current, refs[day].get(code), day, source="tdx"
                )
                if reason in {"accepted", "already_complete"}:
                    part = pl.concat([current, delta], how="diagonal_relaxed").sort("datetime")
                    reason = "accepted"
            else:
                reason = _intraday_reason(part, day, max(start, begin), cutoff)
                if reason == "accepted":
                    part, reason = _preserve_old(part, old)
                elif reason == "source_incomplete":
                    prefix = _validated_prefix(part, day, max(start, begin), cutoff)
                    if not prefix.is_empty():
                        prefix, preservation = _preserve_old(prefix, old)
                        if preservation == "accepted":
                            output.append(prefix)
                            accepted_rows = prefix.height
                        else:
                            reason = preservation
            if reason == "source_empty" and failures:
                reason = failures[-1]["reason"]
            if tdx_skipped:
                reason = "source_unsupported"
            issues.append({"symbol": code, "date": str(day), "source": "tdx", "reason": reason,
                           **(_safe_detail(failures[-1]) if part.is_empty() and failures else {}),
                           "primary_reason": primary_reason, "rows": part.height if reason == "accepted" else accepted_rows,
                           "retryable": reason not in {"accepted", "existing_values_conflict", "existing_symbol_alias_conflict"},
                           **evidence})
            if reason == "accepted":
                output.append(part)
            logger.info("分钟备用源 Promax -> TDX %s %s: %s", code, day, reason)
        if on_chunk_done:
            on_chunk_done(len(unresolved) + index, 2 * len(unresolved))
    if allow_stockdb_tail:
        _stockdb_fallback(root, unresolved, refs, existing_days, output, issues, allow_replacement)
    if issues and write_journal:
        folder = root / "import_reports/minute_failover" / str(now.date())
        folder.mkdir(parents=True, exist_ok=True)
        # Unique per fetch: concurrent readers never truncate a shared journal.
        (folder / f"{uuid.uuid4().hex}.json").write_text(json.dumps({
            "requested_at": now.isoformat(), "primary": "promax", "backup": "tdx",
            "local_backup": "stockdb" if settings.stockdb_enabled and settings.stockdb_minute_fallback_enabled else None,
            "start": start.isoformat(), "end": end.isoformat(), "details": issues,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    if on_chunk_done:
        on_chunk_done(1, 1)
    if not output:
        _record_source_coverage(root, [*issues, *primary_issues])
        return pl.DataFrame(), issues or primary_issues
    result = pl.concat(output, how="diagonal_relaxed")
    _record_source_coverage(root, [*issues, *primary_issues])
    return result.filter(pl.col("datetime").is_between(start, end)).sort(["symbol", "datetime"]), issues
