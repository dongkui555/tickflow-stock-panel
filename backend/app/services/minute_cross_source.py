"""Cross-source daily reconciliation for minute-K import failures.

This module only normalizes evidence and returns a decision.  It deliberately
does not write daily or minute partitions; callers keep the existing backup
and atomic replacement gates in the import path.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from math import isfinite
import json
import time as wall_time
import csv
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PRICE_FIELDS = ("open", "high", "low", "close")
RETRYABLE_SOURCE_ERRORS = {
    "transport_error", "request_timeout", "rate_limited", "upstream_unavailable",
    "source_pending", "gateway_rejected",
}

EASTMONEY_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_UT = "fa5fd1943c7b386f172d6893dbfba10b"


def _canonical_stock_symbol(symbol: str) -> str:
    """Resolve a stock identity before any source-specific request."""
    from app.services.security_identity import IdentityError, parse_identity

    try:
        return parse_identity(symbol, asset_type="stock").symbol
    except IdentityError as exc:
        raise ValueError("stock_identity_unresolved") from exc


def _number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _normalized_snapshot(raw: dict, expected_symbol: str, expected_date: str) -> tuple[dict | None, str | None]:
    symbol = _source_identity(raw) or expected_symbol
    if not _identity_matches(raw, expected_symbol) or _record_date(
        raw.get("date") or raw.get("trade_date")
    ) != expected_date:
        return None, "identity_conflict"
    source = str(raw.get("source") or "").strip().lower()
    raw_unit = str(raw.get("source_volume_unit") or raw.get("volume_unit") or "").strip().lower()
    already_normalized = bool(raw.get("source_volume_unit"))
    factor = 1.0 if already_normalized else {"shares": 1.0, "lots": 100.0}.get(raw_unit)
    if not source or factor is None:
        return None, "source_schema_invalid"
    values = {field: _number(raw.get(field)) for field in (*PRICE_FIELDS, "amount")}
    volume = _number(raw.get("volume"))
    if volume is None or values["amount"] is None or any(values[field] is None for field in PRICE_FIELDS):
        return None, "source_schema_invalid"
    return {
        "source": source,
        "symbol": symbol,
        "date": expected_date,
        **{field: values[field] for field in PRICE_FIELDS},
        "volume": volume if already_normalized else volume * factor,
        "amount": values["amount"],
        "volume_unit": "shares",
        "source_volume_unit": raw_unit,
    }, None


def _matches(left: dict | None, right: dict | None, *, tolerance: float = 0.002) -> bool:
    if not left or not right:
        return False
    # A missing/unknown unit is evidence of an unresolved source contract, not
    # permission to compare raw numbers as if they were shares.
    if left.get("volume_unit") not in {"shares", "lots"} or right.get("volume_unit") not in {"shares", "lots"}:
        return False
    close = _number(right.get("close")) or 0.0
    price_tolerance = max(0.011, abs(close) * 1e-6)
    if any(abs(float(left[field]) - float(right[field])) > price_tolerance for field in PRICE_FIELDS):
        return False
    for field in ("volume", "amount"):
        expected = _number(right.get(field))
        actual = _number(left.get(field))
        if expected is None or actual is None or abs(actual - expected) > max(1.0, abs(expected) * tolerance):
            return False
    return True


def _as_reference(raw: dict | None, expected_symbol: str, expected_date: str) -> dict | None:
    if not raw:
        return None
    symbol = str(raw.get("symbol") or expected_symbol).strip().upper()
    if symbol != expected_symbol or str(raw.get("date") or expected_date)[:10] != expected_date:
        return None
    result = {field: _number(raw.get(field)) for field in (*PRICE_FIELDS, "volume", "amount")}
    if any(value is None for value in result.values()):
        return None
    volume_unit = str(raw.get("volume_unit") or "unknown").strip().lower()
    if volume_unit not in {"shares", "lots", "unknown"}:
        return None
    result["volume"] *= 100.0 if volume_unit == "lots" else 1.0
    return {"symbol": symbol, "date": expected_date, "volume_unit": volume_unit, **result}


def infer_volume_unit(raw: dict | None) -> str | None:
    """Infer a *source unit label* only from the amount/price relationship.

    Local legacy daily files do not carry a unit column.  The relationship
    ``amount / (close * volume)`` is approximately 1 for shares and 100 for
    lots.  This helper labels the row for comparison; it never rewrites stored
    data.  Ambiguous rows remain ``None`` and must stay quarantined.
    """
    if not raw:
        return None
    explicit = str(raw.get("volume_unit") or "").strip().lower()
    if explicit in {"shares", "lots"}:
        return explicit
    volume = _number(raw.get("volume"))
    close = _number(raw.get("close"))
    amount = _number(raw.get("amount"))
    if not volume or not close or amount is None:
        return None
    ratio = amount / (close * volume)
    if 0.70 <= ratio <= 1.30:
        return "shares"
    if 70.0 <= ratio <= 130.0:
        return "lots"
    return None


def _record_date(value) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8 and (len(text) == 8 or text.isdigit()):
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return text[:10]


def _source_identity(raw: dict) -> str:
    """Return the source identity without inventing an exchange suffix."""
    return str(raw.get("symbol") or raw.get("ts_code") or "").strip().upper()


def _identity_matches(raw: dict, expected_symbol: str) -> bool:
    identity = _source_identity(raw)
    # A six-digit response is not enough to establish the exchange.  The
    # verifier must reject it instead of silently treating it as the request's
    # market.
    return bool(identity and "." in identity and identity == expected_symbol)


def _rows(frame) -> list[dict]:
    """Convert pandas/polars/list responses without coupling the verifier to one SDK."""
    if frame is None:
        return []
    if hasattr(frame, "to_dicts"):
        return list(frame.to_dicts())
    if hasattr(frame, "to_dict"):
        try:
            return list(frame.to_dict(orient="records"))
        except TypeError:
            value = frame.to_dict()
            return list(value) if isinstance(value, list) else []
    if isinstance(frame, list):
        return [dict(item) for item in frame if isinstance(item, dict)]
    return []


def _daily_snapshot_from_row(row: dict, *, symbol: str, day: date, source: str,
                             volume_unit: str) -> dict | None:
    if _record_date(row.get("date") or row.get("trade_date")) != day.isoformat():
        return None
    # A provider may omit the exchange in normalized output. The request was
    # identity-qualified; only the requested code may be accepted.
    raw_code = str(row.get("symbol") or row.get("ts_code") or "")
    if raw_code and not _identity_matches(row, symbol):
        return None
    result = {
        "source": source,
        "symbol": symbol,
        "date": day.isoformat(),
        **{field: row.get(field) for field in (*PRICE_FIELDS, "volume", "amount")},
        "volume_unit": volume_unit,
    }
    return result if all(result.get(field) is not None for field in (*PRICE_FIELDS, "volume", "amount")) else None


def fetch_promax_daily(symbol: str, day: date, provider=None) -> tuple[dict | None, dict | None]:
    """Fetch one identity-qualified daily bar from the configured Tushare gateway."""
    owns_provider = provider is None
    try:
        symbol = _canonical_stock_symbol(symbol)
        from app.plugins.tushare.provider import TushareProvider, _to_ts_code, minute_error_details

        provider = provider or TushareProvider()
        api = provider._ensure_api()  # noqa: SLF001 - bounded single-day verifier
        frame = api.daily(
            ts_code=_to_ts_code(symbol),
            start_date=day.strftime("%Y%m%d"),
            end_date=day.strftime("%Y%m%d"),
            fields="ts_code,trade_date,open,high,low,close,vol,amount",
        )
        rows = _rows(frame)
        row = next((item for item in rows if _record_date(item.get("date") or item.get("trade_date")) == day.isoformat()), None)
        if row:
            raw_code = str(row.get("symbol") or row.get("ts_code") or "")
            if raw_code and not _identity_matches(row, symbol):
                return None, {"source": "promax", "reason": "identity_conflict", "retryable": False}
            row = {
                **row,
                "volume": (float(row.get("volume") if row.get("volume") is not None else row.get("vol")) * 100.0),
                "amount": (float(row.get("amount")) * 1000.0),
            }
        source = str(getattr(api, "last_source", None) or "promax").lower()
        snapshot = _daily_snapshot_from_row(row, symbol=symbol, day=day, source=source, volume_unit="shares") if row else None
        error = None if snapshot else {"source": "promax", "reason": "source_no_record", "retryable": False}
        return snapshot, error
    except Exception as exc:  # noqa: BLE001
        try:
            from app.plugins.tushare.provider import minute_error_details
            detail = minute_error_details(exc)
        except Exception:
            detail = {"reason": "source_error_unclassified", "retryable": False}
        if detail.get("reason") == "source_error_unclassified":
            detail["retryable"] = True
        return None, {"source": "promax", **detail}
    finally:
        if owns_provider and provider is not None:
            close = getattr(provider, "close", None)
            if callable(close):
                close()


def fetch_eastmoney_daily(symbol: str, day: date, *, opener=urlopen) -> tuple[dict | None, dict | None]:
    """Fetch a single daily bar from Eastmoney's public historical endpoint.

    This is a read-only independent check. An empty successful response is
    ``source_no_record``; network/HTTP/JSON failures remain retryable source
    failures and are never converted to a missing-data verdict.
    """
    try:
        canonical = _canonical_stock_symbol(symbol)
    except ValueError:
        return None, {"source": "eastmoney", "reason": "identity_conflict", "retryable": False}
    code, exchange = canonical.split(".", 1)
    secid = f"1.{code}" if exchange == "SH" else f"0.{code}"
    day_text = day.strftime("%Y%m%d")
    params = {
        # Cross-source validation compares raw daily bars.  Forward-adjusted
        # values can legitimately differ from the local raw daily baseline.
        "secid": secid, "klt": "101", "fqt": "0", "beg": day_text, "end": day_text,
        "lmt": "5", "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61", "ut": EASTMONEY_UT,
    }
    request = Request(
        f"{EASTMONEY_URL}?{urlencode(params)}",
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
    )
    last_error = None
    for attempt in range(2):
        try:
            with opener(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            for raw in (payload.get("data") or {}).get("klines") or []:
                fields = str(raw).split(",")
                if len(fields) < 7 or _record_date(fields[0]) != day.isoformat():
                    continue
                return {
                    "source": "eastmoney", "symbol": canonical, "date": day.isoformat(),
                    "open": float(fields[1]), "close": float(fields[2]),
                    "high": float(fields[3]), "low": float(fields[4]),
                    "volume": float(fields[5]), "amount": float(fields[6]),
                    "volume_unit": "lots",
                }, None
            return None, {"source": "eastmoney", "reason": "source_no_record", "retryable": False}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 0:
                wall_time.sleep(0.25)
    return None, {"source": "eastmoney", "reason": "transport_error", "retryable": True,
                  "retry_after_seconds": 300, "error_type": type(last_error).__name__}


def fetch_fuyao_daily(symbol: str, day: date, *, client=None) -> tuple[dict | None, dict | None]:
    """Fetch one raw daily bar from Fuyao when its optional key is available."""
    owns_client = client is None
    try:
        symbol = _canonical_stock_symbol(symbol)
        from app.plugins.fuyao.client import FuyaoClient, FuyaoError
        from app.plugins.fuyao.provider import _date_of_ms, _ms_of_date, _to_thscode, get_api_key

        if client is None:
            api_key = get_api_key()
            if not api_key:
                return None, {"source": "fuyao", "reason": "source_unsupported", "retryable": False}
            client = FuyaoClient(api_key=api_key, timeout=15.0)
        thscode = _to_thscode(symbol)
        rows = client.historical_kline(
            thscode, _ms_of_date(day), _ms_of_date(day), adjust="none",
        )
        matched = [row for row in rows if _date_of_ms(row.get("date_ms")) == day]
        if len(matched) != 1:
            return None, {"source": "fuyao", "reason": "source_no_record", "retryable": False}
        row = matched[0]
        identity = str(row.get("thscode") or row.get("symbol") or "").strip().upper()
        if identity and not _identity_matches({"symbol": identity}, symbol):
            return None, {"source": "fuyao", "reason": "identity_conflict", "retryable": False}
        mapped = {
            "symbol": identity or symbol,
            "date": day.isoformat(),
            "open": row.get("open_price"), "high": row.get("high_price"),
            "low": row.get("low_price"), "close": row.get("close_price"),
            "volume": row.get("volume"), "amount": row.get("turnover"),
        }
        snapshot = _daily_snapshot_from_row(
            mapped, symbol=symbol, day=day, source="fuyao", volume_unit="shares",
        )
        return (snapshot, None) if snapshot else (
            None, {"source": "fuyao", "reason": "source_schema_invalid", "retryable": False}
        )
    except Exception as exc:  # noqa: BLE001
        try:
            from app.plugins.fuyao.client import FuyaoError
            retryable = isinstance(exc, FuyaoError) and "网络请求失败" in str(exc)
        except Exception:
            retryable = False
        return None, {
            "source": "fuyao", "reason": "source_unavailable" if retryable else "source_error",
            "retryable": retryable, "error_type": type(exc).__name__,
        }
    finally:
        if owns_client and client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def collect_daily_snapshots(symbol: str, day: date, *, promax_provider=None,
                            fuyao_client=None) -> tuple[list[dict], list[dict]]:
    """Run the bounded third-party chain for one stock-day."""
    try:
        symbol = _canonical_stock_symbol(symbol)
    except ValueError:
        return [], [{"source": "identity", "reason": "identity_conflict", "retryable": False}]
    snapshots: list[dict] = []
    errors: list[dict] = []
    snapshot, error = fetch_promax_daily(symbol, day, provider=promax_provider)
    if snapshot:
        snapshots.append(snapshot)
    if error:
        errors.append(error)
    snapshot, error = fetch_eastmoney_daily(symbol, day)
    if snapshot:
        snapshots.append(snapshot)
    if error:
        errors.append(error)
    if fuyao_client is not False:
        snapshot, error = fetch_fuyao_daily(symbol, day, client=fuyao_client)
        if snapshot:
            snapshots.append(snapshot)
        if error and error.get("reason") != "source_unsupported":
            errors.append(error)
    return snapshots, errors


def persist_verification_report(data_dir, results: Iterable[dict], *, task_kind="minute",
                                output_dir=None) -> str:
    """Write a bounded, secret-free audit report for later review/retry."""
    from collections import Counter

    rows = [dict(item) for item in results]
    data_root = Path(data_dir).resolve()
    if output_dir is None:
        report_dir = data_root / "import_reports" / "cross_source_verify" / (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        )
    else:
        report_dir = Path(output_dir).resolve()
        if not report_dir.is_relative_to(data_root):
            raise ValueError("核验报告输出目录必须位于 data_dir 内")
    report_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "task_kind": task_kind,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(rows),
        "status_counts": dict(Counter(str(item.get("status") or "unknown") for item in rows)),
        "results": rows,
    }
    (report_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (report_dir / "summary.json").write_text(
        json.dumps({key: payload[key] for key in ("task_kind", "created_at", "count", "status_counts")},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    columns = sorted({key for item in rows for key in item})
    with (report_dir / "details.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    retryable = [item for item in rows if item.get("retryable") or item.get("status") == "source_unavailable"]
    with (report_dir / "retry_queue.jsonl").open("w", encoding="utf-8") as handle:
        for item in retryable:
            handle.write(json.dumps({
                "symbol": item.get("symbol"), "date": item.get("date"),
                "status": item.get("status"), "reason": item.get("reason"),
                "retry_after_seconds": item.get("retry_after_seconds", 0),
            }, ensure_ascii=False, default=str) + "\n")
    return str(report_dir)


def verify_stock_day(
    *,
    symbol: str,
    trade_date: date | str,
    local_daily: dict | None,
    minute_aggregate: dict | None,
    external_snapshots: Iterable[dict] = (),
    source_errors: Iterable[dict] = (),
    existing_values_conflict: bool = False,
) -> dict:
    """Return a safe cross-source decision for one identity-qualified stock-day."""
    expected_symbol = str(symbol).strip().upper()
    expected_date = trade_date.isoformat() if isinstance(trade_date, date) else str(trade_date)[:10]
    local = _as_reference(local_daily, expected_symbol, expected_date)
    minute = _as_reference(minute_aggregate, expected_symbol, expected_date)
    if minute is not None:
        minute["rows"] = int(minute_aggregate.get("rows") or 0)

    normalized: list[dict] = []
    invalid_reasons: list[str] = []
    for raw in external_snapshots:
        item, reason = _normalized_snapshot(raw, expected_symbol, expected_date)
        if item is not None:
            normalized.append(item)
        elif reason:
            invalid_reasons.append(reason)

    unique_sources: dict[str, dict] = {}
    for item in normalized:
        unique_sources[item["source"]] = item
    normalized = list(unique_sources.values())
    external_agreement = len(normalized) >= 2 and all(
        _matches(left, right)
        for index, left in enumerate(normalized)
        for right in normalized[index + 1:]
    )

    local_unit_basis = "explicit" if local and local.get("volume_unit") in {"shares", "lots"} else "unresolved"
    if local and local.get("volume_unit") == "unknown" and external_agreement:
        # Legacy daily partitions omitted the volume unit.  Resolve it only
        # when two independent, pairwise-agreeing daily sources match price,
        # amount and exactly one plausible volume scale.  This labels the
        # comparison; it never mutates the stored daily row.
        scale_candidates = []
        for scale_name, scale in (("shares", 1.0), ("lots", 100.0)):
            probe = {**local, "volume": float(local["volume"]) * scale, "volume_unit": "shares"}
            if all(_matches(probe, item) for item in normalized):
                scale_candidates.append((scale_name, scale))
        if len(scale_candidates) == 1:
            scale_name, scale = scale_candidates[0]
            local = {
                **local,
                "volume": float(local["volume"]) * scale,
                "volume_unit": "shares",
                "local_volume_unit": scale_name,
                "local_volume_unit_basis": "external_daily_consensus",
            }
            local_unit_basis = "external_daily_consensus"
    # Recompute both matches after resolving an unknown legacy daily volume
    # unit.  Computing them before the comparison-only conversion would turn a
    # real minute turnover conflict into the less actionable ``no_consensus``.
    matched_sources = [item["source"] for item in normalized if _matches(minute, item)]
    local_sources = [item["source"] for item in normalized if _matches(local, item)]
    errors = [dict(item) for item in source_errors]
    if invalid_reasons:
        errors.extend({"reason": reason, "retryable": False} for reason in invalid_reasons)

    retryable_errors = [item for item in errors if item.get("reason") in RETRYABLE_SOURCE_ERRORS or item.get("retryable")]
    max_retry_after = max((int(item.get("retry_after_seconds") or 0) for item in retryable_errors), default=0)
    identity_conflict = "identity_conflict" in invalid_reasons or any(
        item.get("reason") == "identity_conflict" for item in errors
    )
    if identity_conflict:
        status, reason, write_allowed = "quarantine", "identity_conflict", False
    elif local is not None and local.get("volume_unit") == "unknown":
        status, reason, write_allowed = "quarantine", "local_daily_unit_unknown", False
    elif external_agreement and len(matched_sources) >= 2 and not _matches(local, normalized[0]):
        status, reason, write_allowed = "daily_baseline_repair_candidate", "independent_daily_agreement", False
    elif external_agreement and len(matched_sources) >= 2 and _matches(local, normalized[0]):
        status, reason, write_allowed = "minute_import_candidate", "independent_daily_agreement", True
    elif external_agreement and len(local_sources) >= 2 and len(matched_sources) < 2:
        # The published daily baseline is corroborated by independent daily
        # sources, while the stored minute aggregate differs.  Keep the
        # minute record quarantined and let the next minute source try; never
        # replace the daily baseline merely because a minute source disagrees.
        status, reason, write_allowed = "qmt_turnover_conflict", "daily_agreement_minute_conflict", False
    elif len(normalized) == 1 and local_sources:
        status, reason, write_allowed = "qmt_turnover_conflict", "local_daily_agreement", False
    elif len(normalized) >= 2 and not external_agreement:
        status, reason, write_allowed = "quarantine", "no_external_consensus", False
    elif not normalized and retryable_errors:
        status, reason, write_allowed = "source_unavailable", "third_party_source_unavailable", False
    elif not normalized and errors and all(item.get("reason") == "source_no_record" for item in errors):
        status, reason, write_allowed = "source_no_record", "third_party_source_no_record", False
    else:
        status, reason, write_allowed = "quarantine", "no_consensus", False

    if existing_values_conflict and write_allowed:
        status, reason, write_allowed = "existing_values_conflict", "existing_values_conflict", False
    return {
        "symbol": expected_symbol,
        "date": expected_date,
        "status": status,
        "reason": reason,
        "write_allowed": write_allowed,
        "retryable": status == "source_unavailable",
        "retry_after_seconds": max_retry_after,
        "matched_sources": matched_sources,
        "local_agreement_sources": local_sources,
        "source_errors": errors,
        "sources_checked": [item["source"] for item in normalized],
        "minute_rows": int(minute.get("rows") or 0) if minute else 0,
        "local_volume_unit": (local.get("local_volume_unit") if local else None),
        "local_volume_unit_basis": local_unit_basis,
    }
