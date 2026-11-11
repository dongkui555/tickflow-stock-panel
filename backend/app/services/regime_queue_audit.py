"""Evidence-only reconciliation of legacy market-history work; never writes prices/results."""

from __future__ import annotations

import math
from collections import Counter
from datetime import date
from pathlib import Path

import polars as pl

from app.services import regime_builder

VERSION = 1
VERIFIED_REASONS = {"input_changed", "missing_result", "manual_recompute"}


def requires_review(options):
    # force without an explicit manual request is not evidence of an algorithm upgrade.
    return options.get("reason") not in VERIFIED_REASONS


def classify(data_dir, days):
    root = Path(data_dir)
    result_path = regime_builder.regime_path(root)
    # A corrupt result must not masquerade as an empty history to trigger a full rebuild.
    history = pl.read_parquet(result_path) if result_path.exists() else pl.DataFrame()
    if not history.is_empty() and "date" not in history.columns:
        raise ValueError("regime_result_schema_invalid")
    records = {}
    for row in history.iter_rows(named=True):
        records.setdefault(str(row["date"]), []).append(row)
    state = regime_builder._source_state(root)
    findings = []
    for day in sorted(set(days)):
        finding = {"date": day, "decision": "review"}
        try:
            date.fromisoformat(day)
            current = regime_builder._source_stamp(root, day)
            previous = state.get("dates", {}).get(day)
            finding.update(previous_sources=previous, current_sources=current)
            sources = [root / p for p in current if p.replace("\\", "/").startswith("kline_daily_enriched/")]
            old = records.get(day, [])
            if not sources:
                finding["reason"] = "input_missing"
            elif not old:
                # Only missing results need a narrow input readability check. No historical
                # indicators are recalculated during this audit.
                scan = pl.scan_parquet(sources, hive_partitioning=False)
                required = {"date", "symbol", "close", "amount"}
                if not required.issubset(scan.collect_schema().names()):
                    finding["reason"] = "input_schema_invalid"
                else:
                    stats = scan.select(
                        pl.len().alias("rows"),
                        ((pl.col("date").cast(pl.String) == day)
                         & pl.col("close").is_finite() & (pl.col("close") > 0)
                         & pl.col("amount").is_finite() & (pl.col("amount") >= 0)
                         & pl.col("symbol").is_not_null()).fill_null(False).all().alias("valid"),
                    ).collect().row(0, named=True)
                    finding["input_rows"] = stats["rows"]
                    if stats["rows"] and stats["valid"]:
                        finding.update(decision="keep", reason="result_missing")
                    else:
                        finding["reason"] = "input_invalid"
            elif len(old) != 1 or not _valid_result(old[0]):
                finding["reason"] = "result_requires_validation"
            elif previous and previous == current:
                finding.update(decision="skip", reason="same_recorded_sources")
            else:
                finding["reason"] = "source_version_changed_unproven" if previous else "no_source_baseline"
            if current != regime_builder._source_stamp(root, day):
                finding.update(decision="review", reason="input_changed_during_audit")
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            finding.update(decision="review", reason="input_unreadable", error=type(exc).__name__)
        findings.append(finding)
    return findings


def _valid_result(row):
    score = row.get("score")
    up, down = row.get("up_count"), row.get("down_count")
    counts_valid = all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (up, down))
    return (isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 100
            and counts_valid and up + down > 0)


def reconcile(data_dir, rows, *, recheck=False):
    """Mutate the supplied queue copy only. Caller owns queue/execution locks and backup."""
    reports = []
    owners = {d: t["id"] for t in rows if t.get("audit_version") == VERSION
              for d in t.get("review_items", [])}
    for task in rows:
        if (task["kind"] != "regime" or task["status"] == "running"
                or not requires_review(task.get("original_options", task.get("options", {})))
                or (task.get("audit_version") == VERSION and not recheck)
                or task["status"] not in {"pending", "retry", "paused_review"}):
            continue
        days = sorted(set(task["items"]) | set(task.get("review_items", [])))
        if not days:
            continue
        try:
            findings = classify(data_dir, days)
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            findings = [{"date": d, "decision": "review", "reason": "audit_baseline_unreadable",
                         "error": type(exc).__name__} for d in days]
        for finding in findings:
            day = finding["date"]
            if finding["decision"] == "review":
                if day in owners and owners[day] != task["id"]:
                    finding.update(decision="linked", reason="already_under_review", task_id=owners[day])
                else:
                    owners[day] = task["id"]
        counts = dict(Counter(f["decision"] for f in findings))
        task["items"] = [f["date"] for f in findings if f["decision"] == "keep"]
        task["review_items"] = [f["date"] for f in findings if f["decision"] == "review"]
        task["skipped_items"] = sorted(set(task.get("skipped_items", [])) | {
            f["date"] for f in findings if f["decision"] == "skip"})
        task["linked_review_items"] = {**task.get("linked_review_items", {}),
                                     **{f["date"]: f["task_id"] for f in findings if f["decision"] == "linked"}}
        task.update(audit_version=VERSION, audit_counts=counts, batch_progress=0,
                    status="pending" if task["items"] else "paused_review" if task["review_items"] else "reviewed",
                    message=f"已核验: 需补算 {len(task['items'])} 日, 无需重复 {len(task['skipped_items'])} 日, 待核验 {len(task['review_items'])} 日, 并入其他核验 {len(task['linked_review_items'])} 日; 待核验不自动重算",
                    error=None)
        if task["items"]:
            task.setdefault("original_options", task.get("options", {}))
            task["options"] = {"reason": "missing_result"}
        task.pop("retry_at", None)
        reports.append({"task_id": task["id"], "counts": counts, "dates": findings})
    return reports
