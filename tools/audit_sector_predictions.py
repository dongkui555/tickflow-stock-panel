"""Read-only audit of stored sector forecasts; never regenerate predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import mean


def metrics(rows: list[dict], cutoff: str, horizon: int | None = None) -> dict:
    samples = []
    for row in rows:
        valid = {
            v["horizon_days"]: v for v in row.get("validations", [])
            if v.get("status") in {"hit", "miss"}
            and v.get("verify_date", "9999") < cutoff
        }
        if horizon is None:
            if set(valid) != {1, 2}:
                continue
            hit = any(v["status"] == "hit" for v in valid.values())
            samples.append((hit, None))
        elif horizon in valid:
            v = valid[horizon]
            samples.append((v["status"] == "hit", v.get("actual_ret_1d")))
    returns = [r for _, r in samples if r is not None]
    return {
        "n": len(samples), "hits": sum(h for h, _ in samples),
        "hit_rate": mean(h for h, _ in samples) if samples else None,
        "mean_actual_day_return": mean(returns) if returns else None,
    }


def audit(snapshots: list[dict], cutoff: str) -> dict:
    ordered = sorted(snapshots, key=lambda s: s["as_of"])
    all_rows = [dict(r, as_of=s["as_of"]) for s in ordered for r in s["rows"]]
    top10 = [r for s in ordered for r in s["rows"][:10]]
    top5 = [r for s in ordered for r in s["rows"][:5]]
    strong = [r for r in all_rows if r.get("prediction_level") == "strong"]
    groups = {
        "all_stored": all_rows, "top5": top5, "top10": top10, "strong": strong,
        "top10_low_breadth_le_25pct": [r for r in top10 if (r.get("up_ratio") or 0) <= .25],
        "top10_breadth_gt_25pct": [r for r in top10 if (r.get("up_ratio") or 0) > .25],
        "strong_without_positive_pre3_or_pre5_flow": [r for r in strong if (r.get("pre_net_3d") or 0) <= 0 and (r.get("pre_net_5d") or 0) <= 0],
    }
    result = {
        "scope": "Stored/reconstructed predictions only, not a prospective backtest. Excludes validation dates on/after cutoff; no forecasts are rewritten.",
        "cutoff_exclusive": cutoff,
        "snapshots": len(ordered),
        "generated_after_as_of_date": sum(s.get("generated_at", "")[:10] > s["as_of"] for s in ordered),
        "daily": [],
        "groups": {name: {"T1": metrics(rows, cutoff, 1), "T2_day_only": metrics(rows, cutoff, 2), "either_day_fully_matured": metrics(rows, cutoff)} for name, rows in groups.items()},
        "score_bands": {},
    }
    actual = []
    for s in ordered:
        summary = s.get("validation_summary") or {}
        if (summary.get("verify_1d_date") or "9999") < cutoff:
            actual.extend(summary.get("actual_top10_1d", []))
        result["daily"].append({
            "as_of": s["as_of"], "generated_at": s.get("generated_at"),
            "stored": len(s["rows"]), "strong": s["summary"].get("strong_count"),
            "top10_T1": metrics(s["rows"][:10], cutoff, 1),
            "top10_either_day": metrics(s["rows"][:10], cutoff),
            "top10_T1_excess": summary.get("top10_excess_1d"),
            "nextday_missed": [a["sector"] for a in summary.get("missed_actual_top10_1d", [])],
            "fund_end": s.get("source_meta", {}).get("moneyflow", {}).get("date_end"),
        })
    result["nextday_actual_top10_recall"] = dict(Counter(a.get("verdict") for a in actual))
    for low, high in [(0, 64), (64, 72), (72, 80), (80, 90), (90, 101)]:
        subset = [r for r in all_rows if low <= (r.get("prediction_score") or 0) < high]
        result["score_bands"][f"{low}-{high}"] = metrics(subset, cutoff, 1)
    result["strong_component_means"] = {
        k: mean(r["component_scores"][k] for r in strong)
        for k in strong[0]["component_scores"]
    } if strong else {}
    result["top10_examples"] = [r for r in all_rows if r["as_of"] in {"2026-09-01", "2026-09-07"} and r.get("prediction_rank", 999) <= 3]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--cutoff", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.store.read_bytes()
    snapshots = json.loads(raw)
    report = audit(snapshots, args.cutoff.isoformat())
    report["source_sha256"] = hashlib.sha256(raw).hexdigest()
    report["source_unchanged"] = args.store.read_bytes() == raw
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        if args.output.resolve() == args.store.resolve():
            raise ValueError("Output must not overwrite source predictions")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
