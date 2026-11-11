"""Run leakage-safe L2 ablation and freeze a research candidate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.l2_experiment import L2ExperimentConfig, load_l2_experiment_inputs
from app.services.l2_model_selection import audit_2026_archives, run_model_selection
from app.services.stockformer import _json_safe


def _date(value: str) -> date:
    return date.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--research-dir", required=True, type=Path)
    parser.add_argument("--extra-research-dir", action="append", type=Path, default=[])
    parser.add_argument("--universe-file", required=True, type=Path)
    parser.add_argument("--history-start", required=True, type=_date)
    parser.add_argument("--train-start", required=True, type=_date)
    parser.add_argument("--validation-start", required=True, type=_date)
    parser.add_argument("--test-start", required=True, type=_date)
    parser.add_argument("--end", required=True, type=_date)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fresh-archive-root", type=Path)
    parser.add_argument("--fresh-start", type=_date, default=date(2026, 1, 1))
    parser.add_argument("--fresh-end", type=_date, default=date(2026, 12, 31))
    args = parser.parse_args(argv)

    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run_id):
            raise ValueError("Invalid run-id")
        output = (args.research_dir / "experiments" / args.run_id).resolve()
        if not output.is_relative_to(args.research_dir.resolve()) or output.exists():
            raise ValueError("Output must be a new directory inside research-dir/experiments")
        universe = json.loads(args.universe_file.read_text("utf-8"))
        symbols = universe.get("symbols") if isinstance(universe, dict) else None
        if symbols is None and isinstance(universe, dict):
            frozen = universe.get("universe")
            symbols = frozen.get("symbols") if isinstance(frozen, dict) else None
        if not isinstance(symbols, list) or len(symbols) != 200:
            raise ValueError("Frozen universe must contain exactly 200 symbols")
        config = L2ExperimentConfig(
            history_start=args.history_start,
            train_start=args.train_start,
            validation_start=args.validation_start,
            test_start=args.test_start,
            end=args.end,
            symbols=tuple(symbols),
            max_symbols=200,
            min_history_days=60,
            top_k=args.top_k,
            round_trip_cost_bps=10.0,
        )
        panel, l2 = load_l2_experiment_inputs(
            args.data_dir,
            config,
            research_dir=args.research_dir,
            extra_research_dirs=tuple(args.extra_research_dir),
        )
        frozen, scored, regimes = run_model_selection(panel, l2, config)
        output.mkdir(parents=True, exist_ok=False)
        (output / "frozen_model.json").write_text(
            json.dumps(_json_safe(frozen), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        scored.write_parquet(output / "frozen_predictions.parquet")
        regimes.write_parquet(output / "market_regimes.parquet")

        readiness = None
        if args.fresh_archive_root:
            market_dates = (
                pl.scan_parquet(str(args.data_dir / "kline_daily_enriched" / "**" / "*.parquet"))
                .filter(pl.col("date").is_between(args.fresh_start, args.fresh_end))
                .select("date")
                .unique()
                .sort("date")
                .collect(engine="streaming")["date"]
                .to_list()
            )
            readiness = audit_2026_archives(
                args.fresh_archive_root,
                market_dates,
                start=args.fresh_start,
                end=args.fresh_end,
            )
            (output / "fresh_oos_readiness.json").write_text(
                json.dumps(readiness, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            (output / "missing_fresh_l2_dates.txt").write_text(
                "\n".join(readiness["missing_dates"]) + "\n",
                encoding="ascii",
            )

        print(
            json.dumps(
                {
                    "status": "research_model_frozen",
                    "output": str(output),
                    "candidate": frozen["candidate"],
                    "selected_l2_features": frozen["selected_l2_features"],
                    "production_eligible": frozen["production_eligible"],
                    "fresh_oos": readiness,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        print(
            json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
