"""Run only the local paired L2 research experiment, never the raw extractor."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from polars.exceptions import PolarsError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.l2_experiment import (
    L2_FEATURE_COLUMNS,
    L2ExperimentConfig,
    run_l2_experiment,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Research-only paired OHLCV ridge vs OHLCV+L2 ridge; no extraction/backtest.",
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--research-dir",
        type=Path,
        help="Isolated L2 feature/universe/output directory; defaults to data-dir/l2_research",
    )
    parser.add_argument(
        "--extra-research-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional validated extraction batch; overlapping stock/date rows fail closed",
    )
    for name in (
        "history-start",
        "train-start",
        "validation-start",
        "test-start",
        "end",
    ):
        parser.add_argument(
            f"--{name}",
            required=True,
            type=date.fromisoformat,
            help="ISO information-date boundary; end also caps label exit dates",
        )
    parser.add_argument("--horizons", default="1,3,5,10")
    universe_args = parser.add_mutually_exclusive_group()
    universe_args.add_argument(
        "--symbols", default="", help="Comma-separated predeclared fixed universe"
    )
    universe_args.add_argument(
        "--universe-file",
        type=Path,
        help="Frozen extractor universe.json; defaults to its standard path when present",
    )
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--min-history-days", type=int, default=60)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--round-trip-cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--l2-features",
        default=",".join(L2_FEATURE_COLUMNS),
        help="Comma-separated subset of the explicit 13-feature whitelist",
    )
    parser.add_argument(
        "--run-id",
        help="Save a NEW research-dir/experiments/<id> directory; no overwrite",
    )
    args = parser.parse_args(argv)
    try:
        research_root = (args.research_dir or args.data_dir / "l2_research").resolve()
        symbols = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
        universe_file = args.universe_file or research_root / "universe.json"
        if not symbols and (args.universe_file or universe_file.exists()):
            universe = json.loads(universe_file.read_text(encoding="utf-8"))
            values = universe.get("symbols") if isinstance(universe, dict) else None
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(s, str) for s in values)
            ):
                raise ValueError("Invalid frozen universe file")
            symbols = tuple(values)
        config = L2ExperimentConfig(
            history_start=args.history_start,
            train_start=args.train_start,
            validation_start=args.validation_start,
            test_start=args.test_start,
            end=args.end,
            horizons=tuple(int(h.strip()) for h in args.horizons.split(",")),
            symbols=symbols,
            l2_features=tuple(s.strip() for s in args.l2_features.split(",")),
            max_symbols=args.max_symbols,
            min_history_days=args.min_history_days,
            ridge_alpha=args.ridge_alpha,
            top_k=args.top_k,
            round_trip_cost_bps=args.round_trip_cost_bps,
        )
        output = None
        if args.run_id:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run_id):
                raise ValueError(
                    "Invalid run-id: use 1-64 ASCII letters, digits, _, - or ."
                )
            output = (research_root / "experiments" / args.run_id).resolve()
            if not output.is_relative_to(research_root) or output.exists():
                raise ValueError("Output must be a NEW directory inside research-dir")
        result = run_l2_experiment(
            args.data_dir,
            config,
            research_dir=research_root,
            extra_research_dirs=tuple(args.extra_research_dir),
        )
        if output is None:
            print(json.dumps(result.report, indent=2, allow_nan=False))
        else:
            output.mkdir(parents=True, exist_ok=False)
            with (output / "report.json").open("x", encoding="utf-8") as handle:
                json.dump(result.report, handle, indent=2, allow_nan=False)
            result.predictions.write_parquet(output / "paired_predictions.parquet")
            print(
                json.dumps(
                    {
                        "status": "research_complete",
                        "output": str(output),
                        "diagnostics": result.report["diagnostics"],
                    },
                    indent=2,
                )
            )
    except (ValueError, OSError, PolarsError) as exc:
        print(
            json.dumps(
                {
                    "status": "experiment_not_run_or_incomplete",
                    "research_only": True,
                    "error": str(exc),
                },
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
