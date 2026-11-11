"""Run sector OOS research without importing the application/job registry."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--train-days", type=int, default=80)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--holdout-days", type=int, default=40)
    parser.add_argument(
        "--experiment",
        choices=("baseline", "early_layout", "stock_entry"),
        default="baseline",
    )
    args = parser.parse_args()
    from app.services.sector_research import run

    result = run(
        args.data_dir,
        lookback_days=args.lookback_days,
        train_days=args.train_days,
        test_days=args.test_days,
        holdout_days=args.holdout_days,
        experiment=args.experiment,
        on_progress=lambda stage, pct, msg: print(
            f"{pct:3d}% {stage} {msg}", flush=True
        ),
    )
    print(args.data_dir / "research" / "sector_forecasts" / f"{result['id']}.json")


if __name__ == "__main__":
    main()
