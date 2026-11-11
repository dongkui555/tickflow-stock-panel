"""Reproduce a frozen pilot, then diagnose groups and market bias without promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services import l2_diagnostics as diagnostics
from app.services import l2_experiment as experiment


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, allow_nan=False)


def _verify_inputs(snapshot: dict, execution: dict) -> dict:
    for path, expected in snapshot["partitions"].items():
        file = Path(path)
        if file.stat().st_size != expected["bytes"] or _sha(file) != expected["sha256"]:
            raise ValueError(f"Frozen baseline input changed: {path}")
    for path, expected in execution["extraction_manifests"].items():
        if _sha(Path(path)) != expected:
            raise ValueError(f"Frozen extraction manifest changed: {path}")
    return {
        "baseline_partitions_verified": len(snapshot["partitions"]),
        "extraction_manifests_verified": len(execution["extraction_manifests"]),
    }


def _tables(report: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows, state_rows = [], []
    metrics = (
        "rows",
        "dates",
        "rmse",
        "mae",
        "rank_ic",
        "direction_accuracy",
        "mean_target",
        "mean_prediction",
        "bias_prediction_minus_target",
        "majority_direction_reference",
        "predicted_positive_fraction",
        "top_k_forward_return",
        "top_k_excess",
        "top_k_after_cost_proxy",
    )
    for horizon, values in report["horizons"].items():
        for arm, details in values["arms"].items():
            for split, modes in details["splits"].items():
                for mode, result in modes.items():
                    key = {
                        "horizon": int(horizon),
                        "arm": arm,
                        "split": split,
                        "calibration": mode,
                    }
                    for scope in ("summary", "sufficient_cross_section"):
                        rows.append(
                            {
                                **key,
                                "scope": scope,
                                **{m: result[scope].get(m) for m in metrics},
                            }
                        )
                    for state, state_result in result["by_market_state"].items():
                        state_rows.append(
                            {
                                **key,
                                "market_state": state,
                                **{m: state_result.get(m) for m in metrics},
                            }
                        )
    return pl.DataFrame(rows), pl.DataFrame(state_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--extra-research-dir", type=Path, action="append", default=[])
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--baseline-snapshot", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="NEW isolated research directory"
    )
    args = parser.parse_args(argv)
    try:
        reference_report = args.reference_dir / "report.json"
        reference_predictions = args.reference_dir / "paired_predictions.parquet"
        saved = _json(reference_report)
        values = dict(saved["config"])
        for key in (
            "history_start",
            "train_start",
            "validation_start",
            "test_start",
            "end",
        ):
            values[key] = date.fromisoformat(values[key])
        for key in ("horizons", "symbols", "l2_features"):
            values[key] = tuple(values[key])
        config = experiment.L2ExperimentConfig(**values)
        settings = diagnostics.DiagnosticConfig()
        snapshot = _json(args.baseline_snapshot)
        execution = _json(args.reference_dir / "execution_inputs.json")
        verification = _verify_inputs(snapshot, execution)
        output = args.output_dir.resolve()
        research = args.research_dir.resolve()
        if not output.is_relative_to(research) or output == research or output.exists():
            raise ValueError("Output must be a NEW directory inside research-dir")
        output.mkdir(parents=True, exist_ok=False)
        code_paths = [
            Path(__file__).resolve(),
            Path(experiment.__file__),
            Path(diagnostics.__file__),
        ]
        plan = {
            **diagnostics.diagnostic_plan(settings),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "experiment_config": saved["config"],
            "inputs": {
                str(p.resolve()): _sha(p)
                for p in (
                    reference_report,
                    reference_predictions,
                    args.baseline_snapshot,
                    args.reference_dir / "execution_inputs.json",
                )
            },
            "code": {str(p): _sha(p) for p in code_paths},
            "research_roots": [
                str(p.resolve()) for p in [args.research_dir, *args.extra_research_dir]
            ],
        }
        _write(output / "plan.json", plan)
        print("Frozen plan saved; loading verified compact inputs", flush=True)
        panel, l2 = experiment.load_l2_experiment_inputs(
            args.data_dir,
            config,
            research_dir=args.research_dir,
            extra_research_dirs=tuple(args.extra_research_dir),
        )
        print(
            f"Inputs: {panel.height} daily rows / {l2.height} L2 rows; evaluating 14 arms",
            flush=True,
        )
        result = diagnostics.evaluate_l2_diagnostics(panel, l2, config, settings)
        if not result.reference.predictions.equals(
            pl.read_parquet(reference_predictions)
        ):
            raise ValueError(
                "Frozen pilot predictions differ; do not interpret new contrasts"
            )
        verification.update(_verify_inputs(snapshot, execution))
        for path, expected in {**plan["inputs"], **plan["code"]}.items():
            if _sha(Path(path)) != expected:
                raise ValueError(f"Input/code changed during diagnostics: {path}")
        _write(output / "report.json", result.report)
        result.predictions.write_parquet(output / "predictions.parquet")
        result.market.write_parquet(output / "market_history.parquet")
        metrics, states = _tables(result.report)
        metrics.write_csv(output / "metrics.csv")
        states.write_csv(output / "market_state_metrics.csv")
        verification.update(
            {
                "status": "complete",
                "prior_predictions_exact": True,
                "reference_arms_exact": True,
                "production_writes": 0,
                "test_status": result.report["test_status"],
                "artifacts": {p.name: _sha(p) for p in output.iterdir() if p.is_file()},
            }
        )
        _write(output / "verification.json", verification)
        print(
            json.dumps(
                {
                    "status": "diagnostics_complete",
                    "output": str(output),
                    **verification,
                },
                indent=2,
            )
        )
        return 0
    except (ValueError, OSError, KeyError, pl.exceptions.PolarsError) as exc:
        print(
            json.dumps({"status": "diagnostics_incomplete", "error": str(exc)}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
