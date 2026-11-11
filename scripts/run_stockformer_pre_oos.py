"""Select five-day features and exits before holdout, or validate a new window."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.stockformer_pre_oos import select_pre_oos_policy, validate_new_window  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--selection", type=Path, help="Frozen selection report to validate on a new window")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    if args.selection:
        result = validate_new_window(data_dir, args.selection.resolve(), output)
    elif args.model_id:
        result = select_pre_oos_policy(
            data_dir, model_id=args.model_id, output_path=output,
            on_progress=lambda index, total, family: print(
                f"[{index}/{total}] training family: {family}", flush=True
            ),
        )
    else:
        parser.error("provide --model-id for selection or --selection for new-window validation")
    print(json.dumps({
        "output": str(output) if output.exists() else None,
        "status": result.get("status") or result.get("new_oos", {}).get("status"),
        "selected": {key: result.get("selected", {}).get(key)
                     for key in ("feature_family", "exit_mode", "compounded_return_pct")},
        "training_pass": result.get("training_pass"),
        "new_oos": result.get("new_oos"),
        "available_sessions": result.get("available_sessions"),
        "required_sessions": result.get("required_sessions"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
