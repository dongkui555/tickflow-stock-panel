"""Run frozen five-day StockFormer shadow tracking."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.stockformer_paper import run_shadow_tracking  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--as-of", type=date.fromisoformat)
    args = parser.parse_args(argv)
    result = run_shadow_tracking(
        args.data_dir.resolve(), args.selection.resolve(), args.output.resolve(), as_of=args.as_of,
    )
    print(json.dumps({
        "output": str(args.output.resolve()), "status": result["status"],
        "available_sessions": result["available_sessions"],
        "signal_count": result.get("signal_count", 0),
        "activation_eligible": result["activation_eligible"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
