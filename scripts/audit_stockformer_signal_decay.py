"""Audit five-day signal decay by minute features, score and market regime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.stockformer_signal_audit import audit_signal_decay  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit_signal_decay(
        args.data_dir.resolve(), args.model_id, args.output.resolve(),
        on_progress=lambda family: print(f"audited {family}", flush=True),
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "training_rank_ic": report["training_rank_ic"],
        "families": {key: value["summary"].get("all", {})
                     for key, value in report["families"].items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
