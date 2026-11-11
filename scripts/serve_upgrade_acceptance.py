"""Serve the real application with automatic jobs disabled for upgrade acceptance.

Use only an independent data copy. Business APIs, providers, repositories,
factor calculations and execution rules retain their application implementations.
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--static-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3019)
    args = parser.parse_args()
    if args.port == 3018:
        parser.error("port 3018 is reserved for the production service")
    root = args.data_dir.resolve(strict=True)
    if "tick-stock-panel-v022-integrated" in root.parts:
        parser.error("the production data directory cannot be used for acceptance")
    os.environ["DATA_DIR"] = str(root)
    os.environ["TICKFLOW_ENV_FILE"] = os.environ.get("TICKFLOW_ENV_FILE", "/tmp/no-env")
    os.environ["WATCHDOG_ENABLED"] = "false"
    os.environ["BACKTEST_MATRIX_CACHE_PREWARM"] = "false"
    if args.static_dir:
        os.environ["STATIC_DIR"] = str(args.static_dir.resolve(strict=True))
    backend = Path(__file__).resolve().parents[1] / "backend"
    if backend.is_dir():
        sys.path.insert(0, str(backend))

    import uvicorn

    # Disable only automatic startup entry points. Manual requests exercise
    # the real code and publish exclusively into the independent data copy.
    automatic_hooks = (
        "app.jobs.daily_pipeline.start_scheduler",
        "app.services.quote_service.QuoteService.boot_check",
        "app.services.depth_service.DepthService.boot_check",
        "app.services.depth_service.DepthService.start_polling",
        "app.services.minute_refresh.MinuteRefreshService.start",
        "app.services.data_integrity.boot_integrity_check",
        "app.services.wecom_bot_service.WecomBotService.boot_check",
        "app.services.ext_pull.pull_scheduler.start",
        "app.services.ext_pull.pull_scheduler.refresh",
        "app.services.financial_sync.financial_scheduler.start",
    )
    with ExitStack() as stack:
        for target in automatic_hooks:
            stack.enter_context(patch(target, return_value=None))
        uvicorn.run("app.main:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
