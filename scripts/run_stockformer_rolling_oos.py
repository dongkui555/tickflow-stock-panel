"""Run the research-only rolling 5-day StockFormer portfolio validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.stockformer_rolling import run_rolling_portfolio_research


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-id")
    parser.add_argument(
        "--feature-family",
        choices=(
            "full", "compact", "short_reversal", "trend", "price_volume",
            "daily_ohlcv", "daily_plus_moneyflow", "daily_plus_auction",
            "daily_plus_minute", "daily_full",
        ),
        help="Override the pre-OOS selected feature family for an exploratory rolling run",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Override ridge regularization for an exploratory rolling run",
    )
    parser.add_argument(
        "--forecast-floor",
        type=float,
        default=None,
        help="Override the frozen calibrated prediction floor for exploratory runs",
    )
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--window-days", type=int, default=63)
    parser.add_argument("--min-window-days", type=int, default=30)
    parser.add_argument("--required-positive-windows", type=int, default=3)
    parser.add_argument("--minimum-trades-per-window", type=int, default=3)
    parser.add_argument(
        "--training-lookback-months", type=int, choices=(12, 15, 18), default=18,
        help="每个滚动窗口只使用窗口开始前的有限历史重新拟合",
    )
    parser.add_argument(
        "--min-net-edge", type=float, default=0.0005,
        help="扣除估算往返成本后仍需保留的最小收益优势",
    )
    parser.add_argument(
        "--turnover-penalty", type=float, default=0.0005,
        help="新候选相对近期持仓的换手惩罚",
    )
    parser.add_argument(
        "--holding-lag-days", type=int, default=1,
        help="候选排序中保留近期候选的天数",
    )
    parser.add_argument(
        "--frozen-oos-predictions",
        action="store_true",
        help="Use the historical frozen OOS predictions instead of refitting before each window",
    )
    parser.add_argument("--skip-final-backtest", action="store_true")
    parser.add_argument(
        "--report-name",
        help="Write the report under this experiments filename; default keeps rolling_oos_latest.json",
    )
    args = parser.parse_args(argv)

    result = run_rolling_portfolio_research(
        args.data_dir.resolve(),
        model_id=args.model_id,
        feature_family=args.feature_family,
        alpha=args.alpha,
        forecast_floor=args.forecast_floor,
        max_positions=args.max_positions,
        initial_capital=args.initial_capital,
        window_days=args.window_days,
        min_window_days=args.min_window_days,
        required_positive_windows=args.required_positive_windows,
        minimum_trades_per_window=args.minimum_trades_per_window,
        training_lookback_months=args.training_lookback_months,
        min_net_edge=args.min_net_edge,
        turnover_penalty=args.turnover_penalty,
        holding_lag_days=args.holding_lag_days,
        refit_each_window=not args.frozen_oos_predictions,
        run_final_backtest=not args.skip_final_backtest,
        report_filename=args.report_name,
        on_progress=lambda stage, pct, message: print(
            f"[{pct:3d}%] {stage}: {message}", flush=True
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
