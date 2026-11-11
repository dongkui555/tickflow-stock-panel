"""资金监控 API — 日级主力资金榜 + 盘中资金雷达。"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.services.fund_monitor import build_daily_fund_monitor, build_intraday_fund_monitor

router = APIRouter(prefix="/api/fund-monitor", tags=["fund-monitor"])


@router.get("/daily")
def daily_fund_monitor(
    request: Request,
    window: int = Query(20, ge=1, le=60, description="最近 N 个资金流交易日"),
    limit: int = Query(100, ge=1, le=6000),
    min_score: float = Query(0.0, ge=0.0, le=100.0),
    signal: str | None = Query(None, description="all/main_accumulation/early_layout/low_absorption/strong_inflow/inflow_confirm/watch/distribution_risk/outflow"),
):
    """日级主力资金榜: 基于本地 Tushare moneyflow 的盘后确认和主力建仓/吸筹识别。"""
    repo = request.app.state.repo
    quote_service = getattr(request.app.state, "quote_service", None)
    return build_daily_fund_monitor(
        repo,
        quote_service,
        window=window,
        limit=limit,
        min_score=min_score,
        signal=signal,
    )


@router.get("/intraday")
def intraday_fund_monitor(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    min_score: float = Query(0.0, ge=0.0, le=100.0),
    signal: str | None = Query(None, description="all/active_inflow/breakout_flow/watch/distribution_risk/outflow"),
):
    """盘中资金雷达: 实时行情 + 分钟K方向估算 + 日级 moneyflow 确认。"""
    repo = request.app.state.repo
    quote_service = getattr(request.app.state, "quote_service", None)
    minute_refresh = getattr(request.app.state, "minute_refresh", None)
    return build_intraday_fund_monitor(
        repo,
        quote_service,
        minute_refresh,
        limit=limit,
        min_score=min_score,
        signal=signal,
    )
