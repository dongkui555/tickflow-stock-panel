"""AI Agents 增强 API。

这些端点是 TickFlow 内置 AI Agents 能力的稳定接入层：
多源行情、资金流、海外市场/新闻、结构化多 Agent 研究结果、持仓优化和模拟盘。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.aiagents_bridge import (
    analyze,
    data_health,
    legacy_holdings,
    market_context,
    paper_store,
)

router = APIRouter(prefix="/api/aiagents", tags=["aiagents"])


class ContextRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    include_fund_flow: bool = True
    include_news: bool = True
    include_overseas: bool = True
    fund_flow_days: int = Field(60, ge=5, le=120)


class AnalysisRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    include_fund_flow: bool = True
    include_news: bool = True


class PaperOrderRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    side: str
    quantity: int = Field(..., gt=0)
    price: float | None = Field(default=None, gt=0)
    reason: str = ""


@router.get("/health")
def health():
    return data_health()


@router.post("/context")
def context(req: ContextRequest):
    result = market_context(
        req.symbol,
        include_fund_flow=req.include_fund_flow,
        include_news=req.include_news,
        include_overseas=req.include_overseas,
        fund_flow_days=req.fund_flow_days,
    )
    return result


@router.post("/analyze")
def structured_analysis(req: AnalysisRequest):
    return analyze(
        req.symbol,
        include_fund_flow=req.include_fund_flow,
        include_news=req.include_news,
    )


@router.get("/holdings")
def holdings():
    return legacy_holdings()


@router.get("/paper")
def paper_snapshot():
    return paper_store.snapshot()


@router.post("/paper/order")
def paper_order(req: PaperOrderRequest):
    try:
        return paper_store.order(req.symbol, req.side, req.quantity, req.price, req.reason)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/quote")
def quote(symbol: str = Query(..., min_length=1)):
    result = market_context(symbol, include_fund_flow=False, include_news=False, include_overseas=False)
    return {
        "symbol": result["symbol"],
        "quote": result["quote"],
        "technical": result["technical"],
        "generated_at": result["generated_at"],
    }
