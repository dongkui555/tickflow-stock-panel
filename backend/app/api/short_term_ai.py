"""Short-term AI recommendation API."""
from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.services.global_risk import build_global_risk
from app.services.sector_prediction import (
    list_sector_prediction_records,
    refresh_sector_prediction_records,
)
from app.services.sector_layout import build_sector_rotation
from app.services.short_term_ai import build_short_term_agent_review, build_short_term_recommendations

router = APIRouter(prefix="/api/short-term-ai", tags=["short-term-ai"])


@router.get("/workflow")
def prediction_workflow(request: Request):
    from app.services.prediction_workflow import status
    return status(request.app.state.repo.store.data_dir, getattr(request.app.state, "wecom_bot_service", None))


class WorkflowConfigRequest(BaseModel):
    enabled: bool = True
    notify: bool = True
    chat_id: str | None = Field(default=None, max_length=256)


@router.post("/workflow/config")
def prediction_workflow_config(request: Request, req: WorkflowConfigRequest):
    from app.services import preferences
    updates = {"prediction_workflow_enabled": req.enabled, "prediction_workflow_notify": req.notify}
    if req.chat_id is not None:
        chat_id = req.chat_id.strip()
        bot = getattr(request.app.state, "wecom_bot_service", None)
        known = bot.status().get("chats", []) if bot else []
        if chat_id and not any(c.get("chatid") == chat_id and c.get("kind") == "single" for c in known):
            raise HTTPException(400, "请先向智能机器人发送单聊消息, 再选择已识别的接收用户")
        updates.update(prediction_workflow_chat_id=chat_id,
                       prediction_workflow_chat_bot_id=preferences.get_wecom_bot_id())
    preferences.save(updates)
    return prediction_workflow(request)


class AgentReviewRequest(BaseModel):
    model_id: str | None = None
    as_of: date | None = None
    limit: int = Field(default=30, ge=1, le=100)
    profile: str = Field(default="balanced", pattern="^(balanced|aggressive|conservative)$")
    review_count: int = Field(default=5, ge=1, le=10)


class SectorPredictionRefreshRequest(BaseModel):
    as_of: date | None = None
    days: int = Field(default=8, ge=1, le=60)
    lookback_days: int = Field(default=70, ge=25, le=160)
    limit: int = Field(default=30, ge=5, le=100)


class SectorResearchRequest(BaseModel):
    experiment: Literal["baseline", "early_layout", "stock_entry"] = "baseline"
    lookback_days: int = Field(default=252, ge=160, le=504)
    train_days: int = Field(default=80, ge=40, le=200)
    test_days: int = Field(default=20, ge=10, le=60)
    holdout_days: int = Field(default=40, ge=20, le=80)
    initial_capital: float = Field(default=100000, ge=10000, le=10000000)


@router.get("/sector-research")
def sector_research_result(request: Request, experiment: Literal["baseline", "early_layout", "stock_entry"] = "baseline"):
    from app.services.sector_research import latest
    return latest(request.app.state.repo.store.data_dir, experiment=experiment)


@router.post("/sector-research")
async def sector_research_start(request: Request, body: SectorResearchRequest):
    import asyncio
    import logging
    from app.api.pipeline import _long_task_executor
    from app.services.pipeline_jobs import job_store, try_acquire_run_slot, release_run_slot, JobCancelledError
    from app.services.sector_research import run
    overhead = {"baseline": 30, "early_layout": 36, "stock_entry": 38}[body.experiment]
    if body.lookback_days < body.train_days + body.holdout_days + overhead:
        raise HTTPException(400, "历史窗口不足以覆盖训练、隔离和留出期")
    job_id, is_new = job_store.create(long_running=True)
    if not is_new:
        raise HTTPException(409, "已有数据/研究任务运行，请完成后再开始实验")

    def task():
        if not try_acquire_run_slot(job_id):
            job_store.fail(job_id, "数据任务正在运行，请稍后重试")
            return
        try:
            job_store.start(job_id)
            result = run(request.app.state.repo.store.data_dir,
                         lookback_days=body.lookback_days, train_days=body.train_days,
                         test_days=body.test_days, holdout_days=body.holdout_days,
                         capital=body.initial_capital,
                         experiment=body.experiment,
                         on_progress=lambda stage, pct, msg: job_store.progress(job_id, stage, pct, msg))
            job_store.succeed(job_id, {"research_id": result["id"]})
        except JobCancelledError:
            pass
        except Exception as exc:
            logging.getLogger(__name__).exception("sector research failed")
            job_store.fail(job_id, str(exc))
        finally:
            release_run_slot(job_id)
    asyncio.get_running_loop().run_in_executor(_long_task_executor, task)
    return {"job_id": job_id}


@router.get("/recommendations")
def recommendations(
    request: Request,
    model_id: str | None = Query(None),
    as_of: date | None = Query(None),
    limit: int = Query(30, ge=1, le=100),
    profile: str = Query("balanced", pattern="^(balanced|aggressive|conservative)$"),
):
    """Combined short-term recommendation center."""
    return build_short_term_recommendations(
        request.app.state.repo,
        getattr(request.app.state, "quote_service", None),
        getattr(request.app.state, "minute_refresh", None),
        getattr(request.app.state, "strategy_engine", None),
        getattr(request.app.state, "monitor_engine", None),
        model_id=model_id,
        as_of=as_of,
        limit=limit,
        profile=profile,
    )


@router.get("/sector-rotation")
def sector_rotation(
    request: Request,
    as_of: date | None = Query(None),
    lookback_days: int = Query(70, ge=25, le=160),
    level: int = Query(2, ge=1, le=3),
    limit: int = Query(30, ge=1, le=100),
):
    """行业板块轮动提前预判：识别资金开始布局但涨幅未过热的方向。"""
    return build_sector_rotation(
        request.app.state.repo.store.data_dir,
        as_of=as_of,
        lookback_days=lookback_days,
        level=level,
        limit=limit,
    )


@router.get("/sector-predictions")
def sector_predictions(
    request: Request,
    as_of: date | None = Query(None),
    days: int = Query(8, ge=1, le=60),
    limit: int = Query(30, ge=5, le=100),
    origin: str = Query("published", pattern="^(published|replay|legacy)$"),
):
    """持久化板块提前预判记录: T日预测, T+1/T+2 验证。"""
    return list_sector_prediction_records(
        request.app.state.repo.store.data_dir,
        as_of=as_of,
        days=days,
        limit=limit,
        origin=origin,
    )


@router.post("/sector-predictions/refresh")
def refresh_sector_predictions(request: Request, req: SectorPredictionRefreshRequest):
    """生成/更新板块提前预判记录。"""
    data_dir = request.app.state.repo.store.data_dir
    return refresh_sector_prediction_records(
        data_dir,
        days=req.days,
        as_of=req.as_of,
        lookback_days=req.lookback_days,
        limit=req.limit,
    )


@router.get("/global-risk")
def global_risk(
    request: Request,
    force_refresh: bool = Query(False),
    include_news: bool = Query(True),
):
    """隔夜全球风险因子：美股/中概/汇率/波动率，用作A股盘前风险闸门。"""
    return build_global_risk(
        request.app.state.repo,
        force_refresh=force_refresh,
        include_news=include_news,
    )


@router.post("/agent-review")
async def agent_review(request: Request, req: AgentReviewRequest):
    """LLM review for top short-term recommendation candidates."""
    return await build_short_term_agent_review(
        request.app.state.repo,
        getattr(request.app.state, "quote_service", None),
        getattr(request.app.state, "minute_refresh", None),
        getattr(request.app.state, "strategy_engine", None),
        getattr(request.app.state, "monitor_engine", None),
        model_id=req.model_id,
        as_of=req.as_of,
        limit=req.limit,
        profile=req.profile,
        review_count=req.review_count,
    )
