"""StockFormer integration API."""
from __future__ import annotations

import asyncio
import logging
from datetime import date

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.pipeline import _long_task_executor
from app.services.pipeline_jobs import (
    JobCancelledError,
    job_store,
    release_run_slot,
    try_acquire_run_slot,
)
from app.services.stockformer import (
    StockFormerTrainConfig,
    factor_analysis,
    list_models,
    multifactor_select,
    predict,
    strict_oos_backtest,
    train_model,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stockformer", tags=["stockformer"])


@router.get("/daily-shadow")
def daily_shadow_status(request: Request):
    from app.services.stockformer_daily_shadow import status

    result = status(request.app.state.repo.store.data_dir)
    scheduler = getattr(request.app.state, "scheduler", None)
    job = scheduler.get_job("stockformer_daily_shadow") if scheduler else None
    result["scheduler"] = {"registered": job is not None,
                           "next_run_at": str(job.next_run_time) if job else None}
    return result


class TrainRequest(BaseModel):
    start: date | None = None
    end: date | None = None
    sample_out_ratio: float = Field(0.30, ge=0.10, le=0.60)
    max_symbols: int = Field(1000, ge=20, le=2000)
    ridge_alpha: float = Field(1.0, ge=0.0, le=100.0)
    target_mode: str = Field("industry_relative", pattern="^(absolute|industry_relative|net|industry_relative_net)$")
    training_profile: str = Field("daily_baseline", pattern="^(daily_baseline|minute_enhanced)$")


class MultiPeriodRequest(BaseModel):
    start: date
    end: date
    max_symbols: int = Field(50, ge=20, le=200)


class TransformerExperimentRequest(BaseModel):
    start: date | None = None
    end: date | None = None
    max_symbols: int = Field(200, ge=20, le=1000)
    epochs: int = Field(8, ge=1, le=30)


class ModelComparisonRequest(BaseModel):
    start: date | None = None
    end: date | None = None
    max_symbols: int = Field(2000, ge=20, le=3000)
    sample_out_ratio: float = Field(0.30, ge=0.20, le=0.50)


@router.get("/model-comparison")
def model_comparison_latest(request: Request):
    from app.services.stockformer_model_comparison import latest
    return latest(request.app.state.repo.store.data_dir)


@router.post("/model-comparison")
def model_comparison_run(req: ModelComparisonRequest, request: Request):
    from app.services.stockformer_model_comparison import run
    if req.start and req.end and req.start >= req.end:
        raise HTTPException(status_code=422, detail="开始日期必须早于结束日期")
    try:
        return run(request.app.state.repo.store.data_dir, **req.model_dump())
    except Exception as exc:  # noqa: BLE001
        logger.exception("StockFormer model comparison failed")
        return {"error": f"模型对照实验失败: {type(exc).__name__}: {exc}"}


@router.get("/transformer-experiment")
def transformer_experiment_latest(request: Request):
    from app.services.transformer_experiment import latest_experiment
    return latest_experiment(request.app.state.repo.store.data_dir)


@router.post("/transformer-experiment")
def transformer_experiment_run(req: TransformerExperimentRequest, request: Request):
    from app.services.transformer_experiment import run_experiment
    if req.start and req.end and req.start >= req.end:
        raise HTTPException(status_code=422, detail="开始日期必须早于结束日期")
    return run_experiment(request.app.state.repo.store.data_dir, **req.model_dump())


@router.get("/multiperiod")
def multiperiod_report(request: Request):
    from app.services.multiperiod_experiment import latest_report
    return latest_report(request.app.state.repo.store.data_dir)


@router.post("/multiperiod")
async def multiperiod_run(req: MultiPeriodRequest, request: Request):
    from app.services.multiperiod_experiment import run_multiperiod
    from app.services.pipeline_jobs import JobCancelledError, job_store, try_acquire_run_slot, release_run_slot
    if req.start >= req.end:
        raise HTTPException(status_code=422, detail="开始日期必须早于结束日期")
    job_id, created = job_store.create(long_running=True)
    if not created:
        raise HTTPException(status_code=409, detail="已有数据任务运行，请完成后再开始实验")
    if not try_acquire_run_slot(job_id):
        job_store.fail(job_id, "数据执行槽忙碌")
        raise HTTPException(status_code=409, detail="数据执行槽忙碌，请稍后再试")
    def run():
        try:
            job_store.start(job_id)
            result = run_multiperiod(request.app.state.repo, **req.model_dump(),
                                    on_progress=lambda stage, pct, msg: job_store.progress(job_id, stage, pct, msg))
            job_store.succeed(job_id, result)
        except JobCancelledError:
            logger.info("multiperiod experiment cancelled: %s", job_id)
        except Exception as exc:
            logger.exception("multiperiod experiment failed")
            job_store.fail(job_id, str(exc))
        finally:
            release_run_slot(job_id)
    asyncio.create_task(asyncio.to_thread(run))
    return {"job_id": job_id}


class PredictRequest(BaseModel):
    model_id: str | None = None
    symbols: list[str] | None = None
    as_of: date | None = None
    limit: int = Field(100, ge=1, le=5000)


class FactorRequest(BaseModel):
    model_id: str | None = None
    horizon: int = Field(5, ge=1, le=10)
    n_groups: int = Field(5, ge=3, le=10)


class MultiFactorRequest(BaseModel):
    model_id: str | None = None
    as_of: date | None = None
    limit: int = Field(100, ge=1, le=1000)


class StrictOOSRequest(BaseModel):
    model_id: str | None = None
    max_positions: int = Field(2, ge=1, le=20)
    holding_days: int = Field(3, ge=1, le=30)
    initial_capital: float = Field(10000.0, ge=1000)
    candidate_symbols: list[str] | None = None


class PortfolioResearchRequest(BaseModel):
    model_id: str | None = None
    max_positions: int = Field(3, ge=1, le=20)
    initial_capital: float = Field(10000.0, ge=1000)


@router.get("/models")
def models(request: Request):
    return list_models(request.app.state.repo.store.data_dir)


@router.post("/train")
async def train(req: TrainRequest, request: Request):
    """Start training in the long-task executor and return immediately.

    The model computation can take several minutes for a large universe.  Keep
    the HTTP request short-lived and expose progress through the shared job
    endpoint at ``/api/pipeline/jobs/{job_id}``.
    """
    job_id, is_new = job_store.create(long_running=True)
    if not is_new:
        raise HTTPException(status_code=409, detail="已有数据或研究任务运行，请完成后再开始训练")

    data_dir = request.app.state.repo.store.data_dir
    config = StockFormerTrainConfig(**req.model_dump())

    def task() -> None:
        if not try_acquire_run_slot(job_id):
            job_store.fail(job_id, "已有数据任务运行，训练暂未启动，请稍后重试")
            return
        try:
            job_store.start(job_id)

            def progress(stage: str, pct: int, msg: str) -> None:
                job_store.progress(job_id, stage, pct, msg)

            progress("init", 2, "训练任务已启动")
            result = train_model(
                data_dir,
                config,
                activate=False,
                on_progress=progress,
            )
            if result.get("error"):
                job_store.fail(job_id, str(result["error"]))
                return

            # 预测明细已经写入模型目录; job 记录只保留页面需要的摘要,
            # 避免把数千行预测结果重复写入 job_store JSON。
            predictions = result.get("predictions") or {}
            job_store.succeed(job_id, {
                "model": result.get("model"),
                "predictions_count": len(predictions.get("rows") or []),
            })
        except JobCancelledError:
            logger.info("StockFormer training cancelled: %s", job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("StockFormer train failed")
            job_store.fail(job_id, f"StockFormer 训练失败: {type(exc).__name__}: {exc}")
        finally:
            release_run_slot(job_id)

    asyncio.get_running_loop().run_in_executor(_long_task_executor, task)
    return {"job_id": job_id, "reused": False}


@router.post("/predict")
def model_predict(req: PredictRequest, request: Request):
    try:
        return predict(
            request.app.state.repo.store.data_dir,
            model_id=req.model_id,
            symbols=req.symbols,
            as_of=req.as_of,
            limit=req.limit,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("StockFormer predict failed")
        return {"error": f"StockFormer 预测失败: {type(exc).__name__}: {exc}", "rows": []}


@router.post("/inference-job")
async def model_inference_job(req: PredictRequest, request: Request):
    job_id, created = job_store.create(long_running=True)
    if not created:
        raise HTTPException(status_code=409, detail="已有后台任务运行，请完成后刷新预测")
    data_dir = request.app.state.repo.store.data_dir

    def task() -> None:
        if not try_acquire_run_slot(job_id):
            job_store.fail(job_id, "已有数据任务运行，预测暂未启动，请稍后刷新")
            return
        try:
            job_store.start(job_id)
            job_store.progress(job_id, "load_features", 10, "加载特征并生成校准预测")
            predictions = predict(data_dir, model_id=req.model_id, symbols=req.symbols,
                                  as_of=req.as_of, limit=0, write_artifact=False)
            if predictions.get("error"):
                raise ValueError(predictions["error"])
            job_store.progress(job_id, "rank_candidates", 65, "计算入场规则与综合排名")
            multifactor = multifactor_select(data_dir, model_id=req.model_id,
                                            as_of=req.as_of, limit=req.limit,
                                            prediction_result=predictions)
            if multifactor.get("error"):
                raise ValueError(multifactor["error"])
            displayed = dict(predictions)
            if req.limit > 0:
                displayed["rows"] = predictions["rows"][:req.limit]
                displayed["total"] = len(displayed["rows"])
            job_store.succeed(job_id, {"model_id": predictions.get("model", {}).get("model_id"),
                                       "limit": req.limit, "predictions": displayed,
                                       "multifactor": multifactor})
        except JobCancelledError:
            logger.info("StockFormer inference cancelled: %s", job_id)
        except Exception as exc:
            logger.exception("StockFormer inference failed")
            job_store.fail(job_id, f"StockFormer 预测失败: {type(exc).__name__}: {exc}")
        finally:
            release_run_slot(job_id)

    asyncio.get_running_loop().run_in_executor(_long_task_executor, task)
    return {"job_id": job_id, "reused": False}


@router.post("/factor-analysis")
def model_factor_analysis(req: FactorRequest, request: Request):
    try:
        return factor_analysis(
            request.app.state.repo.store.data_dir,
            model_id=req.model_id,
            horizon=req.horizon,
            n_groups=req.n_groups,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("StockFormer factor analysis failed")
        return {"error": f"StockFormer 因子分析失败: {type(exc).__name__}: {exc}"}


@router.post("/multifactor")
def model_multifactor(req: MultiFactorRequest, request: Request):
    try:
        return multifactor_select(
            request.app.state.repo.store.data_dir,
            model_id=req.model_id,
            as_of=req.as_of,
            limit=req.limit,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("StockFormer multifactor failed")
        return {"error": f"StockFormer 多因子选股失败: {type(exc).__name__}: {exc}", "rows": []}


@router.post("/strict-oos")
async def model_strict_oos(req: StrictOOSRequest, request: Request):
    """Start the expensive OOS report in the shared background executor."""
    job_id, created = job_store.create(long_running=True)
    if not created:
        raise HTTPException(status_code=409, detail="已有后台任务运行，请完成后再验证样本外")
    data_dir = request.app.state.repo.store.data_dir

    def task() -> None:
        if not try_acquire_run_slot(job_id):
            job_store.fail(job_id, "已有数据任务运行，样本外评估暂未启动，请稍后重试")
            return
        try:
            job_store.start(job_id)

            def progress(stage: str, pct: int, msg: str) -> None:
                job_store.progress(job_id, stage, pct, msg)

            result = strict_oos_backtest(
                data_dir,
                model_id=req.model_id,
                max_positions=req.max_positions,
                holding_days=req.holding_days,
                initial_capital=req.initial_capital,
                candidate_symbols=req.candidate_symbols,
                on_progress=progress,
            )
            job_store.succeed(job_id, result)
        except JobCancelledError:
            logger.info("StockFormer strict OOS cancelled: %s", job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("StockFormer strict OOS failed")
            job_store.fail(job_id, f"StockFormer 严格样本外回测失败: {type(exc).__name__}: {exc}")
        finally:
            release_run_slot(job_id)

    asyncio.get_running_loop().run_in_executor(_long_task_executor, task)
    return {"job_id": job_id, "reused": False}


@router.post("/portfolio-research")
async def model_portfolio_research(req: PortfolioResearchRequest, request: Request):
    from app.services.stockformer_portfolio import run_portfolio_research

    job_id, created = job_store.create(long_running=True)
    if not created:
        raise HTTPException(status_code=409, detail="已有后台任务运行，请完成后再执行组合回测")
    data_dir = request.app.state.repo.store.data_dir

    def task() -> None:
        if not try_acquire_run_slot(job_id):
            job_store.fail(job_id, "已有数据任务运行，真实组合回测暂未启动，请稍后重试")
            return
        try:
            job_store.start(job_id)

            def progress(stage: str, pct: int, msg: str) -> None:
                job_store.progress(job_id, stage, pct, msg)

            result = run_portfolio_research(
                data_dir,
                model_id=req.model_id,
                max_positions=req.max_positions,
                initial_capital=req.initial_capital,
                on_progress=progress,
            )
            job_store.succeed(job_id, result)
        except JobCancelledError:
            logger.info("StockFormer portfolio research cancelled: %s", job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("StockFormer portfolio research failed")
            job_store.fail(job_id, f"StockFormer 真实组合回测失败: {type(exc).__name__}: {exc}")
        finally:
            release_run_slot(job_id)

    asyncio.get_running_loop().run_in_executor(_long_task_executor, task)
    return {"job_id": job_id, "reused": False}
