"""盘后管道 API — 异步触发 + 进度跟踪。"""
from __future__ import annotations

import asyncio
import concurrent.futures as _cf
import logging
from datetime import date

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.jobs import daily_pipeline
from app.services.pipeline_jobs import (
    JobCancelledError,
    job_store,
    release_run_slot,
    run_with_capacity,
    try_acquire_run_slot,
)
from app.api.data import invalidate_storage_cache

# 长时间任务专用线程池（隔离于 FastAPI 默认线程池，防止阻塞请求处理）
_long_task_executor = _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="long-task")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


class MarketRepairRequest(BaseModel):
    start_date: date
    end_date: date
    datasets: list[str] = Field(default_factory=lambda: ["daily", "moneyflow", "auction"])
    max_requests: int = Field(default=120, ge=1, le=1000)
    use_local_minutes: bool = False
    rebuild_enriched: bool = False


class MinuteAuthorizationRequest(BaseModel):
    task_id: str = Field(min_length=1)
    ids: list[str] = Field(min_length=1)


@router.get("/after-close-status")
def after_close_status(request: Request) -> dict:
    import json
    from app.services import preferences

    schedule = preferences.get_full_sync_schedule()
    path = request.app.state.repo.store.data_dir / "user_data" / "after_close_sync.json"
    runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    evening_path = path.with_name("evening_sync.json")
    evening_runs = json.loads(evening_path.read_text(encoding="utf-8")) if evening_path.exists() else {}
    from app.services.close_data_sync import read_status
    return {"time": schedule["after_close"], "timezone": "Asia/Shanghai",
            "scope": "all", "mode": "incremental", "retry_minutes": 5,
            "failure_retry_minutes": 15, "max_attempts": 3,
            "minute_enabled": preferences.get_minute_sync_enabled(), "runs": runs,
            "dataset_retry_minutes": 30, "moneyflow_publication_hour": 19,
            "evening_time": schedule["evening"], "evening_runs": evening_runs,
            "dataset_runs": read_status(request.app.state.repo.store.data_dir)}


@router.get("/maintenance-queue")
def maintenance_queue_status(request: Request) -> dict:
    from app.services.maintenance_queue import status
    data_dir = request.app.state.repo.store.data_dir
    from app.services import minute_gap_queue
    return {"tasks": status(data_dir), "minute_gap_queue": minute_gap_queue.summary(data_dir)}


@router.get("/maintenance-queue/authorizations")
def maintenance_queue_authorizations(request: Request, task_id: str | None = None) -> dict:
    from app.services import minute_repair_authorization

    snapshot = minute_repair_authorization.snapshot(request.app.state.repo.store.data_dir)
    if task_id:
        snapshot["pending"] = [row for row in snapshot["pending"] if row["task_id"] == task_id]
        snapshot["audit"] = [row for row in snapshot["audit"] if row.get("task_id") == task_id]
    return snapshot


@router.post("/maintenance-queue/authorizations/approve")
def approve_maintenance_authorizations(request: Request, body: MinuteAuthorizationRequest) -> dict:
    from app.services import maintenance_queue, minute_repair_authorization

    try:
        approved = minute_repair_authorization.approve(
            request.app.state.repo.store.data_dir, body.ids, task=body.task_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    resumed = maintenance_queue.resume_authorized(
        request.app.state.repo.store.data_dir, body.task_id,
    )
    return {"approved": approved, "resumed_tasks": resumed}


@router.post("/sync-close-data")
def sync_close_data(request: Request) -> dict:
    from app.market_time import cn_now
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None or scheduler.get_job("close_dataset_retry") is None:
        raise HTTPException(503, "盘后分项同步调度器尚未启动")
    scheduler.modify_job("close_dataset_retry", next_run_time=cn_now())
    return {"queued": True, "active_job_id": job_store.active_id(),
            "status_url": "/api/pipeline/after-close-status"}


@router.post("/maintenance-queue/audit-regime")
def audit_regime_queue(request: Request) -> dict:
    from app.services.maintenance_queue import audit_regime
    owner = "regime-queue-audit"
    if not try_acquire_run_slot(owner):
        raise HTTPException(409, "数据任务正在写入, 请在空闲后核验")
    try:
        return audit_regime(request.app.state.repo.store.data_dir)
    finally:
        release_run_slot(owner)


@router.post("/repair-market")
async def repair_market(request: Request, body: MarketRepairRequest) -> dict:
    from app.market_time import cn_now
    from app.services.market_repair import repair_market_gaps
    from app.services.tushare_ext_sync import EXT_SPECS

    if body.start_date > body.end_date or body.end_date >= cn_now().date():
        raise HTTPException(400, "仅修复已经结束的历史交易日")
    if not body.datasets or set(body.datasets) - {"daily", "moneyflow", "auction", *EXT_SPECS}:
        raise HTTPException(400, "未知数据类型")
    if body.use_local_minutes and body.datasets != ["daily"]:
        raise HTTPException(400, "本地分钟聚合只支持补日K")
    job_id, is_new = job_store.create(long_running=True)
    if not is_new:
        return {"job_id": job_id, "reused": True}

    def task():
        if not try_acquire_run_slot(job_id):
            job_store.fail(job_id, "已有数据任务占用执行槽")
            return
        try:
            job_store.start(job_id)
            def progress(stage, pct, msg, **kw):
                job_store.progress(job_id, stage, pct, msg, **kw)
            result = repair_market_gaps(
                request.app.state.repo, body.start_date, body.end_date,
                datasets=body.datasets, max_requests=body.max_requests,
                use_local_minutes=body.use_local_minutes,
                on_progress=progress,
            )
            if body.rebuild_enriched:
                from app.indicators.pipeline import run_pipeline
                repo = request.app.state.repo
                repo.clear_cache()
                result["enriched_rows"] = run_pipeline(
                    data_dir=repo.store.data_dir,
                    on_batch_done=lambda cur, total: progress(
                        "rebuild_enriched", 98, f"重建历史指标 {cur}/{total}",
                    ),
                )
                repo.rebuild_views()
                repo.refresh_cache(background=True)
            if result["complete"]:
                job_store.succeed(job_id, result)
            else:
                job_store.fail(job_id,
                    f"部分完成：新增 {result['rows_added']} 行；仍有缺口，"
                    f"详见 import_reports/{result['run_id']}/summary.json")
        except JobCancelledError:
            logger.info("market repair cancelled: %s", job_id)
        except Exception as exc:
            logger.exception("market repair failed")
            job_store.fail(job_id, str(exc))
        finally:
            invalidate_storage_cache()
            release_run_slot(job_id)

    asyncio.get_running_loop().run_in_executor(_long_task_executor, task)
    return {"job_id": job_id, "reused": False}


@router.post("/run")
async def run_now(request: Request) -> dict:
    """异步触发盘后管道,立即返回 job_id。客户端轮询 /jobs/{id} 拿进度。"""
    return await trigger_pipeline_job(
        request.app.state.repo,
        request.app.state.capabilities,
        getattr(request.app.state, "quote_service", None),
    )


async def trigger_pipeline_job(repo, capset, quote_service=None) -> dict:
    """触发盘后管道后台任务(HTTP 端点与 AI 助手共用同一条触发路径)。

    若已有任务在跑,**返回该任务 id 而不是开新任务**(防止并发拉数据撞限流)。
    卡死判定按「进度停滞」而非总时长(慢带宽下长任务不会被误杀), 见 reap_stale。
    """
    return await _submit_job(
        repo, quote_service,
        lambda progress: (lambda: daily_pipeline.run_now(repo, capset, on_progress=progress)),
    )


async def _submit_job(repo, quote_service, make_run) -> dict:
    """任务化执行公共机制: 单飞 create → 重任务槽 → 行情暂停 → 缓存失效。

    make_run(progress) 返回实际执行闭包(阻塞 IO + CPU, 在 executor 里跑)。
    盘后管道与独立除权因子同步共用 —— 两者写同一批 parquet, 必须共享
    单飞与重任务槽防并发覆写。
    """
    # 检测卡死的 running job (如 reload 后孤儿 task / 网络读无限阻塞)。
    # reap_stale 会在 /run 和 /jobs/{id} 轮询端点都调用,保证卡死后能自愈。
    job_store.reap_stale()

    # 单飞: 复用任何活跃 (pending∨running) 任务, is_new=False 时不再调度新任务
    job_id, is_new = job_store.create()
    if not is_new:
        return {"job_id": job_id, "reused": True}

    def progress(stage: str, pct: int, msg: str, stage_pct: int | None = None,
                 skip_log: bool = False) -> None:
        job_store.progress(job_id, stage, pct, msg, stage_pct=stage_pct, skip_log=skip_log)

    run = make_run(progress)

    # 在 executor 里跑同步任务(pipeline 内部都是阻塞 IO + CPU)
    async def task() -> None:
        # 重任务执行槽: 防僵尸并发(reap 后线程仍活时新任务不得并行写 parquet)
        if not try_acquire_run_slot(job_id):
            job_store.fail(job_id, "已有数据任务在运行(或上一次任务卡死未结束),请稍后再试")
            return
        # 任务运行期间暂停实时行情取数, 防止覆写同一批 parquet 竞态
        qs = quote_service
        try:
            loop = asyncio.get_event_loop()

            def _run() -> dict:
                try:
                    if qs:
                        with qs.paused():
                            return run()
                    return run()
                finally:
                    repo.refresh_cache()

            result = await loop.run_in_executor(_long_task_executor, run_with_capacity, job_id, _run)
            job_store.succeed(job_id, result)
            invalidate_storage_cache()
        except JobCancelledError:
            # 已被 reap/手动取消终止: job 状态已由 terminate() 写为 failed,
            # 拉取线程在分块回调处自行退出, 这里无需(也无法)再写状态。
            logger.warning("pipeline job %s cancelled", job_id)
        except Exception as e:  # noqa: BLE001
            logger.exception("pipeline job failed")
            job_store.fail(job_id, str(e))
            invalidate_storage_cache()
        finally:
            release_run_slot(job_id)

    asyncio.create_task(task())
    return {"job_id": job_id, "reused": False}


@router.post("/adj-factor/run")
async def run_adj_factor_now(request: Request) -> dict:
    """独立同步除权因子(全历史 + 受影响个股 enriched 局部重算), 任务化+进度轮询。

    与盘后管道共用单飞与重任务槽 (两者写同一批 parquet); 门控与矩阵
    usable 同口径: TickFlow 档位不足且未路由到其他可用源时 400。
    """
    from app.services import preferences
    from app.tickflow.capabilities import Cap

    capset = request.app.state.capabilities
    if not (capset.has(Cap.ADJ_FACTOR) or preferences.get_adj_factor_provider() != "tickflow"):
        raise HTTPException(
            status_code=400,
            detail="除权因子能力不可用: 请先在 设置 → 数据源 中配置可用数据源",
        )
    repo = request.app.state.repo
    return await _submit_job(
        repo,
        getattr(request.app.state, "quote_service", None),
        lambda progress: (lambda: daily_pipeline.run_adj_factor_sync(repo, capset, on_progress=progress)),
    )


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    # 每次轮询都检查卡死 job — 前端持续轮询, 进度停滞超阈值后必定自愈,
    # 无需用户再次手动点「同步」。
    job_store.reap_stale()
    j = job_store.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    return j


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """手动取消一个 running 的 job(协作式: 拉取线程在当前分块完成后自行退出)。"""
    j = job_store.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    if j["status"] not in ("running", "pending"):
        raise HTTPException(status_code=400, detail=f"job status is {j['status']}, cannot cancel")
    job_store.terminate(job_id, "用户手动取消")
    return {"cancelled": job_id}


@router.get("/jobs")
def list_jobs(limit: int = 20) -> dict:
    return {
        "active_id": job_store.active_id(),
        "jobs": job_store.list_recent(limit=limit),
    }
