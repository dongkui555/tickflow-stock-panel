"""Durable close-to-open predictions; never retrain or submit orders."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from app.market_time import CN_TZ, cn_now

_LOCK = threading.Lock()
logger = logging.getLogger(__name__)
MIN_COVERAGE = 0.98


def _root(data_dir):
    return Path(data_dir) / "user_data" / "prediction_workflow"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def config():
    from app.services import preferences
    p = preferences.load()
    return {"enabled": p.get("prediction_workflow_enabled", True),
            "notify": p.get("prediction_workflow_notify", True),
            "webhook_configured": bool(preferences.get_wecom_webhook_url()),
            "notification_channel": "wecom_smart_bot",
            "bot_configured": bool(preferences.get_wecom_bot_id() and preferences.get_wecom_bot_secret()),
            "bot_enabled": preferences.get_wecom_bot_enabled(),
            "chat_id": p.get("prediction_workflow_chat_id", ""),
            "chat_bot_id": p.get("prediction_workflow_chat_bot_id", "")}


def _frame(data_dir, table, day):
    files = list((Path(data_dir) / table / f"date={day}").glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.scan_parquet(files).filter(pl.col("date").cast(pl.Date) == day).collect()


def _symbols(frame):
    return {str(s).split(".")[0] for s in frame["symbol"].drop_nulls()} if "symbol" in frame.columns else set()


def _valid_prices(frame):
    required = ["open", "high", "low", "close", "volume", "amount"]
    if frame.is_empty() or not set(required).issubset(frame.columns):
        return frame.clear()
    return frame.filter(
        pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in required])
        & (pl.col("high") >= pl.max_horizontal("open", "close", "low"))
        & (pl.col("low") <= pl.min_horizontal("open", "close"))
    )


def _closed_prices(frame, day):
    frame = _valid_prices(frame)
    if "quote_ts" not in frame.columns:
        return frame
    cutoff = datetime.combine(day, datetime.min.time(), tzinfo=CN_TZ).replace(hour=15).timestamp() * 1000
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=CN_TZ).timestamp() * 1000
    stamp = pl.col("quote_ts").cast(pl.Float64, strict=False)
    # Null identifies batch daily data; intraday snapshots must reach the close.
    return frame.filter(pl.col("quote_ts").is_null() | (stamp.is_finite() & (stamp >= cutoff) & (stamp < end)))


def readiness(data_dir, day):
    """Check row dates and non-null values, not task completion or folder names."""
    raw = _closed_prices(_frame(data_dir, "kline_daily", day), day)
    expected = _symbols(raw)
    previous = sorted(p.name[5:] for p in (Path(data_dir) / "kline_daily").glob("date=????-??-??")
                      if p.name[5:] < str(day))
    prior = _symbols(_valid_prices(_frame(data_dir, "kline_daily", date.fromisoformat(previous[-1])))) if previous else set()
    enriched_frame = _closed_prices(_frame(data_dir, "kline_daily_enriched", day), day)
    if "raw_close" in enriched_frame.columns and not raw.is_empty():
        from app.indicators.pipeline import _normalize_raw_volume_unit

        key = pl.col("symbol").cast(pl.String).str.split(".").list.first().alias("_key")
        # Compare the same share volume that produced enriched; legacy daily
        # partitions can contain lots. This transforms only the validation frame.
        current = _normalize_raw_volume_unit(raw).with_columns(key).select("_key", pl.col("close").alias("_close"),
                                               pl.col("volume").alias("_volume"), pl.col("amount").alias("_amount"))
        enriched_frame = enriched_frame.with_columns(key).join(current, on="_key", how="inner").filter(
            pl.all_horizontal([(pl.col(a) - pl.col(b)).abs() <= pl.col(b).abs() * 1e-6 + 1e-8
                               for a, b in (("raw_close", "_close"), ("volume", "_volume"), ("amount", "_amount"))]))
    enriched = _symbols(enriched_frame)
    flows = _frame(data_dir, "moneyflow", day)
    flow_cols = [c for c in ["buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"] if c in flows.columns]
    good_flows = flows.filter(pl.all_horizontal([pl.col(c).is_finite() for c in flow_cols])) if len(flow_cols) == 4 else flows.clear()
    funds = _symbols(good_flows)
    from app.services.index_const import CORE_INDEX_SYMBOLS
    indices = _symbols(_closed_prices(_frame(data_dir, "kline_index_daily", day), day))
    core = {s.split(".")[0] for s in CORE_INDEX_SYMBOLS}
    checks = {
        "日K": bool(expected) and bool(prior) and len(expected) >= len(prior) * MIN_COVERAGE,
        "技术特征": bool(expected) and len(expected & enriched) >= len(expected) * MIN_COVERAGE,
        "资金": bool(expected) and len(expected & funds) >= len(expected) * MIN_COVERAGE,
        "核心指数": core.issubset(indices),
    }
    from app.services.maintenance_queue import status as maintenance_status
    checks["复权历史一致性"] = not any(
        task["kind"] == "adjustment" and task["status"] != "succeeded"
        for task in maintenance_status(data_dir)
    )
    valid = expected & enriched & funds
    daily_frame = _frame(data_dir, "kline_daily", day)
    halted = _symbols(daily_frame.filter(pl.col("volume") == 0)) if "volume" in daily_frame.columns else set()
    missing = {"daily": sorted(prior - expected - halted), "enriched": sorted(expected - enriched),
               "moneyflow": sorted(expected - funds), "index": sorted(core - indices)}
    signature = []
    for table in ("kline_daily", "kline_daily_enriched", "moneyflow", "kline_index_daily"):
        for p in sorted((Path(data_dir) / table / f"date={day}").glob("*.parquet")):
            signature.append((table, p.name, hashlib.sha256(p.read_bytes()).hexdigest()))
    return {"date": str(day), "ready": all(checks.values()), "checks": checks,
            "input_version": hashlib.sha256(json.dumps(signature).encode()).hexdigest(),
            "counts": {"daily": len(expected), "previous_daily": len(prior),
                       "enriched": len(expected & enriched), "moneyflow": len(expected & funds)},
            "minimum_coverage": MIN_COVERAGE, "eligible_symbols": sorted(valid),
            "missing": missing,
            "reason": "数据就绪" if all(checks.values()) else "等待同步/校验: " + "、".join(k for k, v in checks.items() if not v)}


def retry_missing_data(repo, capset, day, *, now=None, on_progress=None):
    """Repair one close only, sharing checkpoints with the existing sync worker."""
    from app.indicators.pipeline import run_date_pipeline
    from app.services import close_data_sync, kline_sync

    now = now or cn_now()
    emit = on_progress or (lambda *args: None)
    data_dir = repo.store.data_dir
    state = close_data_sync.read_status(data_dir)
    results = []
    touched = False
    baseline = readiness(data_dir, day)
    if not baseline["counts"]["previous_daily"]:
        return {"status": "needs_attention", "datasets": [],
                "reason": "缺少历史日K基线, 请先完成首次历史同步; 自动补差不会全量重拉",
                "next_retry_at": (now + timedelta(minutes=30)).isoformat()}
    for dataset, label in (("daily", "日K"), ("enriched", "技术特征"),
                           ("index", "核心指数"), ("moneyflow", "主力资金")):
        ready = readiness(data_dir, day)
        missing = set(ready["missing"][dataset])
        if not missing:
            continue
        item = state.setdefault(str(day), {}).setdefault(dataset, {})
        if item.get("retry_at", "") > now.isoformat():
            results.append({"dataset": label, **item})
            continue
        attempts = item.get("attempts", 0) + 1
        delay = (5, 15, 30, 60, 120)[min(attempts - 1, 4)]
        item.update(status="running", attempts=attempts, started_at=now.isoformat(),
                    target_date=str(day), retry_at=(now + timedelta(minutes=delay)).isoformat())
        close_data_sync._save(data_dir, state)
        emit("prediction_data", 10, f"增量补{label} {day}: {len(missing)} 只, 第{attempts}次")
        rows, error = 0, None
        try:
            touched = True
            if dataset == "daily":
                rows = kline_sync.sync_and_persist_daily_batch(
                    sorted(missing), repo, capset,
                    start_date=datetime.combine(day, datetime.min.time()),
                    end_date=datetime.combine(day, datetime.max.time()),
                    on_chunk_done=lambda i, n: emit("prediction_data", 30, f"日K补差 {i}/{n}"),
                )
            elif dataset == "enriched":
                rows = run_date_pipeline(data_dir, day)
            else:
                rows = close_data_sync._fetch_missing(repo, capset, dataset, day, missing, emit)
        except Exception as exc:
            from app.services.pipeline_jobs import JobCancelledError
            if isinstance(exc, JobCancelledError):
                raise
            error = type(exc).__name__
        remaining = readiness(data_dir, day)["missing"][dataset]
        item.update(status="succeeded" if not remaining else "needs_attention" if attempts >= 5 else "retry",
                    complete=not remaining, missing_symbols=remaining, rows_written=rows,
                    checked_at=now.isoformat(), error=error)
        if not remaining:
            item["attempts"] = 0
        close_data_sync._save(data_dir, state)
        results.append({"dataset": label, **item})
        emit("prediction_data", 80, f"{label}: 仍缺{len(remaining)}只; {error or item['status']}")
    if touched:
        from app.api.data import invalidate_storage_cache
        repo.refresh_index_views()
        repo.refresh_cache()
        invalidate_storage_cache()
    retry_times = [r["retry_at"] for r in results if r.get("status") != "succeeded"]
    return {"status": "retry" if retry_times else "checked", "datasets": results,
            "next_retry_at": min(retry_times) if retry_times else None}


def status(data_dir, bot_service=None):
    root = _root(data_dir)
    return {"config": config(), "state": _read(root / "state.json"),
            "snapshot": _read(root / "latest.json"),
            "premarket": _read(root / "premarket.json"),
            "bot": bot_service.status() if bot_service else {"connected": False, "chats": []}}


def _notify(data_dir, snapshot, now, bot_service=None):
    from app.services import preferences
    if snapshot.get("as_of") != str(now.date()) or now.hour < 19:
        return {"status": "blocked", "reason": "预测日期不是当天盘后, 禁止发送旧预测"}
    latest = readiness(data_dir, now.date())
    if not latest["ready"] or latest["input_version"] != (snapshot.get("readiness") or {}).get("input_version"):
        return {"status": "blocked", "reason": "发送前数据未就绪或版本已改变, 等待重新预测"}
    if snapshot.get("no_candidates"):
        return {"status": "no_candidates", "reason": "本日无合格候选，已留档，不发送候选通知"}
    path = _root(data_dir) / "notifications.json"
    entries = _read(path)
    item = entries.get(snapshot["id"], {})
    # Ambiguous delivery (crash while sending) is not automatically replayed.
    if item.get("status") in {"sent", "sending", "unknown"}:
        return item
    if item.get("attempts", 0) >= 3 or (item.get("retry_at") and item["retry_at"] > now.isoformat()):
        return item
    cfg = config()
    if not cfg["notify"]:
        return {"status": "disabled"}
    if not cfg.get("bot_configured") or not cfg.get("bot_enabled"):
        return {"status": "unconfigured"}
    chat_id = cfg.get("chat_id")
    if not chat_id or cfg.get("chat_bot_id") != preferences.get_wecom_bot_id():
        return {"status": "missing_recipient"}
    if bot_service is None or not bot_service.status().get("connected"):
        return {"status": "not_connected"}
    lines = [f"数据截止: {snapshot['as_of']}; 版本: {snapshot['id'][:10]}",
             "下一交易日观察候选, 不是开盘买入指令。", "盘前复核事件, 盘中通过量价确认和风险过滤后才考虑执行。"]
    for row in snapshot.get("rows", [])[:5]:
        def pct(key, row=row):
            v = row.get(key)
            return f"{v * 100:+.2f}%" if isinstance(v, (int, float)) and math.isfinite(v) else "暂无"
        name = str(row.get("name") or "").replace("\n", " ")[:20]
        lines.append(f"- {name} {row['symbol']}: 1日 {pct('pred_ret_1d')}, 3日 {pct('pred_ret_3d')}, 5日 {pct('pred_ret_5d')}")
    lines.append("市场环境: " + str((snapshot.get("market_gate") or {}).get("gate_label") or "未确认"))
    lines.append("模型结果仅供研究; 未通过样本外验证不得用于实盘。")
    item = {"status": "sending", "attempts": item.get("attempts", 0) + 1, "attempted_at": now.isoformat(),
            "channel": "wecom_smart_bot", "recipient": chat_id}
    entries[snapshot["id"]] = item
    _write(path, entries)
    try:
        receipt = bot_service.send_markdown(chat_id, "## 盘后预测候选\n" + "\n".join(lines))
        result = receipt.get("status")
        item.update(status=result if result in {"sent", "failed", "not_connected", "unknown"} else "unknown")
        if item["status"] == "not_connected":
            item["attempts"] -= 1
        if isinstance(receipt.get("errcode"), int):
            item["errcode"] = receipt["errcode"]
    except Exception:
        item.update(status="unknown")
    item["retry_at"] = (now + timedelta(minutes=30)).isoformat()
    entries[snapshot["id"]] = item
    _write(path, entries)
    return item


def _premarket(data_dir, snapshot, now, repo):
    from app.services.global_risk import build_global_risk
    risk = build_global_risk(repo, force_refresh=True, include_news=True)
    # Date-only news cannot establish that an event was known before the open.
    events = []
    first_day = date.fromisoformat(snapshot["as_of"])
    for offset in range(min((now.date() - first_day).days + 1, 15)):
        day = first_day + timedelta(days=offset)
        frame = _frame(data_dir, "news", day)
        if frame.is_empty():
            continue
        time_col = next((c for c in ("published_at", "datetime", "pub_time", "publish_time") if c in frame.columns), None)
        if not time_col:
            continue
        for row in frame.to_dicts():
            try:
                t = datetime.fromisoformat(str(row[time_col]).replace("Z", "+00:00"))
                t = t.replace(tzinfo=now.tzinfo) if t.tzinfo is None else t.astimezone(now.tzinfo)
                if datetime.fromisoformat(snapshot["generated_at"]) < t <= now:
                    events.append({"title": str(row.get("title") or "")[:180], "published_at": t.isoformat(),
                                   "symbol": str(row.get("symbol") or row.get("ts_code") or row.get("symbols") or ""),
                                   "market": str(row.get("market") or row.get("market_scope") or "")})
            except (ValueError, TypeError):
                continue
    updated = str(risk.get("updated_at") or "")
    # A successful request with a stale fallback is not a completed morning review.
    fresh = bool(risk.get("data_success")) and not risk.get("cache_fallback") and updated[:10] == str(now.date())
    risks = [e for e in events if any(word in e["title"] for word in ("立案", "退市", "业绩预亏", "重大亏损", "违约", "停牌", "制裁"))]
    prior_days = sorted(p.name[5:] for p in (Path(data_dir) / "kline_daily").glob("date=????-??-??") if p.name[5:] < str(now.date()))
    snapshot_current = bool(prior_days) and prior_days[-1] == snapshot["as_of"]
    domestic = [e for e in events if e["market"].lower() in {"cn", "a", "a_share", "domestic", "china", "a股", "国内"}]
    blocked = (snapshot.get("no_candidates", False) or not fresh or not domestic or bool(risks) or not snapshot_current
               or bool((risk.get("recommendation_adjustment") or {}).get("block_new_position")))
    review = {"date": str(now.date()), "snapshot_id": snapshot["id"], "checked_at": now.isoformat(),
              "status": "observe" if blocked else "reviewed", "block_new_position": blocked,
              "reason": "隔夜数据不完整或风险偏高, 只观察" if blocked else "隔夜复核完成, 仍需盘中确认",
              "global_risk": risk, "events": events[:50], "risk_events": risks[:20],
              "snapshot_current": snapshot_current, "domestic_event_count": len(domestic)}
    _write(_root(data_dir) / "premarket.json", review)
    return review


def due(data_dir, now):
    """Cheap scheduling check; do not claim the execution slot during a retry cooldown."""
    if not config()["enabled"]:
        return False
    root = _root(data_dir)
    if now.hour >= 19:
        return _read(root / "state.json").get("next_check_at", "") <= now.isoformat()
    if 8 <= now.hour < 10:
        snapshot = _read(root / "latest.json")
        if not snapshot or snapshot.get("as_of", "") >= str(now.date()):
            return False
        previous = _read(root / "premarket.json")
        return (previous.get("date") != str(now.date()) or previous.get("snapshot_id") != snapshot.get("id")
                or (previous.get("status") == "observe"
                    and previous.get("checked_at", "") < (now - timedelta(minutes=30)).isoformat()))
    return False


def tick(repo, *, now=None, trading_day=None, on_progress=None, repair_data=None, bot_service=None):
    """Run under the pipeline execution slot; JSON lock also protects API callers."""
    now = now or cn_now()
    if not config()["enabled"] or trading_day is not True or not _LOCK.acquire(blocking=False):
        return {"status": "skipped"}
    root = _root(repo.store.data_dir)
    try:
        current = _read(root / "state.json")
        snapshot = _read(root / "latest.json")
        if 8 <= now.hour < 10 and snapshot and snapshot["as_of"] < str(now.date()):
            previous = _read(root / "premarket.json")
            retry = previous.get("status") == "observe" and previous.get("checked_at", "") < (now - timedelta(minutes=30)).isoformat()
            if previous.get("date") != str(now.date()) or previous.get("snapshot_id") != snapshot["id"] or retry:
                return _premarket(repo.store.data_dir, snapshot, now, repo)
            return previous
        if now.hour < 19:
            return {"status": "waiting_close"}
        if current.get("next_check_at", "") > now.isoformat():
            return current
        ready = readiness(repo.store.data_dir, now.date())
        current = {"status": "waiting_data", "checked_at": now.isoformat(), "readiness": ready,
                   "next_check_at": (now + timedelta(minutes=30)).isoformat()}
        if not ready["ready"] and repair_data is not None:
            _write(root / "state.json", current)
            current["data_retry"] = repair_data(ready)
            ready = readiness(repo.store.data_dir, now.date())
            current["readiness"] = ready
            if current["data_retry"].get("next_retry_at"):
                current["next_check_at"] = current["data_retry"]["next_retry_at"]
        if not ready["ready"]:
            _write(root / "state.json", current)
            return current
        # Shares the same close-data retry slot; a failed stock model must not
        # prevent an otherwise valid sector forecast from being frozen.
        from app.services.sector_prediction import refresh_sector_prediction_records
        try:
            current["sector_forecast"] = refresh_sector_prediction_records(repo.store.data_dir, days=8).get("generation")
        except Exception as exc:
            current["sector_forecast"] = {"status": "error", "reason": str(exc)}
            logger.exception("Sector forecast failed; stock workflow remains independent")
        if snapshot.get("as_of") != str(now.date()) or (snapshot.get("readiness") or {}).get("input_version") != ready["input_version"]:
            from app.services.stockformer import multifactor_select
            payload = multifactor_select(repo.store.data_dir, as_of=now.date(), limit=100)
            if payload.get("error") or payload.get("as_of") != str(now.date()):
                current.update(status="prediction_failed", error="模型不可用或预测日期不匹配")
                _write(root / "state.json", current)
                return current
            allowed = set(ready["eligible_symbols"])
            rows = [r for r in payload.get("rows", []) if str(r.get("symbol", "")).split(".")[0] in allowed][:30]
            no_candidates = (
                not payload.get("rows") and payload.get("total") == 0
                and payload.get("signal_status") == "no_signal"
                and isinstance(payload.get("universe_total"), int) and payload["universe_total"] > 0
            )
            if not rows and not no_candidates:
                current.update(status="prediction_failed", error="没有通过数据校验的候选")
                _write(root / "state.json", current)
                return current
            from app.services.market_gate import build_market_gate
            market_gate = build_market_gate(repo.store.data_dir, repo=repo, as_of=now.date(), use_overview=False)
            verified = readiness(repo.store.data_dir, now.date())
            if not verified["ready"] or verified["input_version"] != ready["input_version"]:
                current.update(status="waiting_data", readiness=verified,
                               error="预测期间数据发生变化, 丢弃本次结果并等待重算")
                _write(root / "state.json", current)
                return current
            snapshot = {**payload, "rows": rows, "generated_at": now.isoformat(), "stage": "after_close",
                        "purpose": "下一交易日观察候选", "readiness": ready, "market_gate": market_gate}
            if no_candidates:
                snapshot.update(no_candidates=True, empty_reason="no_validated_horizon_weights")
            snapshot["id"] = hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
            _write(root / "snapshots" / f"{snapshot['as_of']}-{snapshot['id']}.json", snapshot)
            _write(root / "latest.json", snapshot)
        current.update(status="predicted", snapshot_id=snapshot["id"], notification=_notify(repo.store.data_dir, snapshot, now, bot_service))
        _write(root / "state.json", current)
        if on_progress:
            on_progress("prediction", 100, "盘后候选已留档, 通知状态: " + current["notification"]["status"])
        return current
    finally:
        _LOCK.release()


def apply_execution_gate(payload, workflow, now=None):
    """Never turn a frozen research candidate into an unconditional buy."""
    now = now or cn_now()
    snapshot = workflow.get("snapshot") or {}
    review = workflow.get("premarket") or {}
    is_session = (9, 30) <= (now.hour, now.minute) <= (11, 30) or (13, 0) <= (now.hour, now.minute) < (15, 0)
    allow = (bool(snapshot) and snapshot.get("as_of", "") < str(now.date())
             and not snapshot.get("no_candidates", False)
             and review.get("date") == str(now.date()) and review.get("snapshot_id") == snapshot.get("id")
             and review.get("block_new_position") is False and is_session and now.weekday() < 5)
    payload["workflow"] = {"state": workflow.get("state"), "snapshot_id": snapshot.get("id"),
                           "as_of": snapshot.get("as_of"), "premarket": review, "execution_allowed": allow}
    if not allow:
        for row in payload.get("rows", []):
            reason = "盘后候选/盘前复核尚未通过, 禁止新开仓"
            row.update(decision="回避" if row.get("decision") == "回避" else "观察", decision_reason=reason, action=reason,
                       buy_confirmation_status="blocked", buy_confirmation_label="等待盘前复核及交易时段",
                       position_multiplier=0.0, combined_position_multiplier=0.0)
            row["reasons"] = [r for r in row.get("reasons", []) if r != "买入确认通过"]
            row["risk_warnings"] = list(dict.fromkeys([*row.get("risk_warnings", []), reason]))
            row["trade_plan"] = {**(row.get("trade_plan") or {}), "entry": reason,
                                 "position": "新开仓仓位为0", "confirmation": "等待盘前复核及交易时段"}
    return payload
