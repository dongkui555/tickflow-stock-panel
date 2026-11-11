"""Daily, timestamped frozen-model observations and isolated paper execution.

Only a same-session evening snapshot may create next-open paper orders. Late
imports never become backdated forward evidence. No training or activation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from app.market_time import CN_TZ, cn_now
from app.services import stockformer as sf
from app.services import stockformer_portfolio as portfolio
from app.services.fs_utils import atomic_write_parquet, atomic_write_text
from app.services.prediction_workflow import readiness
from app.services.stockformer_paper import _daily_observations
from app.strategy import paper

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
_EXECUTION = {"initial_cash": 10000.0, "max_positions": 3, "holding_sessions": 5,
              "commission_pct": 0.0003, "stamp_tax_pct": 0.0005, "slippage_bps": 5.0,
              "queue_limit_orders": False, "minimum_commission": paper.MIN_COMMISSION}


def _root(data_dir: Path) -> Path:
    return Path(data_dir) / "stockformer" / "daily_shadow"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(sf._json_safe(value), ensure_ascii=False,
                                      allow_nan=False, indent=2))


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _frozen_assets(data_dir: Path, model_id: str) -> dict:
    meta = sf._model_meta(data_dir, model_id)
    if not meta or not portfolio.load_frozen_signal_policy(data_dir, meta):
        raise ValueError("研究模型或冻结信号规则不存在")
    model_path = sf._resolve_artifact_path(data_dir, meta.get("model_file"))
    payload = sf._load_model_payload(model_path)
    paths = {model_path}
    for model in payload.get("models", {}).values():
        for key in ("model_file", "xgboost_file", "lightgbm_file"):
            if model.get(key):
                paths.add(sf._resolve_artifact_path(data_dir, model[key]))
    from app.services import market_gate, regime_builder
    modules = (sf, portfolio, paper, market_gate, regime_builder)
    code = {Path(module.__file__).name: _digest(Path(module.__file__)) for module in modules}
    code[Path(__file__).name] = _digest(Path(__file__))
    return {"metadata": _hash(meta), "policy": portfolio.load_frozen_signal_policy(data_dir, meta),
            "models": {str(path.relative_to(data_dir)): _digest(path) for path in sorted(paths)},
            "code": code}


def configure(data_dir: Path, model_id: str, *, start_date: date) -> dict:
    """One-time installation; changing the frozen protocol requires a new run."""
    path = _root(data_dir) / "config.json"
    if path.exists():
        raise ValueError("daily shadow already configured; preserve the existing run")
    assets = _frozen_assets(data_dir, model_id)
    protocol = {"model_id": model_id, "start_date": str(start_date), "assets": assets,
                "execution": _EXECUTION, "cutoff": "same_session_19:00_to_23:59_Asia/Shanghai",
                "horizon": 5, "version": 1}
    cfg = {"enabled": True, "protocol": protocol, "protocol_id": _hash(protocol),
           "created_at": cn_now().isoformat(), "account_id": "sf_shadow_" + _hash(protocol)[:16]}
    _write(path, cfg)
    return cfg


def _input_version(data_dir: Path, day: date) -> str:
    """Bounded input manifest; exact features and scores are also saved per day."""
    start = day - timedelta(days=160)
    signature = []
    for table in ("kline_daily", "kline_daily_enriched", "kline_minute", "moneyflow",
                  "auction_open", "kline_index_daily", "regime_history", "instruments",
                  "instruments_ext", "news", "adj_factor"):
        root = Path(data_dir) / table
        for path in sorted(root.rglob("*.parquet")):
            partition = next((part[5:] for part in path.parts if part.startswith("date=")), None)
            if partition and not str(start) <= partition <= str(day):
                continue
            stat = path.stat()
            signature.append((str(path.relative_to(data_dir)), stat.st_size, stat.st_mtime_ns))
    return _hash(signature)


def _compute(data_dir: Path, cfg: dict, day: date) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    model_id = cfg["protocol"]["model_id"]
    prediction = sf.predict(data_dir, model_id=model_id, as_of=day, limit=0, write_artifact=False)
    if prediction.get("error") or prediction.get("as_of") != str(day) or not prediction.get("rows"):
        raise ValueError(prediction.get("error") or "当天无完整可预测截面")
    meta = sf._model_meta(data_dir, model_id)
    payload = sf._load_model_payload(sf._resolve_artifact_path(data_dir, meta["model_file"]))
    features, actual, _ = sf._load_prediction_base(
        data_dir, day, required_feature_columns=payload["feature_columns"],
        strict_minute=payload.get("training_profile") == "minute_enhanced",
        symbols=[row["symbol"] for row in prediction["rows"]],
    )
    if actual != day or features.is_empty():
        raise ValueError("闸门特征日期不一致")
    factor = pl.DataFrame(prediction["rows"]).with_columns(pl.col("date").cast(pl.Date))
    policy = portfolio.load_frozen_signal_policy(data_dir, meta)
    signal = portfolio._build_gated_signal_frame(
        data_dir, factor, features, meta, horizon=5, signal_policy=policy,
        min_net_edge=float(policy["prediction_rule"]["min_net_edge"]),
    )
    return prediction, signal, features


def _account(data_dir: Path, cfg: dict) -> dict:
    expected = cfg["protocol"]["execution"]
    account_id = cfg["account_id"]
    acc = paper.get_account(data_dir, account_id)
    if acc is None:
        acc = paper.create_account(data_dir, expected["initial_cash"], account_id=account_id,
                                   name="StockFormer 五日每日影子（研究）",
                                   **{k: expected[k] for k in ("commission_pct", "stamp_tax_pct",
                                                              "slippage_bps", "queue_limit_orders")})
        acc["managed_by"] = "stockformer_daily_shadow"
        paper.save_account(data_dir, acc, account_id)
    keys = ("initial_cash", "commission_pct", "stamp_tax_pct", "slippage_bps", "queue_limit_orders")
    if any(acc.get(k) != expected[k] for k in keys) or acc.get("managed_by") != "stockformer_daily_shadow":
        raise ValueError("影子账户设置已改变，暂停自动模拟成交")
    if any(not str(o.get("source", "")).startswith("shadow:") for o in paper.load_orders(data_dir, account_id)):
        raise ValueError("影子账户存在非研究订单，暂停自动模拟成交")
    return acc


def _settle(data_dir: Path, cfg: dict, day: date) -> dict:
    """One settlement per actual session, independently of dataset pipeline failures."""
    folder = _root(data_dir) / "settlements"
    path = folder / f"{day}.json"
    existing = _read(path)
    if existing:
        if existing["status"] != "settled":
            raise ValueError("影子结算中断，需核对成交台账后恢复；禁止重复顺延")
        return existing
    # An ambiguous crash is fail-closed: a later run must not replay postponements.
    _write(path, {"status": "settling", "date": str(day), "started_at": cn_now().isoformat()})
    result = paper.settle_day(data_dir, str(day), account_id=cfg["account_id"])
    result.update(status="settled", date=str(day), completed_at=cn_now().isoformat())
    _write(path, result)
    return result


def _orders(data_dir: Path, cfg: dict, snapshot: dict, now: datetime) -> dict:
    """Recover safely after snapshot publication; existing source IDs are not resubmitted."""
    folder = _root(data_dir)
    path = folder / "executions" / f"{snapshot['as_of']}.json"
    existing = _read(path)
    if existing:
        return existing
    aid = cfg["account_id"]
    execution = cfg["protocol"]["execution"]
    results = []
    with paper.PAPER_LOCK:
        acc = _account(data_dir, cfg)
        existing_orders = {o["source"]: o for o in paper.load_orders(data_dir, aid)}
        timely = snapshot["eligible_for_forward"] and snapshot["as_of"] == str(now.date())
        if not timely:
            result = {"status": "late_no_orders", "orders": [], "recorded_at": now.isoformat()}
            _write(path, result)
            return result
        positions = paper.load_positions(data_dir, aid)
        results.extend(_queue_exits(data_dir, cfg, date.fromisoformat(snapshot["as_of"])))
        pending = [o for o in existing_orders.values() if o["status"] == "pending" and o["side"] == "buy"]
        occupied = {s for s, p in positions.items() if p["qty"] > 0} | {o["symbol"] for o in pending}
        reserved = sum(o["qty"] * float(o.get("ref_price") or 0) * (1 + execution["slippage_bps"] / 10000)
                       + paper.buy_fee(o["qty"], float(o.get("ref_price") or 0), execution["commission_pct"]) for o in pending)
        cash = float(acc["cash"]) - reserved
        budget = min(cash, execution["initial_cash"] / execution["max_positions"])
        for row in snapshot["selected"]:
            symbol = row["symbol"]
            source = f"shadow:{snapshot['id']}:buy:{symbol}"
            if source in existing_orders:
                continue
            reason = None
            if symbol in occupied:
                reason = "already_held_or_pending"
            elif len(occupied) >= execution["max_positions"]:
                reason = "position_limit"
            # Read raw close for sizing, never the qfq feature close.
            bar = paper.read_daily_bar(data_dir, symbol, "stock", snapshot["as_of"])
            price = bar["close"] if bar else 0.0
            qty = paper.qty_from_amount(max(0., budget - execution["minimum_commission"]),
                                       price * (1 + execution["slippage_bps"] / 10000)) if price > 0 else 0
            if qty <= 0:
                reason = reason or "missing_raw_price_or_one_lot_budget"
            order = None
            if reason is None:
                order, reason = paper.create_order(data_dir, symbol, "buy", account_id=aid, qty=qty,
                                                  ref_price=price, order_type="next_open", source=source)
            if order:
                occupied.add(symbol)
                cash -= qty * price * (1 + execution["slippage_bps"] / 10000) + paper.buy_fee(qty, price, execution["commission_pct"])
                budget = min(cash, execution["initial_cash"] / execution["max_positions"])
            results.append({"symbol": symbol, "side": "buy", "order_id": order["id"] if order else None, "reason": reason})
    result = {"status": "recorded", "orders": results, "recorded_at": now.isoformat()}
    _write(path, result)
    return result


def _queue_exits(data_dir: Path, cfg: dict, day: date) -> list[dict]:
    """Five held sessions then next-open exit, even if entry features are late."""
    if cn_now().date() != day:
        return []
    aid = cfg["account_id"]
    results = []
    with paper.PAPER_LOCK:
        orders = paper.load_orders(data_dir, aid)
        sessions = sorted(p.name[5:] for p in (Path(data_dir) / "kline_daily").glob("date=????-??-??")
                          if p.name[5:] <= str(day))
        for symbol, pos in paper.load_positions(data_dir, aid).items():
            lots = pos.get("lots", [])
            if not lots or sum(d >= min(lot["date"] for lot in lots) for d in sessions) < cfg["protocol"]["execution"]["holding_sessions"]:
                continue
            if any(o["symbol"] == symbol and o["side"] == "sell" and o["status"] == "pending" for o in orders):
                continue
            source = f"shadow:{_hash([cfg['protocol_id'], str(day)])}:sell:{symbol}"
            if any(o["source"] == source for o in orders):
                continue
            order, error = paper.create_order(data_dir, symbol, "sell", account_id=aid,
                                             qty=int(pos["qty"]), order_type="next_open", source=source)
            results.append({"symbol": symbol, "side": "sell", "order_id": order["id"] if order else None, "reason": error})
    return results


def due(data_dir: Path, now: datetime) -> bool:
    cfg = _read(_root(data_dir) / "config.json")
    if not cfg.get("enabled") or str(now.date()) < cfg["protocol"]["start_date"] or now.hour < 19:
        return False
    record = _root(data_dir) / "days" / str(now.date()) / "snapshot.json"
    if record.exists():
        return not (_root(data_dir) / "executions" / f"{now.date()}.json").exists()
    state = _read(_root(data_dir) / "state.json")
    return state.get("next_check_at", "") <= now.isoformat()


def tick(data_dir: Path, *, now: datetime, trading_day: bool | None, on_progress=None) -> dict:
    if not _LOCK.acquire(blocking=False):
        return {"status": "busy"}
    cfg = _read(_root(data_dir) / "config.json")
    emit = on_progress or (lambda *a: None)

    def state(kind, **fields):
        result = {"status": kind, "checked_at": now.isoformat(), "as_of": str(now.date()),
                  "next_check_at": (now + timedelta(minutes=15)).isoformat(), **fields}
        _write(_root(data_dir) / "state.json", result)
        _write(_root(data_dir) / "sessions" / f"{now.date()}.json", result)
        return result

    try:
        now = now.astimezone(CN_TZ)
        if not cfg.get("enabled"):
            return {"status": "disabled"}
        if str(now.date()) < cfg["protocol"]["start_date"]:
            return state("before_start")
        if trading_day is not True:
            return state("non_trading_day" if trading_day is False else "calendar_unknown")
        if now.hour < 19:
            return state("waiting_close", reason="每日19:00后等候完整收盘特征")
        if _hash(cfg["protocol"]) != cfg["protocol_id"] or _frozen_assets(data_dir, cfg["protocol"]["model_id"]) != cfg["protocol"]["assets"]:
            return state("blocked_protocol_changed", reason="模型、校准、规则或执行代码与冻结指纹不同")
        path = _root(data_dir) / "days" / str(now.date()) / "snapshot.json"
        saved = _read(path)
        if saved:
            _orders(data_dir, cfg, saved, now)
            return saved
        emit("shadow_features", 5, "影子记录：核对收盘数据与冻结版本")
        ready = readiness(data_dir, now.date())
        settlement = None
        if ready.get("ready") or ready.get("checks", {}).get("日K"):
            _account(data_dir, cfg)
            settlement = _settle(data_dir, cfg, now.date())
            _queue_exits(data_dir, cfg, now.date())
        if not ready["ready"]:
            return state("waiting_data", readiness=ready, reason=ready["reason"], settlement=settlement)
        before = _input_version(data_dir, now.date())
        emit("shadow_predict", 20, "影子记录：冻结模型预测与分数校准")
        prediction, signal, features = _compute(data_dir, cfg, now.date())
        emit("shadow_gates", 75, "影子记录：预测、净优势、入场和市场闸门逐层统计")
        if before != _input_version(data_dir, now.date()):
            return state("inputs_changed", reason="计算期间数据版本变化，丢弃本次预测后重试")
        if _frozen_assets(data_dir, cfg["protocol"]["model_id"]) != cfg["protocol"]["assets"]:
            return state("blocked_protocol_changed")
        finished = cn_now()
        timely = finished.date() == now.date()
        observations, _ = _daily_observations(signal, available_dates=[now.date()], pred_col="pred_ret_5d",
                                             max_positions=cfg["protocol"]["execution"]["max_positions"], forecast_ceiling=None)
        observation = observations[0]
        observation["prediction_net_entry_pass"] = int(signal.filter(
            pl.col("_prediction_pass") & pl.col("_net_margin_pass") & pl.col("_entry_rules_pass")
        ).height) if not signal.is_empty() else 0
        enabled = prediction.get("signal_status") != "no_signal" and float(prediction.get("signal_weights", {}).get("5d", 0)) > 0
        selected = signal.filter(pl.col("_forecast_pass") & pl.col("_entry_rules_pass") & pl.col("_market_gate_pass")) if not signal.is_empty() else signal
        selected = selected.sort(["net_margin_score", "symbol"], descending=[True, False]).head(
            cfg["protocol"]["execution"]["max_positions"]) if not selected.is_empty() else selected
        if not enabled or not timely:
            selected = selected.clear()
            observation.update(selected_count=0, no_trade_reason="late_snapshot" if not timely else "model_no_signal")
        snapshot = {"status": "recorded" if timely else "late_record", "id": _hash([cfg["protocol_id"], str(now.date())]),
                    "model_id": cfg["protocol"]["model_id"], "account_id": cfg["account_id"],
                    "as_of": str(now.date()), "started_at": now.isoformat(), "generated_at": finished.isoformat(),
                    "eligible_for_forward": timely, "entry_rule": "next_actual_session_open_after_generated_at",
                    "protocol_id": cfg["protocol_id"], "input_version": before, "readiness": ready,
                    "signal_status": prediction.get("signal_status"), "signal_weights": prediction.get("signal_weights"),
                    "minute_coverage": prediction.get("minute_coverage"), "observation": observation,
                    "prediction_rows": prediction.get("rows", []), "selected": selected.to_dicts(), "settlement": settlement}
        if not signal.is_empty():
            reasons = []
            for row in signal.to_dicts():
                reasons.append([name for name, passed in (
                    ("model_no_signal", enabled), ("prediction_gate", row.get("_prediction_pass")),
                    ("net_margin", row.get("_net_margin_pass")), ("entry_rules", row.get("_entry_rules_pass")),
                    ("market_gate", row.get("_market_gate_pass")), ("late_snapshot", timely)) if not passed])
            signal = signal.with_columns(pl.Series("filter_reasons", reasons, dtype=pl.List(pl.String)))
        gated_symbols = set(signal["symbol"].to_list()) if not signal.is_empty() else set()
        snapshot["excluded_before_gates"] = [row["symbol"] for row in prediction.get("rows", [])
                                             if row["symbol"] not in gated_symbols]
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(features, path.parent / "features.parquet")
        atomic_write_parquet(signal, path.parent / "signals.parquet")
        snapshot["feature_sha256"] = _digest(path.parent / "features.parquet")
        snapshot["signal_sha256"] = _digest(path.parent / "signals.parquet")
        finished = cn_now()  # Publication timestamp includes persisting the exact inputs.
        snapshot["generated_at"] = finished.isoformat()
        if finished.date() != now.date():
            snapshot.update(status="late_record", eligible_for_forward=False, selected=[])
            observation.update(selected_count=0, no_trade_reason="late_snapshot")
        _write(path, snapshot)  # Publication precedes recoverable order creation.
        _write(path.parent / "summary.json", {k: snapshot[k] for k in (
            "as_of", "generated_at", "status", "observation", "eligible_for_forward")})
        _orders(data_dir, cfg, snapshot, finished)
        state(snapshot["status"], observation=observation, generated_at=finished.isoformat())
        emit("done", 100, f"影子记录完成：{observation['total_rows']}条预测，{observation['selected_count']}个候选")
        return _read(path)
    except Exception as exc:  # Durable visible failure; never alter a frozen snapshot.
        logger.exception("daily research shadow failed")
        return state("error", reason=f"{type(exc).__name__}: {exc}")
    finally:
        _LOCK.release()


def status(data_dir: Path) -> dict:
    root = _root(data_dir)
    cfg = _read(root / "config.json")
    records = [_read(p) for p in sorted((root / "days").glob("*/summary.json"))]
    # Compact summaries keep polling independent of the number of prediction rows.
    daily = {row["as_of"]: row for row in records}
    for path in sorted((root / "sessions").glob("*.json")):
        row = _read(path)
        daily.setdefault(row["as_of"], row)
    if cfg:
        for path in (Path(data_dir) / "kline_daily").glob("date=????-??-??"):
            day = path.name[5:]
            if cfg["protocol"]["start_date"] <= day < str(cn_now().date()) and day not in {r["as_of"] for r in records}:
                daily[day] = {**daily.get(day, {}), "as_of": day, "status": "missed_session",
                              "reason": "当日截止前未生成冻结快照；不补记过去开盘成交"}
    fills = paper.load_fills(data_dir, cfg["account_id"]) if cfg else []
    return {"enabled": cfg.get("enabled", False), "model_id": cfg.get("protocol", {}).get("model_id"),
            "start_date": cfg.get("protocol", {}).get("start_date"), "account_id": cfg.get("account_id"),
            "state": _read(root / "state.json"), "recorded_days": len(records), "days": [daily[d] for d in sorted(daily)],
            "fills": len([f for f in fills if f.get("side") in ("buy", "sell")]),
            "protocol_id": cfg.get("protocol_id"), "activation_allowed": False}
