"""Independent Transformer experiment for StockFormer.

This module is deliberately isolated from production prediction.  It requires
the optional ``transformer`` extra and writes its own model and metrics files.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from app.services import stockformer as sf
from app.services import stockformer_training as training

EXPERIMENT_KIND = "stockformer_transformer_experiment_v1"
WINDOW = 20
EXPERIMENT_FEATURES = tuple(sf.COMPACT_FEATURES if hasattr(sf, "COMPACT_FEATURES") else (
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ma20_gap", "vol_ratio_5d",
    "intraday_strength", "amplitude", "risk_20d", "gap_open",
    "amount_ratio_5d", "price_position_20d", "price_range_10d",
    "minute_session_coverage", "minute_intraday_return", "minute_early_return",
    "minute_open_strength", "minute_intraday_volatility",
    "minute_early_amount_ratio", "minute_late_amount_ratio",
    "minute_volume_concentration", "minute_vwap_gap", "minute_close_position",
))


def _json_safe(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _build_sequences(frame, feature_columns, horizon=5, window=WINDOW):
    frame = frame.sort(["symbol", "date"])
    xs, ys, dates = [], [], []
    for symbol in frame["symbol"].unique().to_list():
        part = frame.filter(frame["symbol"] == symbol)
        if part.height <= window:
            continue
        values = part.select(feature_columns).to_numpy().astype(np.float32)
        targets = part[f"target_{horizon}d"].to_numpy().astype(np.float32)
        part_dates = part["date"].to_list()
        for idx in range(window, len(values)):
            if not np.isfinite(targets[idx]) or not np.isfinite(values[idx-window:idx]).all():
                continue
            xs.append(values[idx-window:idx])
            ys.append(targets[idx])
            dates.append(part_dates[idx])
    if not xs:
        return np.empty((0, window, len(feature_columns)), dtype=np.float32), np.empty((0,), dtype=np.float32), []
    return np.asarray(xs), np.asarray(ys), dates


def run_experiment(data_dir: Path, *, start: date | None = None, end: date | None = None,
                   max_symbols: int = 200, epochs: int = 8) -> dict:
    try:
        import torch
        from torch import nn
    except ImportError:
        return {"error": "Transformer 实验需要 PyTorch，请使用 BACKEND_EXTRAS=transformer 重建容器"}

    t0 = time.perf_counter()
    min_date, max_date = sf._date_range_from_data(data_dir)
    if min_date is None or max_date is None:
        return {"error": "本地日线特征为空，请先同步并重算日线数据"}
    end = end or max_date
    start = start or max(min_date, end - timedelta(days=365 * 2))
    panel = sf._load_panel(data_dir, start - timedelta(days=120), end,
                           max_symbols=max_symbols, universe_as_of=start, repair_adjusted=False)
    if panel.is_empty():
        return {"error": "实验区间没有可用行情数据"}
    features = training.execution_targets(sf._feature_frame(panel, include_targets=False), panel)
    features = features.filter((features["date"] >= start) & (features["date"] <= end))
    columns = [c for c in EXPERIMENT_FEATURES if c in features.columns]
    if len(columns) < 5:
        return {"error": "Transformer 实验可用特征不足"}
    x, y, sample_dates = _build_sequences(features, columns)
    if len(y) < 100:
        return {"error": f"Transformer 实验样本不足: {len(y)}，至少需要 100 条序列样本"}
    split = max(1, min(len(y) - 20, int(len(y) * 0.7)))
    train_x, test_x = x[:split], x[split:]
    train_y, test_y = y[:split], y[split:]
    torch.manual_seed(42)
    device = torch.device("cpu")

    class SequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            width = 32
            self.input = nn.Linear(len(columns), width)
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=64,
                                               dropout=0.10, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

        def forward(self, value):
            encoded = self.encoder(self.input(value))
            return self.head(encoded[:, -1]).squeeze(-1)

    model = SequenceModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss()
    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(train_y).to(device)
    model.train()
    for _ in range(max(1, min(epochs, 30))):
        optimizer.zero_grad()
        loss = loss_fn(model(tx), ty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(test_x).to(device)).cpu().numpy()
    actual = test_y.astype(float)
    valid = np.isfinite(pred) & np.isfinite(actual)
    direction = float(np.mean((pred[valid] > 0) == (actual[valid] > 0))) if valid.any() else None
    corr = float(np.corrcoef(pred[valid], actual[valid])[0, 1]) if valid.sum() >= 3 else None
    experiment_dir = data_dir / "stockformer" / "transformer_experiments"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = f"tf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_file = experiment_dir / f"{experiment_id}.pt"
    metrics_file = experiment_dir / f"{experiment_id}.json"
    torch.save({"state_dict": model.state_dict(), "feature_columns": columns,
                "window": WINDOW, "horizon": 5, "kind": EXPERIMENT_KIND}, model_file)
    result = _json_safe({
        "experiment_id": experiment_id, "kind": EXPERIMENT_KIND,
        "model_file": str(model_file), "metrics_file": str(metrics_file),
        "feature_columns": columns, "window": WINDOW, "horizon": 5,
        "train_samples": len(train_y), "oos_samples": len(test_y),
        "oos_start": sample_dates[split], "oos_end": sample_dates[-1],
        "oos_direction_accuracy": direction, "oos_ic": corr,
        "baseline_direction_accuracy": float(np.mean((actual > 0) == (np.mean(train_y) > 0))),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "status": "research_only",
    })
    metrics_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def latest_experiment(data_dir: Path) -> dict:
    root = data_dir / "stockformer" / "transformer_experiments"
    files = sorted(root.glob("tf_*.json"), reverse=True) if root.exists() else []
    if not files:
        return {"experiment": None}
    try:
        return {"experiment": json.loads(files[0].read_text(encoding="utf-8"))}
    except Exception as exc:
        return {"error": f"Transformer 实验结果读取失败: {type(exc).__name__}: {exc}"}
