"""Research-only adapter using the fitted pair's exact reference and transform."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import polars as pl

from app.services import market_gate
from app.services import stockformer as sf
from app.services import stockformer_necessary_oof_research as necessary
from app.services.stockformer_minute_representation import rank_minutes

KEYS = ["symbol", "date"]


@contextmanager
def pinned_market(data_dir, frame):
    """Read the contemporaneous published facts inside one offline process only."""
    require_market_days(frame, frame["date"].unique().to_list())
    original = market_gate.market_gate_frame
    expected = Path(data_dir).resolve()

    def selected(target):
        return frame.clone() if Path(target).resolve() == expected else original(target)

    market_gate.market_gate_frame = selected
    try:
        yield
    finally:
        market_gate.market_gate_frame = original


def assert_post_freeze(days, frozen_at):
    # The label cut is not the model's actual publication time.
    first = datetime.fromisoformat(frozen_at).date()
    if not days or len(days) != len(set(days)) or any(d <= first for d in days):
        raise ValueError("Scoring must use distinct sessions after the actual freeze")


def require_market_days(gate, days):
    required = [
        "date",
        "market_gate",
        "market_regime_score",
        "market_above_ma20_pct",
        "market_strong_down_pct",
        "market_position_multiplier",
        "market_buy_threshold_add",
    ]
    if any(c not in gate.columns for c in required):
        raise ValueError("Missing explicit market inputs; no neutral fallback")
    selected = gate.filter(pl.col("date").is_in(days))
    if selected.height != len(days) or set(selected["date"]) != set(days):
        raise ValueError("Missing or duplicate exact-date market inputs")
    if selected.filter(
        ~pl.all_horizontal([pl.col(c).is_finite().fill_null(False) for c in required[2:]])
    ).height:
        raise ValueError("Nonfinite exact-date market inputs")
    if selected.filter(
        ~pl.col("market_gate").is_in(["open", "strict", "observe", "closed"])
    ).height:
        raise ValueError("Unknown market gate")
    return selected.sort("date")


def reference(data_dir, features, raw, protocol):
    if any(c.startswith(("target_", "label_")) for c in features.columns):
        raise ValueError("Forward features must be label-free")
    if features.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise ValueError("Duplicate forward feature keys")
    frame, audit = necessary.reference_pool(data_dir, features, raw, protocol["execution"])
    frame = rank_minutes(frame, protocol["readiness_columns"])
    return frame, audit


def complete_only(features, columns):
    prepared, minute = sf.prepare_training_features(features, "minute_enhanced")
    _, _, complete = sf._prepare_xy(
        prepared.with_columns(pl.lit(0.0).alias("_ready_mask")), "_ready_mask", columns
    )
    return complete, minute


def annotated_rejections(universe, day, stages, predictions, signals):
    """Keep every frozen-universe member, including stocks with no input row."""
    rows = pl.DataFrame({"symbol": universe}).with_columns(pl.lit(day).alias("date"))
    for name, members in stages.items():
        rows = rows.with_columns(pl.col("symbol").is_in(sorted(members)).alias(name))
    rows = rows.join(
        predictions.select(*KEYS, "pred_ret_raw_5d", "pred_ret_5d"), on=KEYS, how="left"
    )
    signal_cols = [
        "_necessary_entry_pool",
        "_prediction_pass",
        "_net_margin_pass",
        "_entry_rules_pass",
        "_market_gate_pass",
        "entry_score",
        "stockformer_score",
        "net_margin_score",
        "market_gate",
        "market_regime_score",
    ]
    rows = rows.join(signals.select(*KEYS, *signal_cols), on=KEYS, how="left")
    boolean = [c for c in signal_cols if c.startswith("_")]
    rows = rows.with_columns(pl.col(boolean).fill_null(False))
    names = list(stages) + boolean
    output = []
    for item in rows.sort("symbol").to_dicts():
        reasons = [name for name in names if not item[name]]
        item["rejection_reasons"] = reasons
        item["first_rejection"] = reasons[0] if reasons else None
        item["eligible"] = not reasons
        output.append(item)
    return pl.DataFrame(output, schema_overrides={"date": pl.Date})
