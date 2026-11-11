"""Independent daily shadow for the immutable minute-rank research pair.

The original observer, quantities, corporate ledger and paper matcher run in
one dedicated worker. Bindings never enter the shared API or older workers.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import time
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import polars as pl

from app.market_time import cn_now
from app.services import market_gate
from app.services import stockformer_daily_shadow as daily
from app.services import stockformer_frozen_pair as pair
from app.services import stockformer_gross_rank_research as gross
from app.services import stockformer_minute_representation as representation
from app.services import stockformer_necessary_oof_research as necessary
from app.services import stockformer_pair_shadow as observer
from app.services import stockformer_rank_inputs as rank_inputs
from app.services import stockformer_repaired_shadow as repaired
from app.services.stockformer_corporate_ledger import POLICY as LEDGER_POLICY
from app.services.stockformer_execution_quantities import POLICY as QUANTITY_POLICY

PAIR_ID = "2025d8b58d6c601f7e52b2e98d039ae138ada354fd286859e1c79b2fa06b1a83"
_PREPARE = repaired.prepare
logger = logging.getLogger(__name__)


def scope(data_dir):
    return Path(data_dir) / "stockformer/rank_pair_shadow"


def view(data_dir):
    return scope(data_dir) / "view"


def verify_bundle(frozen):
    frozen = Path(frozen)
    model = pair.verify_manifest(frozen / "latest-pair")
    bundle = daily._read(frozen / "bundle.json")
    if bundle.get("pair_id") != model["pair_id"] or model["pair_id"] != PAIR_ID:
        raise ValueError("Unexpected frozen minute-rank pair")
    for name, digest in bundle["files"].items():
        path = (frozen / name).resolve()
        if not path.is_relative_to(frozen.resolve()) or pair.digest(path) != digest:
            raise ValueError("Rank bundle file changed: " + name)
    parent = daily._read(frozen / "parent-protocol.json")
    fit = daily._read(frozen / "fit-design.json")
    if (
        pair.digest(frozen / "fit-design.json") != model["provenance"]["context"]["protocol_sha256"]
        or pair.digest(frozen / "parent-protocol.json") != fit["original_protocol_sha256"]
        or pair.digest(representation.__file__)
        != model["provenance"]["context"]["transform_sha256"]
    ):
        raise ValueError("Rank training protocol or transform changed")
    for name, digest in parent["source_sha256"].items():
        if pair.digest(importlib.import_module(name).__file__) != digest:
            raise ValueError("Rank fitting/inference source changed: " + name)
    if (
        parent["execution"] != daily._EXECUTION
        or parent["feature_columns"]["minute_rank77"] != model["feature_columns"]
        or parent["min_net_edge"] != parent["signal_policy"]["prediction_rule"]["min_net_edge"]
    ):
        raise ValueError("Rank execution, score or feature contract changed")
    for item in daily._read(frozen / "seed.json")["files"].values():
        if pair.digest(item["path"]) != item["sha256"]:
            raise ValueError("Rank matched-input seed changed")
    universe = daily._read(frozen / "universe.json")["symbols"]
    if universe != sorted(set(universe)) or len(universe) != 1000:
        raise ValueError("Unexpected fixed rank universe")
    return model


def assets(frozen):
    verify_bundle(frozen)
    names = [
        "app.services.stockformer_rank_shadow",
        "app.services.stockformer_rank_inputs",
        "app.services.stockformer_minute_representation",
        "app.services.stockformer_pair_shadow",
        "app.services.stockformer_daily_shadow",
        "app.services.stockformer_repaired_shadow",
        "app.services.stockformer_repaired_inputs",
        "app.services.stockformer_execution_quantities",
        "app.services.stockformer_corporate_ledger",
        "app.services.stockformer_corporate_receipts",
        "app.services.market_gate",
        "app.services.regime_builder",
    ]
    return {
        "bundle": pair.digest(Path(frozen) / "bundle.json"),
        "code": {n: pair.digest(importlib.import_module(n).__file__) for n in names},
    }


def configure(data_dir, frozen, *, start_date):
    frozen = Path(frozen)
    target = daily._root(view(data_dir)) / "config.json"
    if target.exists():
        raise ValueError("rank shadow already configured")
    model = verify_bundle(frozen)
    now = cn_now()
    if start_date < max(now.date(), datetime.fromisoformat(model["frozen_at"]).date()):
        raise ValueError("Start cannot use a past date or precede actual model freeze")
    parent = daily._read(frozen / "parent-protocol.json")
    inputs = view(data_dir)
    inputs.mkdir(parents=True, exist_ok=True)
    for table in observer.INPUT_TABLES:
        link = inputs / table
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != (Path(data_dir) / table).resolve():
                raise ValueError("Unexpected rank data view: " + table)
        else:
            link.symlink_to(Path("../../../") / table, target_is_directory=True)
    protocol = dict(
        version="minute_rank_pair_daily_shadow_v1",
        variant="repaired",
        namespace="rank_pair_shadow",
        pair_id=model["pair_id"],
        model_id="frozen_pair:" + model["pair_id"],
        start_date=str(start_date),
        frozen_root=str(frozen),
        assets=assets(frozen),
        actual_model_freeze=model["frozen_at"],
        execution=parent["execution"],
        signal_policy=parent["signal_policy"],
        min_net_edge=parent["min_net_edge"],
        minute_policy=model["minute_policy"],
        feature_columns=model["feature_columns"],
        readiness_columns=parent["readiness_columns"],
        training_universe=daily._read(frozen / "universe.json")["symbols"],
        ranking_column="pred_ret_raw_5d",
        minimum_active_coverage=0.98,
        signal_weights={"5d": 1.0},
        activation_allowed=False,
        run_after="19:30",
        cutoff="same_session_19:30_to_23:59_Asia/Shanghai",
        corporate_action_policy=LEDGER_POLICY,
        corporate_reference_after="21:05",
        corporate_cash_policy="simulated_close_on_or_after_announced_pay_date",
        corporate_tax_policy="sale_accrual_and_max_20_percent_open_reserve_not_broker_debit",
        order_quantity_policy=QUANTITY_POLICY,
        price_limit_mode="authoritative_day",
        research_warnings=[
            "独立分钟排序研究账户；零成交收益为空；旧回放不计前向窗口。",
            "历史源版本未完全证明；公司行动引用未经核实则阻断成交。",
        ],
    )
    pid = daily._hash(protocol)
    cfg = dict(
        enabled=True,
        protocol=protocol,
        protocol_id=pid,
        account_id="sf_rank_" + pid[:16],
        created_at=now.isoformat(),
    )
    daily._write(target, cfg)
    daily._account(inputs, cfg)
    return cfg


def prepare(inputs, cfg, day):
    # Source completeness is checked in the original 80-feature representation;
    # the nine cross-sectional minute transforms exist only after reference selection.
    source_cfg = {
        **cfg,
        "protocol": {**cfg["protocol"], "feature_columns": cfg["protocol"]["readiness_columns"]},
    }
    return _PREPARE(inputs, source_cfg, day)


def checked_market(inputs, day):
    try:
        return rank_inputs.require_market_days(market_gate.market_gate_frame(inputs), [day])
    except ValueError as exc:
        raise observer.WaitingDataError("当日真实市场输入未齐：" + str(exc)) from None


def compute(inputs, cfg, day):
    frozen = Path(cfg["protocol"]["frozen_root"])
    model = verify_bundle(frozen)
    rank_inputs.assert_post_freeze([day], model["frozen_at"])
    report = prepare(inputs, cfg, day)
    folder = Path(report["path"])
    features = pl.read_parquet(folder / "features.parquet")
    raw = pl.read_parquet(folder / "raw.parquet")
    parent = daily._read(frozen / "parent-protocol.json")
    gates = checked_market(inputs, day)
    target = inputs / "research_inputs/rank-materializations" / (str(day) + "-" + uuid4().hex[:12])
    target.mkdir(parents=True, exist_ok=False)
    gates.write_parquet(target / "market.parquet")
    with rank_inputs.pinned_market(inputs, gates):
        reference, audit = rank_inputs.reference(inputs, features, raw, parent)
        prediction, _ = gross.predict_gross_pair(frozen / "latest-pair", reference)
        full, candidates = necessary.reference_signals(
            inputs,
            reference,
            prediction,
            min_net_edge=parent["min_net_edge"],
            signal_policy=parent["signal_policy"],
        )
    complete, _ = rank_inputs.complete_only(features, parent["readiness_columns"])
    from app.services.stockformer_affordability_research import affordable_pool
    from app.services.stockformer_decision_research import decision_pool

    known, _ = decision_pool(complete)
    affordable, _ = affordable_pool(complete, raw, parent["execution"])
    stages = dict(
        closed_raw_bar=set(raw.filter(pl.col("date") == day)["symbol"]),
        produced_features=set(features["symbol"]),
        complete_features=set(complete["symbol"]),
        known_day_pool=set(known["symbol"]),
        affordable_pool=set(affordable["symbol"]),
        fixed_scoring_reference=set(reference["symbol"]),
    )
    trace = rank_inputs.annotated_rejections(
        cfg["protocol"]["training_universe"], day, stages, prediction, full
    )
    trace.write_parquet(target / "filter-trace.parquet")
    source = dict(
        date=str(day),
        matched_inputs=report,
        reference=audit,
        market_captured_at=cn_now().isoformat(),
        files={p.name: pair.digest(p) for p in target.glob("*.parquet")},
    )
    daily._write(target / "report.json", source)
    coverage = {
        **report["coverage"],
        "decision_pool_rows": candidates.height,
        "reference_rows": reference.height,
        "matched_inputs": report,
        "rank_inputs": {"path": str(target), "report_sha256": pair.digest(target / "report.json")},
    }
    return prediction, full, reference, coverage


@contextmanager
def environment(frozen):
    old_scope = observer.scope
    replacements = [
        (
            observer,
            "scope",
            lambda root, *, variant="frozen": (
                scope(root) if variant == "repaired" else old_scope(root, variant=variant)
            ),
        ),
        (observer, "_verify_pair", verify_bundle),
        (observer, "_assets", assets),
        (observer, "_compute", compute),
        (repaired, "prepare", prepare),
    ]
    saved = [(module, name, getattr(module, name)) for module, name, _ in replacements]
    try:
        for module, name, value in replacements:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


def tick(data_dir, *, now=None):
    now = now or cn_now()
    cfg = daily._read(daily._root(view(data_dir)) / "config.json")
    if not cfg:
        return {"status": "disabled"}
    if now < datetime.fromisoformat(cfg["created_at"]):
        return {"status": "before_start"}
    with environment(Path(cfg["protocol"]["frozen_root"])):
        return observer.tick(data_dir, now=now, variant="repaired")


def status(data_dir):
    cfg = daily._read(daily._root(view(data_dir)) / "config.json")
    if not cfg:
        return {"status": "disabled", "activation_allowed": False}
    with environment(Path(cfg["protocol"]["frozen_root"])):
        return observer.status(data_dir, variant="repaired")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--configure", type=Path)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.configure:
        cfg = configure(
            args.data_dir, args.configure, start_date=args.start_date or cn_now().date()
        )
        print("configured independent rank account " + cfg["account_id"])
        return
    while True:
        now = cn_now()
        state = daily._read(daily._root(view(args.data_dir)) / "state.json")
        if state.get("as_of") != str(now.date()) or daily.due(view(args.data_dir), now):
            result = tick(args.data_dir, now=now)
            logger.info("rank shadow %s", result.get("status"))
        daily._write(
            scope(args.data_dir) / "worker.json",
            {
                "checked_at": cn_now().isoformat(),
                "pair_id": PAIR_ID,
                "schedule": "19:30后每15分钟检查完整输入；只记录实际当日",
            },
        )
        if args.once:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
