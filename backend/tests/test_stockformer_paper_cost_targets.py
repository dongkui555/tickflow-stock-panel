from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from app.services import stockformer_training as training
from app.strategy import paper

EXECUTION = {"initial_cash": 10000., "max_positions": 3, "commission_pct": .0003,
             "stamp_tax_pct": .0005, "slippage_bps": 5., "minimum_commission": 5.}


def prices(values=(10., 10., 11.), other=20.):
    return pl.DataFrame([
        {"symbol": symbol, "date": date(2025, 1, 2) + timedelta(days=i),
         "open": price, "close": price, "high": price, "low": price,
         "volume": 10000., "industry": "bank"}
        for symbol, series in (("000001", values), ("000002", [other] * len(values)))
        for i, price in enumerate(series)
    ])


def test_paper_target_matches_actual_fill_ledger_and_net_metric_once(tmp_path, monkeypatch):
    raw = prices()
    result = training.paper_execution_targets(raw, raw, raw, execution=EXECUTION, horizons=(1,))
    row = result.filter((pl.col("symbol") == "000001") & (pl.col("date") == date(2025, 1, 2))).row(0, named=True)
    assert row["label_paper_qty"] == 300
    assert row["label_paper_status_1d"] == "priced"
    account = paper.create_account(tmp_path, 10000., account_id="cost_test", commission_pct=.0003,
                                   stamp_tax_pct=.0005, slippage_bps=5.)
    monkeypatch.setattr(paper, "_prev_close", lambda *a: None)
    buy = {"id": "buy", "symbol": "000001", "side": "buy", "qty": 300, "asset_type": "stock",
           "status": "pending", "created_at": "2025-01-02T19:00:00+08:00", "ref_price": 10., "source": "test"}
    assert paper._fill_order(tmp_path, buy, 10., "2025-01-03", account["id"])
    sell = {**buy, "id": "sell", "side": "sell", "status": "pending"}
    assert paper._fill_order(tmp_path, sell, 11., "2025-01-04", account["id"])
    trade = paper.round_trips(tmp_path, account["id"])[0]
    assert row["target_1d_net"] == pytest.approx(trade["pnl"] / (300 * 10.005 + 5), abs=2e-8)
    assert row["target_1d_rel_net"] == pytest.approx(row["target_1d_net"] - .05)
    metrics = training.metrics(result.head(1), np.array([1.]), np.array([row["target_1d_rel_net"]]), target_mode="industry_relative_net")
    assert metrics["top2_after_cost_proxy"] == pytest.approx(row["target_1d_rel_net"])


def test_flat_small_position_cost_includes_minimum_commission():
    raw = prices((10., 10., 10.))
    row = training.paper_execution_targets(raw, raw, raw, execution=EXECUTION, horizons=(1,)).row(0, named=True)
    assert row["target_1d_net"] < -.004
    assert row["label_cost_1d"] > .004
    legacy = training.execution_targets(raw, raw, horizons=(1,))
    assert legacy.row(0, named=True)["target_1d_net"] == pytest.approx(-.002)


def test_quantity_uses_signal_close_and_cannot_use_future_price():
    raw = prices((10., 20., 21.))
    result = training.paper_execution_targets(raw, raw, raw, execution=EXECUTION, horizons=(1,))
    assert result.row(0, named=True)["label_paper_qty"] == 300
    changed = raw.with_columns(pl.when(pl.col("date") > date(2025, 1, 2)).then(pl.col("open") * .5).otherwise(pl.col("open")).alias("open"))
    changed = changed.with_columns(pl.col("open").alias("high"), pl.col("open").alias("low"), pl.col("open").alias("close"))
    second = training.paper_execution_targets(changed, changed, changed, execution=EXECUTION, horizons=(1,))
    assert second.row(0, named=True)["label_paper_qty"] == 300


def test_unaffordable_lot_and_corporate_action_are_null_not_false_profit():
    raw = prices((40., 40., 40.))
    result = training.paper_execution_targets(raw, raw, raw, execution=EXECUTION, horizons=(1,))
    row = result.row(0, named=True)
    assert row["label_paper_qty"] == 0
    assert row["target_1d_net"] is None
    assert row["label_paper_status_1d"] == "one_lot_budget"
    raw = prices((10., 10., 5.))
    adjusted = prices((5., 5., 5.))
    row = training.paper_execution_targets(adjusted, adjusted, raw, execution=EXECUTION, horizons=(1,)).row(0, named=True)
    assert row["label_paper_status_1d"] == "corporate_action_or_price_mismatch"
    assert row["target_1d_rel_net"] is None


def test_missing_next_session_never_borrows_following_stock_row():
    raw = prices((10., 10., 11., 12.))
    missing = raw.filter(~((pl.col("symbol") == "000001") & (pl.col("date") == date(2025, 1, 3))))
    row = training.paper_execution_targets(raw, raw, missing, execution=EXECUTION, horizons=(1,)).row(0, named=True)
    assert row["target_1d_net"] is None
    assert row["label_paper_status_1d"] == "missing_execution_prices"


def test_identity_conflicts_and_unknown_minimum_fee_fail_closed():
    raw = prices()
    with pytest.raises(ValueError, match="Duplicate"):
        training.paper_execution_targets(raw, raw, pl.concat([raw, raw.head(1)]), execution=EXECUTION, horizons=(1,))
    with pytest.raises(ValueError, match="minimum commission"):
        training.paper_execution_targets(raw, raw, raw, execution={**EXECUTION, "minimum_commission": 0.}, horizons=(1,))


def test_costs_do_not_disappear_in_industry_demeaning_or_unaffordable_filter():
    raw = prices()
    raw = raw.with_columns(pl.when(pl.col("symbol") == "000002")
                           .then(pl.when(pl.col("date") == date(2025, 1, 4)).then(48.).otherwise(40.))
                           .otherwise(pl.col("open")).alias("open"))
    raw = raw.with_columns([pl.col("open").alias(c) for c in ("close", "high", "low")])
    result = training.paper_execution_targets(raw, raw, raw, execution=EXECUTION, horizons=(1,))
    row = result.row(0, named=True)
    assert row["target_1d_rel_net"] == pytest.approx(row["target_1d_net"] - .15)
    expensive = result.filter(pl.col("symbol") == "000002").row(0, named=True)
    assert expensive["target_1d_rel_net"] is None


def test_large_position_uses_percentage_commission_above_minimum():
    raw = prices((10., 10., 10.))
    cfg = {**EXECUTION, "initial_cash": 1000000.}
    row = training.paper_execution_targets(raw, raw, raw, execution=cfg, horizons=(1,)).row(0, named=True)
    qty = row["label_paper_qty"]
    paid = qty * 10.005 + paper.buy_fee(qty, 10.005, .0003)
    received = qty * 9.995 - paper.sell_fee(qty, 9.995, .0003, .0005)
    assert paper.buy_fee(qty, 10.005, .0003) > 5
    assert row["target_1d_net"] == pytest.approx((received - paid) / paid)


def test_inner_training_labels_end_strictly_before_calibration():
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(40)]
    frame = pl.DataFrame({"symbol": ["000001"] * 40, "date": dates,
                          "label_end_5d": [day + timedelta(days=6) for day in dates]})
    train, calibration, cutoff = training.purged_inner_split(frame, 5)
    assert train["label_end_5d"].max() < cutoff
    assert calibration["date"].min() == cutoff
    assert len(frame.filter(pl.col("date") < cutoff)) - train.height == 6


def test_costed_training_cannot_implicitly_activate(tmp_path):
    from app.services import stockformer as sf

    config = sf.StockFormerTrainConfig(execution_costs=EXECUTION, target_mode="industry_relative_net")
    result = sf.train_model(tmp_path, config, activate=True)
    assert "研究" in result["error"]
    assert not (tmp_path / "stockformer/models.json").exists()


def test_raw_loader_uses_final_history_and_quarantines_conflicting_aliases(tmp_path):
    day = date(2025, 1, 2)
    root = tmp_path / 'kline_daily' / f'date={day}'
    root.mkdir(parents=True)
    rows = []
    for symbol, price, stamp in [('000001', 10., None), ('000001.SZ', 99., 1000),
                                  ('000002', 20., None), ('000002.SZ', 21., None)]:
        rows.append(dict(symbol=symbol, date=day, open=price, high=price, low=price,
                         close=price, volume=100., amount=price*100, quote_ts=stamp))
    pl.DataFrame(rows).write_parquet(root / 'part.parquet')
    raw, conflicts = training.load_paper_raw_prices(tmp_path, ['000001', '000002'], start=day, end=day)
    assert raw.select('symbol', 'close').to_dicts() == [{'symbol': '000001', 'close': 10.}]
    assert conflicts['symbol'].to_list() == ['000002']
