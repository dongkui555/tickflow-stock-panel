from contextlib import nullcontext
from datetime import datetime, timedelta

import polars as pl
import pytest

from app.market_time import CN_TZ
from app.services import stockformer_daily_shadow as daily
from app.services import stockformer_pair_shadow as observer
from app.services import stockformer_rank_shadow as rank

NOW = datetime(2026, 11, 4, 22, 0, tzinfo=CN_TZ)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    parent = {
        "execution": daily._EXECUTION,
        "signal_policy": {"prediction_rule": {"min_net_edge": 0.0005}},
        "readiness_columns": ["complete80"],
        "min_net_edge": 0.0005,
    }
    daily._write(frozen / "parent-protocol.json", parent)
    daily._write(frozen / "universe.json", {"symbols": ["000001"]})
    meta = {
        "pair_id": "rank-pair",
        "frozen_at": "2026-10-30T19:06:36+08:00",
        "feature_columns": ["xr_minute_behavior"],
        "minute_policy": {"version": "complete_session_240_v1"},
    }
    monkeypatch.setattr(rank, "verify_bundle", lambda *a: meta)
    monkeypatch.setattr(rank, "assets", lambda *a: {"unchanged": True})
    monkeypatch.setattr(rank, "cn_now", lambda: NOW)
    cfg = rank.configure(tmp_path, frozen, start_date=NOW.date())
    return tmp_path, frozen, cfg


def test_config_has_distinct_namespace_and_real_publication_time(configured):
    root, _, cfg = configured
    assert cfg["account_id"].startswith("sf_rank_")
    assert cfg["protocol"]["pair_id"] == "rank-pair"
    assert cfg["created_at"] == NOW.isoformat()
    assert rank.view(root) == root / "stockformer/rank_pair_shadow/view"
    assert not (root / "stockformer/repaired_pair_shadow").exists()
    assert cfg["protocol"]["activation_allowed"] is False
    assert cfg["protocol"]["execution"] == daily._EXECUTION


def test_existing_config_is_immutable(configured):
    root, frozen, _ = configured
    original = (daily._root(rank.view(root)) / "config.json").read_bytes()
    with pytest.raises(ValueError, match="already configured"):
        rank.configure(root, frozen, start_date=NOW.date())
    assert (daily._root(rank.view(root)) / "config.json").read_bytes() == original


def test_config_cannot_backdate(configured):
    root, frozen, _ = configured
    other = root / "other"
    with pytest.raises(ValueError, match=r"past|freeze"):
        rank.configure(other, frozen, start_date=NOW.date() - timedelta(days=1))
    assert not (daily._root(rank.view(other)) / "config.json").exists()


def test_prepare_does_not_request_rank_features_before_transform(configured, monkeypatch):
    root, _, cfg = configured
    captured = []
    monkeypatch.setattr(
        rank, "_PREPARE", lambda inputs, config, day: captured.append(config) or {"ok": True}
    )
    assert rank.prepare(rank.view(root), cfg, NOW.date()) == {"ok": True}
    assert captured[0]["protocol"]["feature_columns"] == ["complete80"]
    assert cfg["protocol"]["feature_columns"] == ["xr_minute_behavior"]


def test_worker_bindings_are_scoped_and_restored(configured):
    root, frozen, _ = configured
    original = (observer.scope, observer._compute, observer._verify_pair, observer._assets)
    from app.services import stockformer_repaired_shadow as repaired

    original_prepare = repaired.prepare
    with pytest.raises(RuntimeError), rank.environment(frozen):
        assert observer.scope(root, variant="repaired") == rank.scope(root)
        assert observer.scope(root, variant="frozen") == root / "stockformer/frozen_pair_shadow"
        assert observer.scope(root, variant="decision") == root / "stockformer/decision_pair_shadow"
        assert observer._compute is rank.compute
        raise RuntimeError("interruption")
    assert (observer.scope, observer._compute, observer._verify_pair, observer._assets) == original
    assert repaired.prepare is original_prepare


def test_clock_cannot_precede_new_account_creation(configured, monkeypatch):
    root, _, _ = configured
    monkeypatch.setattr(
        rank, "environment", lambda *a: pytest.fail("cannot enter historical execution")
    )
    result = rank.tick(root, now=NOW - timedelta(minutes=1))
    assert result["status"] == "before_start"
    assert not (daily._root(rank.view(root)) / "days").exists()


@pytest.mark.parametrize("kind", ["missing", "duplicate", "nonfinite", "neutral"])
def test_market_inputs_never_default_to_neutral(tmp_path, monkeypatch, kind):
    frame = pl.DataFrame(
        {
            "date": [NOW.date()],
            "market_gate": ["open"],
            **{
                c: [60.0]
                for c in [
                    "market_regime_score",
                    "market_above_ma20_pct",
                    "market_strong_down_pct",
                    "market_position_multiplier",
                    "market_buy_threshold_add",
                ]
            },
            "market_allows_open": [True],
        }
    )
    if kind == "missing":
        frame = frame.filter(pl.lit(False))
    elif kind == "duplicate":
        frame = pl.concat([frame, frame])
    elif kind == "nonfinite":
        frame = frame.with_columns(pl.lit(float("nan")).alias("market_regime_score"))
    else:
        frame = frame.with_columns(pl.lit("neutral").alias("market_gate"))
    monkeypatch.setattr(rank.market_gate, "market_gate_frame", lambda *a: frame)
    with pytest.raises(observer.WaitingDataError):
        rank.checked_market(tmp_path, NOW.date())


def test_tick_calls_original_engine_in_new_namespace(configured, monkeypatch):
    root, _, _ = configured
    captured = []
    monkeypatch.setattr(rank, "environment", lambda *a: nullcontext())
    monkeypatch.setattr(
        observer, "tick", lambda *a, **k: captured.append(k) or {"status": "waiting_data"}
    )
    assert rank.tick(root, now=NOW)["status"] == "waiting_data"
    assert captured == [dict(now=NOW, variant="repaired")]
