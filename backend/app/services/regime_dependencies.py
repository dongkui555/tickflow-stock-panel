"""Determine market repair dates from changed stock inputs, not file timestamps."""

from __future__ import annotations

import polars as pl

from app.parquet import scan_parquet_compat
from app.services import maintenance_queue, regime_builder


def _inputs(frame, instruments, shares):
    from app.indicators.pipeline import compute_indicators, compute_limit_signals

    frame = frame.sort(["symbol", "date"])
    frame = compute_indicators(frame, needed={"change_pct", "ma20"})
    if not instruments.is_empty():
        frame = compute_limit_signals(frame, instruments, historical_shares=shares,
            needed={"signal_limit_up", "signal_limit_down", "signal_broken_limit_up"})
    return frame.select(
        "symbol", "date", "amount", "raw_close", "raw_high", "raw_low", "consecutive_limit_ups",
        pl.col("raw_close").shift(1).over("symbol").alias("previous_raw_close"),
        pl.col("consecutive_limit_ups").shift(1).over("symbol").alias("previous_boards"),
        pl.col("change_pct").round(12),
        # Preserve threshold crossings even if the rounded return is the same.
        (pl.col("change_pct") > 0).alias("up"),
        (pl.col("change_pct") < 0).alias("down"),
        (pl.col("change_pct") >= .03).alias("strong_up"),
        (pl.col("change_pct") <= -.03).alias("strong_down"),
        (pl.col("ma20").is_not_null() & (pl.col("ma20") > 0)).alias("valid_ma20"),
        (pl.col("close") > pl.col("ma20")).alias("above_ma20"),
        *[c for c in ("signal_limit_up", "signal_limit_down", "signal_broken_limit_up") if c in frame.columns],
    )


def changed_dates(before, after, instruments, shares=None):
    """Compare full affected-stock histories, including their rolling dependencies."""
    dates = sorted(set(before["date"].to_list() if "date" in before.columns else [])
                   | set(after["date"].to_list() if "date" in after.columns else []))
    required = {"symbol", "date", "close", "raw_close", "raw_high", "raw_low", "amount", "consecutive_limit_ups"}
    if before.is_empty() or after.is_empty() or not all(required <= set(f.columns) for f in (before, after)):
        return dates
    old = _inputs(before, instruments, shares)
    new = _inputs(after, instruments, shares)
    columns = [c for c in old.columns if c not in {"symbol", "date"}]
    joined = old.with_columns(pl.lit(True).alias("_old")).join(
        new.with_columns(pl.lit(True).alias("_new")), on=["symbol", "date"],
        how="full", coalesce=True, suffix="_new")
    different = pl.any_horizontal([pl.col(c).ne_missing(pl.col(c + "_new")) for c in columns])
    return sorted(joined.filter(pl.col("_old").is_null() | pl.col("_new").is_null() | different)["date"].unique())


class AdjustmentScope:
    """Publish repair intent before stock writes; acknowledge invariance only after commit."""

    def __init__(self, repo, symbols):
        from app.indicators.pipeline import _filter_lazy_by_symbol_keys

        self.root = repo.store.data_dir
        files = sorted((self.root / "kline_daily_enriched").glob("date=*/part.parquet"))
        self.before = (_filter_lazy_by_symbol_keys(scan_parquet_compat(files), symbols).collect()
                       if files else pl.DataFrame())
        self.instruments = repo.get_instruments()
        self.shares = repo.get_historical_shares()
        self.state = regime_builder._source_state(self.root)
        self.fresh = {}
        for day in self.before["date"].unique() if "date" in self.before.columns else []:
            stamp = regime_builder._source_stamp(self.root, day)
            if self.state["dates"].get(str(day)) == stamp:
                self.fresh[str(day)] = stamp
        self.changed = set()
        self.seen = set()

    def before_publish(self, enriched):
        from app.indicators.pipeline import _filter_by_symbol_keys, _select_storage_cols

        after = _select_storage_cols(enriched)
        old = _filter_by_symbol_keys(self.before, after["symbol"].unique().to_list()) if not self.before.is_empty() else self.before
        changed = changed_dates(old, after, self.instruments, self.shares)
        self.changed.update(changed)
        self.seen.update(after["date"].to_list())
        # A crash after any parquet write must leave the real dependency in the durable queue.
        maintenance_queue.enqueue(self.root, "regime", changed, reason="input_changed")

    def committed(self):
        state = regime_builder._source_state(self.root)
        for day in self.seen - self.changed:
            key = str(day)
            prior = self.fresh.get(key)
            if prior is None or state["dates"].get(key) != prior:
                continue
            current = regime_builder._source_stamp(self.root, day)
            old_index = {k: v for k, v in prior.items() if k.startswith("kline_index_daily")}
            new_index = {k: v for k, v in current.items() if k.startswith("kline_index_daily")}
            if old_index == new_index:
                state["dates"][key] = current
        regime_builder._save_source_state(self.root, state)
