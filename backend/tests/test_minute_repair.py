import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl

from app.plugins.tdx.provider import TdxProvider
from app.services.minute_repair import (
    checked_additions,
    conflicting_complete_codes,
    daily_reference,
    qmt_session_is_valid,
    repair_minute_gaps,
)

DAY = date(2026, 5, 8)


def bars(symbol="000001.SZ"):
    times = [datetime(2026, 5, 8, 9, 31) + timedelta(minutes=i) for i in range(120)]
    times += [datetime(2026, 5, 8, 13, 1) + timedelta(minutes=i) for i in range(120)]
    return pl.DataFrame({"symbol": [symbol] * 240, "datetime": times,
                         "open": [10.] * 240, "high": [10.] * 240,
                         "low": [10.] * 240, "close": [10.] * 240,
                         "volume": [1.] * 240, "amount": [1000.] * 240})


def reference():
    return {"symbol": "000001.SZ", "open": 10., "high": 10., "low": 10.,
            "close": 10., "volume": 240., "amount": 240000.}


def bj_bars_with_after_hours():
    core = bars("920000.BJ")
    late_times = [datetime(2026, 5, 8, 15, 1) + timedelta(minutes=i) for i in range(30)]
    late = pl.DataFrame({
        "symbol": ["920000.BJ"] * 30,
        "datetime": late_times,
        "open": [10.] * 30,
        "high": [10.] * 30,
        "low": [10.] * 30,
        "close": [10.] * 30,
        "volume": [30.] + [0.] * 29,
        "amount": [30000.] + [0.] * 29,
    })
    return pl.concat([core, late])


class RepairTests(unittest.TestCase):
    def test_opening_zero_turnover_quote_does_not_define_daily_trade_prices(self):
        frame = bars()
        stamp = pl.col("datetime")
        no_trade = stamp == frame["datetime"][0]
        frame = frame.with_columns(
            [pl.when(no_trade).then(9.0).otherwise(pl.col(c)).alias(c)
             for c in ("open", "high", "low", "close")]
            + [pl.when(no_trade).then(0.0).otherwise(pl.col(c)).alias(c)
               for c in ("volume", "amount")]
        )
        ref = reference() | {"volume": 239.0, "amount": 239000.0}
        delta, reason = checked_additions(frame, frame.clear(), ref, DAY)
        self.assertEqual(reason, "accepted")
        self.assertTrue(delta.equals(frame))
        self.assertEqual(conflicting_complete_codes(frame, daily_reference(pl.DataFrame([ref])), DAY), set())

    def test_terminal_zero_turnover_quote_defines_daily_close(self):
        frame = bars().with_columns(
            pl.when(pl.col("datetime") == bars()["datetime"][-1])
            .then(9.0).otherwise(pl.col("open")).alias("open"),
            pl.when(pl.col("datetime") == bars()["datetime"][-1])
            .then(9.0).otherwise(pl.col("high")).alias("high"),
            pl.when(
                (pl.col("datetime") == datetime(2026, 5, 8, 13, 1))
                | (pl.col("datetime") == bars()["datetime"][-1])
            )
            .then(9.0).otherwise(pl.col("low")).alias("low"),
            pl.when(pl.col("datetime") == bars()["datetime"][-1])
            .then(9.0).otherwise(pl.col("close")).alias("close"),
            pl.when(pl.col("datetime") == bars()["datetime"][-1])
            .then(0.0).otherwise(pl.col("volume")).alias("volume"),
            pl.when(pl.col("datetime") == bars()["datetime"][-1])
            .then(0.0).otherwise(pl.col("amount")).alias("amount"),
        )
        ref = reference() | {"low": 9.0, "close": 9.0, "volume": 239.0, "amount": 239000.0}
        delta, reason = checked_additions(frame, frame.clear(), ref, DAY)
        self.assertEqual((delta.height, reason), (240, "accepted"))
        self.assertEqual(conflicting_complete_codes(frame, daily_reference(pl.DataFrame([ref])), DAY), set())

    def test_opening_snapshot_range_does_not_pollute_daily_low(self):
        opening = bars().head(1).with_columns(
            pl.lit(datetime(2026, 5, 8, 9, 30)).alias("datetime"),
            pl.lit(9.0).alias("open"),
            pl.lit(9.0).alias("low"),
            pl.lit(10.0).alias("high"),
            pl.lit(10.0).alias("close"),
        )
        frame = pl.concat([opening, bars()])
        ref = reference() | {"volume": 241.0, "amount": 241000.0}
        delta, reason = checked_additions(
            frame, frame.clear(), ref, DAY, source="qmt"
        )
        self.assertEqual((delta.height, reason), (241, "accepted"))
        self.assertEqual(
            conflicting_complete_codes(
                frame,
                daily_reference(pl.DataFrame([ref])),
                DAY,
                source="qmt",
            ),
            set(),
        )

    def test_staged_minute_preserves_opening_snapshot_contract(self):
        opening = bars().head(1).with_columns(
            pl.lit(datetime(2026, 5, 8, 9, 30)).alias("datetime"),
            pl.lit(9.0).alias("open"),
            pl.lit(9.0).alias("low"),
            pl.lit(99.0).alias("high"),
            pl.lit(9.0).alias("close"),
        )
        frame = pl.concat([opening, bars()])
        ref = reference() | {"volume": 241.0, "amount": 241000.0}
        delta, reason = checked_additions(
            frame, frame.clear(), ref, DAY, source="staged_minute"
        )
        self.assertEqual((delta.height, reason), (241, "accepted"))

    def test_traded_0930_is_part_of_non_qmt_daily_ohlc(self):
        opening = bars().head(1).with_columns(
            pl.lit(datetime(2026, 5, 8, 9, 30)).alias("datetime"),
            pl.lit(9.0).alias("open"),
            pl.lit(10.0).alias("high"),
            pl.lit(9.0).alias("low"),
            pl.lit(9.5).alias("close"),
        )
        frame = pl.concat([opening, bars()])
        ref = reference() | {
            "open": 9.0,
            "low": 9.0,
            "volume": 241.0,
            "amount": 241000.0,
        }
        delta, reason = checked_additions(
            frame, frame.clear(), ref, DAY, source="tushare"
        )
        self.assertEqual((delta.height, reason), (241, "accepted"))
        refs = daily_reference(pl.DataFrame([ref]))
        self.assertEqual(
            conflicting_complete_codes(frame, refs, DAY, source="tushare"),
            set(),
        )
        rejected, reason = checked_additions(
            frame,
            frame.clear(),
            ref | {"open": 10.0, "low": 10.0},
            DAY,
            source="tushare",
        )
        self.assertEqual((rejected.height, reason), (0, "daily_price_conflict"))

    def test_beijing_after_hours_are_part_of_daily_turnover_contract(self):
        frame = bj_bars_with_after_hours()
        ref = {
            "symbol": "920000.BJ", "open": 10., "high": 10., "low": 10.,
            "close": 10., "volume": 27000., "amount": 270000.,
        }
        accepted, reason = checked_additions(
            frame, frame.clear(), ref, DAY, source="stockdb"
        )
        self.assertEqual((accepted.height, reason), (270, "accepted"))
        self.assertEqual(
            conflicting_complete_codes(
                frame, daily_reference(pl.DataFrame([ref])), DAY, source="stockdb"
            ),
            set(),
        )

    def test_zero_turnover_0930_quote_never_changes_daily_ohlc(self):
        opening = bars().head(1).with_columns(
            pl.lit(datetime(2026, 5, 8, 9, 30)).alias("datetime"),
            pl.lit(9.0).alias("open"),
            pl.lit(10.0).alias("high"),
            pl.lit(9.0).alias("low"),
            pl.lit(9.5).alias("close"),
            pl.lit(0.0).alias("volume"),
            pl.lit(0.0).alias("amount"),
        )
        frame = pl.concat([opening, bars()])
        delta, reason = checked_additions(
            frame, frame.clear(), reference(), DAY, source="tushare"
        )
        self.assertEqual((delta.height, reason), (241, "accepted"))

    def test_positive_amount_with_zero_volume_still_defines_trade_prices(self):
        frame = bars().with_columns(
            pl.when(pl.col("datetime") == bars()["datetime"][0])
            .then(0.0).otherwise(pl.col("volume")).alias("volume"),
            pl.when(pl.col("datetime") == bars()["datetime"][0])
            .then(11.0).otherwise(pl.col("high")).alias("high"),
        )
        ref = reference() | {"volume": 239.0, "high": 11.0}
        delta, reason = checked_additions(frame, frame.clear(), ref, DAY)
        self.assertEqual((delta.height, reason), (240, "accepted"))
        self.assertEqual(conflicting_complete_codes(frame, daily_reference(pl.DataFrame([ref])), DAY), set())
        delta, reason = checked_additions(frame, frame.clear(), ref | {"high": 10.0}, DAY)
        self.assertEqual((delta.height, reason), (0, "daily_price_conflict"))

    def test_real_trade_price_conflict_not_hidden_by_zero_turnover_quotes(self):
        frame = bars().with_columns(
            pl.when(pl.col("datetime") == bars()["datetime"][0])
            .then(11.0).otherwise(pl.col("high")).alias("high")
        )
        delta, reason = checked_additions(frame, frame.clear(), reference(), DAY)
        self.assertEqual((delta.height, reason), (0, "daily_price_conflict"))

    def test_tdx_beijing_repair_reaches_provider_and_keeps_validation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dp = root / f"kline_daily/date={DAY}/part.parquet"
            dp.parent.mkdir(parents=True)
            pl.DataFrame([reference() | {"symbol": "920002.BJ"}]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            seen = []
            def get_minute(symbols, *args, **kw):
                seen.append(symbols)
                return bars("920002.BJ")
            source = SimpleNamespace(get_minute=get_minute)
            result = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="tdx",
                                       source_provider=source, stock_days=[("920002.BJ", DAY)])
            self.assertEqual(result["requests_checked"], 1)
            self.assertEqual(result["added_rows"], 240)
            self.assertNotIn("provider_unsupported_bj", result["reasons"])
            self.assertEqual(seen, [["920002.BJ"]])

    def test_single_day_mode_avoids_truncated_multi_day_responses(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for day in (DAY, DAY + timedelta(days=1)):
                dp = root / f"kline_daily/date={day}/part.parquet"
                dp.parent.mkdir(parents=True)
                pl.DataFrame([reference()]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            def minute(symbols, start, end, **kw):
                self.assertEqual(start.date(), end.date())
                return bars().with_columns(pl.col('datetime') + (start.date() - DAY))
            source = SimpleNamespace(get_minute=minute)
            with patch('app.services.minute_repair.kline_sync._resolve_minute_provider', return_value=(source, False, None)):
                result = repair_minute_gaps(repo, start_date=DAY, end_date=DAY + timedelta(days=1),
                                           provider_name='test', single_day_requests=True)
            self.assertEqual(result['requests_checked'], 2)
            self.assertEqual(result['added_rows'], 480)

    def test_invalid_day_in_partial_response_is_not_reported_empty(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for day in (DAY, DAY + timedelta(days=1)):
                dp = root / f"kline_daily/date={day}/part.parquet"
                dp.parent.mkdir(parents=True)
                pl.DataFrame([reference()]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            source = SimpleNamespace(get_minute_diagnostics=lambda *a, **kw: (bars(), [
                {"date": str(DAY + timedelta(days=1)), "reason": "daily_price_conflict", "retryable": False},
            ]))
            with patch("app.services.minute_repair.kline_sync._resolve_minute_provider", return_value=(source, False, None)):
                result = repair_minute_gaps(repo, start_date=DAY, end_date=DAY + timedelta(days=1), provider_name="test")
            self.assertEqual(result["added_rows"], 240)
            self.assertEqual(result["reasons"]["daily_price_conflict"], 1)
            self.assertNotIn("source_empty", result["reasons"])

    def test_exact_manifest_never_expands_to_other_stocks_or_dates(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for day in (DAY - timedelta(days=1), DAY):
                dp = root / f"kline_daily/date={day}/part.parquet"
                dp.parent.mkdir(parents=True)
                pl.DataFrame([reference(), reference() | {"symbol": "000002.SZ"}]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            calls = []
            def get_minute(symbols, start, end, **kw):
                calls.append((symbols, start.date(), end.date()))
                return bars()
            source = SimpleNamespace(get_minute=get_minute)
            with patch("app.services.minute_repair.kline_sync._resolve_minute_provider", return_value=(source, False, None)):
                result = repair_minute_gaps(repo, start_date=DAY - timedelta(days=1), end_date=DAY,
                                            provider_name="test", stock_days=[("000001.SZ", DAY)])
            self.assertEqual(calls, [(["000001.SZ"], DAY, DAY)])
            self.assertEqual(result["added_rows"], 240)

    def test_request_budget_reports_deferred_stock_days(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dp = root / f"kline_daily/date={DAY}/part.parquet"
            dp.parent.mkdir(parents=True)
            pl.DataFrame([reference(), reference() | {"symbol": "000002.SZ"}]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            source = SimpleNamespace(get_minute=lambda *a, **kw: bars())
            with patch("app.services.minute_repair.kline_sync._resolve_minute_provider", return_value=(source, False, None)):
                result = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test", max_requests=1)
            self.assertEqual(result["requests_checked"], 1)
            self.assertEqual(result["reasons"]["deferred_request_stock_days"], 1)
            self.assertFalse(result["complete"])

    def test_partial_fill_preserves_symbol_and_existing_keys(self):
        old = bars("000001").head(10)
        delta, reason = checked_additions(bars(), old, reference(), DAY)
        self.assertEqual(reason, "accepted")
        self.assertEqual(delta.height, 230)
        self.assertEqual(delta["symbol"].unique().to_list(), ["000001"])
        self.assertEqual(delta.join(old, on="datetime").height, 0)

    def test_scheduled_gap_fill_uses_same_independent_validation(self):
        from app.services.kline_sync import _verified_closed_minute_additions
        old = bars("000001").head(10)
        daily = pl.DataFrame([reference()])
        delta = _verified_closed_minute_additions(bars(), old, daily, DAY)
        self.assertEqual(delta.height, 230)
        self.assertEqual(delta["symbol"].unique().to_list(), ["000001"])
        bad = bars().with_columns(pl.lit(9.0).alias("close"))
        self.assertTrue(_verified_closed_minute_additions(bad, old, daily, DAY).is_empty())
        conflicting_old = old.with_columns(pl.lit(9.0).alias("close"))
        self.assertTrue(_verified_closed_minute_additions(bars(), conflicting_old, daily, DAY).is_empty())

    def test_daily_share_contract_does_not_rescale_minute_volume(self):
        ref = reference() | {"volume": 24000.}
        delta, reason = checked_additions(bars(), bars().clear(), ref, DAY)
        self.assertEqual(reason, "accepted")
        self.assertEqual(delta["volume"].sum(), 240.)

    def test_minute_share_contract_matches_daily_lots_without_rewriting(self):
        shares = bars().with_columns((pl.col("volume") * 100).alias("volume"))
        delta, reason = checked_additions(shares, shares.clear(), reference(), DAY)
        self.assertEqual(reason, "accepted")
        self.assertEqual(delta["volume"].sum(), 24000.)

    def test_inconsistent_minute_units_cannot_pass_via_daily_unit_fallback(self):
        wrong = bars().with_columns((pl.col("volume") / 100).alias("volume"))
        delta, reason = checked_additions(wrong, wrong.clear(), reference(), DAY)
        self.assertEqual((delta.height, reason), (0, "minute_turnover_unit_conflict"))

    def test_incomplete_wrong_date_and_invalid_values_rejected(self):
        variants = [bars().head(239), bars().with_columns(pl.lit(float("nan")).alias("close")),
                    bars().with_columns((pl.col("datetime") + timedelta(days=1)).alias("datetime"))]
        for frame in variants:
            delta, _ = checked_additions(frame, bars().clear(), reference(), DAY)
            self.assertEqual(delta.height, 0)

    def test_qmt_sparse_trade_minutes_are_accepted_only_when_daily_values_match(self):
        sparse = bars("920001.BJ").with_row_index("_row").filter(
            pl.col("_row") % 4 != 0
        ).drop("_row")
        ref = reference() | {
            "symbol": "920001.BJ",
            "volume": 180.0,
            "amount": 180000.0,
        }
        delta, reason = checked_additions(
            sparse, sparse.clear(), ref, DAY, source="qmt",
        )
        self.assertEqual((delta.height, reason), (180, "accepted"))

        strict_delta, strict_reason = checked_additions(
            sparse, sparse.clear(), ref, DAY, source="tushare",
        )
        self.assertEqual((strict_delta.height, strict_reason), (0, "source_incomplete"))

        bad_delta, bad_reason = checked_additions(
            sparse, sparse.clear(), ref | {"amount": 240000.0}, DAY, source="qmt",
        )
        self.assertEqual((bad_delta.height, bad_reason), (0, "daily_turnover_conflict"))

    def test_qmt_only_allows_the_first_opening_snapshot_exception(self):
        opening = bars("000001.SZ").head(1).with_columns(pl.lit(9.0).alias("open"))
        source = pl.concat([opening, bars("000001.SZ").tail(239)])
        ref = reference() | {"open": 9.0, "low": 9.0, "high": 11.0}
        accepted, reason = checked_additions(source, source.clear(), ref, DAY, source="qmt")
        self.assertEqual((accepted.height, reason), (240, "accepted"))

        rejected, reason = checked_additions(source, source.clear(), ref, DAY, source="tushare")
        self.assertEqual((rejected.height, reason), (0, "source_invalid"))

        later_bad = source.with_columns(
            pl.when(pl.col("datetime") == datetime(2026, 5, 8, 9, 32))
            .then(9.0).otherwise(pl.col("open")).alias("open")
        )
        rejected, reason = checked_additions(later_bad, later_bad.clear(), ref, DAY, source="qmt")
        self.assertEqual((rejected.height, reason), (0, "source_invalid"))

    def test_qmt_sparse_first_effective_minute_can_carry_opening_snapshot(self):
        sparse = bars("920957.BJ").filter(
            pl.col("datetime") != datetime(2026, 5, 8, 9, 31)
        ).with_columns(
            pl.when(pl.col("datetime") == datetime(2026, 5, 8, 9, 32))
            .then(9.0).otherwise(pl.col("open")).alias("open")
        )
        ref = reference() | {
            "symbol": "920957.BJ",
            "open": 9.0,
            "low": 9.0,
            "volume": 239.0,
            "amount": 239000.0,
        }

        accepted, reason = checked_additions(
            sparse, sparse.clear(), ref, DAY, source="qmt",
        )

        self.assertEqual((accepted.height, reason), (239, "accepted"))

    def test_qmt_close_minute_convention_is_terminal_when_daily_values_match(self):
        sparse = bars("000001.SZ").filter(
            pl.col("datetime") != datetime(2026, 5, 8, 14, 59)
        )
        ref = reference() | {"volume": 239.0, "amount": 239000.0}
        self.assertTrue(qmt_session_is_valid(sparse, DAY, ref))

    def test_overlap_conflict_is_not_overwritten(self):
        old = bars().head(10).with_columns(pl.lit(9.).alias("close"))
        delta, reason = checked_additions(bars(), old, reference(), DAY)
        self.assertEqual(reason, "existing_values_conflict")
        self.assertEqual(delta.height, 0)

    def test_small_per_bar_volume_differences_cannot_corrupt_merged_day(self):
        old = bars().head(239).with_columns(pl.lit(2.).alias("volume"))
        delta, reason = checked_additions(bars(), old, reference(), DAY)
        self.assertEqual((delta.height, reason), (0, "existing_values_conflict"))

    def test_daily_conflict_and_missing_reference_rejected(self):
        delta, reason = checked_additions(bars(), bars().clear(), reference() | {"close": 9.}, DAY)
        self.assertEqual((delta.height, reason), (0, "daily_price_conflict"))
        delta, reason = checked_additions(bars(), bars().clear(), None, DAY)
        self.assertEqual((delta.height, reason), (0, "daily_reference_missing"))

    def test_reference_prefers_canonical_alias(self):
        native = pl.DataFrame([reference()])
        alias = native.with_columns(pl.lit("000001").alias("symbol"), pl.lit(99.).alias("close"))
        self.assertEqual(daily_reference(pl.concat([native, alias]))["000001"]["close"], 10.)

    def test_complete_conflict_scan_distinguishes_daily_and_minute_units(self):
        refs = daily_reference(pl.DataFrame([reference()]))
        self.assertEqual(conflicting_complete_codes(bars(), refs, DAY), set())
        share_ref = daily_reference(pl.DataFrame([reference() | {"volume": 24000.}]))
        self.assertEqual(conflicting_complete_codes(bars(), share_ref, DAY), set())
        share_units = bars().with_columns((pl.col("volume") * 100).alias("volume"))
        self.assertEqual(conflicting_complete_codes(share_units, refs, DAY), set())
        for bad in (
            bars().with_columns(pl.lit(9.).alias("close")),
            bars().with_columns((pl.col("amount") * 2).alias("amount")),
        ):
            self.assertEqual(conflicting_complete_codes(bad, refs, DAY), {"000001"})
        self.assertEqual(conflicting_complete_codes(bars().head(239), refs, DAY), set())

    def test_conflict_scan_ignores_sparse_reference_metadata(self):
        refs = {f"{i:06d}": reference() | {"_code": f"{i:06d}", "quote_ts": None if i < 150 else 1788274776000}
                for i in range(200)}
        self.assertEqual(conflicting_complete_codes(bars(), refs, DAY), set())

    def test_complete_conflict_is_repaired_only_with_explicit_flag(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            mp = root / f"kline_minute/date={DAY}/part.parquet"
            dp = root / f"kline_daily/date={DAY}/part.parquet"
            mp.parent.mkdir(parents=True)
            dp.parent.mkdir(parents=True)
            old = bars("000001").with_columns(pl.lit(9.0).alias("close"))
            old.write_parquet(mp)
            pl.DataFrame([reference()]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            source = SimpleNamespace(get_minute=lambda *a, **k: bars())
            with patch("app.services.minute_repair.kline_sync._resolve_minute_provider", return_value=(source, False, None)):
                skipped = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test")
                self.assertEqual(skipped["requests_planned"], 0)
                repaired = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test", repair_conflicts=True, task_id="explicit-replacement")
                repeated = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test", repair_conflicts=True, task_id="explicit-replacement")
            self.assertEqual(repaired["replaced_rows"], 240)
            self.assertEqual(repaired["added_rows"], 0)
            self.assertEqual(repeated["requests_planned"], 0)
            self.assertTrue(pl.read_parquet(mp).select(bars().columns).equals(bars()))
            self.assertEqual(pl.read_parquet(mp)["security_key"].unique().to_list(), ["stock|SZ|000001"])
            backup = next((root / "import_backups").glob("*/date=*/part.parquet"))
            self.assertTrue(pl.read_parquet(backup).equals(old))

    def test_full_regular_session_can_add_a_verified_opening_record(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            mp = root / f"kline_minute/date={DAY}/part.parquet"
            dp = root / f"kline_daily/date={DAY}/part.parquet"
            mp.parent.mkdir(parents=True)
            dp.parent.mkdir(parents=True)
            old = bars()
            old.write_parquet(mp)
            opening = old.head(1).with_columns(pl.lit(datetime(2026, 5, 8, 9, 30)).alias("datetime"))
            source_frame = pl.concat([opening, old])
            pl.DataFrame([reference() | {"volume": 241., "amount": 241000.}]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            source = SimpleNamespace(get_minute=lambda *a, **k: source_frame)
            with patch("app.services.minute_repair.kline_sync._resolve_minute_provider", return_value=(source, False, None)):
                result = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test", repair_conflicts=True)
                repeated = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test", repair_conflicts=True)
            self.assertEqual((result["added_rows"], result["replaced_rows"]), (1, 0))
            self.assertEqual(repeated["requests_planned"], 0)
            self.assertTrue(pl.read_parquet(mp).select(source_frame.columns).equals(source_frame))

    def test_end_to_end_missing_only_backup_and_idempotence(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            mp = root / f"kline_minute/date={DAY}/part.parquet"
            dp = root / f"kline_daily/date={DAY}/part.parquet"
            mp.parent.mkdir(parents=True)
            dp.parent.mkdir(parents=True)
            old = bars("000001").head(10)
            old.write_parquet(mp)
            pl.DataFrame([reference()]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            source = SimpleNamespace(get_minute=lambda *a, **k: bars())
            with patch("app.services.minute_repair.kline_sync._resolve_minute_provider", return_value=(source, False, None)):
                result = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test")
                repeated = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test")
                empty_scope = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test", symbols=[])
            self.assertEqual(result["added_rows"], 230)
            self.assertEqual(repeated["added_rows"], 0)
            self.assertEqual(repeated["requests_planned"], 0)
            self.assertEqual(empty_scope["requests_planned"], 0)
            after = pl.read_parquet(mp)
            self.assertEqual(after.height, 240)
            self.assertTrue(after.head(10).select(old.columns).equals(old.with_columns(pl.lit("000001.SZ").alias("symbol"))))
            backup = next((root / "import_backups").glob("*/date=*/part.parquet"))
            self.assertTrue(pl.read_parquet(backup).equals(old))

    def test_scope_and_normalized_provider_lots_contract(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dp = root / f"kline_daily/date={DAY}/part.parquet"
            dp.parent.mkdir(parents=True)
            pl.DataFrame([reference(), reference() | {"symbol": "000002.SZ"}]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            def get_minute(symbols, *args, **kwargs):
                self.assertEqual(symbols, ["000001"])
                return bars()
            source = SimpleNamespace(get_minute=get_minute)
            with patch("app.services.minute_repair.kline_sync._resolve_minute_provider", return_value=(source, False, None)):
                result = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="tushare", symbols=["000001.SZ"])
            self.assertEqual(result["added_rows"], 240)
            frame = pl.read_parquet(root / f"kline_minute/date={DAY}/part.parquet")
            self.assertEqual(frame["volume"].sum(), 240)
            self.assertEqual(frame["symbol"].unique().to_list(), ["000001.SZ"])

    def test_exact_daily_identity_can_fill_missing_lifecycle_metadata(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dp = root / f"kline_daily/date={DAY}/part.parquet"
            ip = root / "instruments/instruments.parquet"
            dp.parent.mkdir(parents=True)
            ip.parent.mkdir(parents=True)
            pl.DataFrame([reference()]).write_parquet(dp)
            # Keep the lifecycle index non-empty while deliberately omitting
            # the target. The exact market-qualified daily row is historical
            # identity evidence for this stock-day.
            pl.DataFrame([{
                "symbol": "000002.SZ", "code": "000002",
                "exchange": "SZ", "list_status": "L",
            }]).write_parquet(ip)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            source = SimpleNamespace(get_minute=lambda *a, **k: bars())
            result = repair_minute_gaps(
                repo,
                start_date=DAY,
                end_date=DAY,
                provider_name="qmt",
                source_provider=source,
                bypass_failover=True,
                stock_days=[("000001", DAY)],
                identity_lookup={"000001": "000001.SZ"},
            )
            self.assertEqual(result["added_rows"], 240)
            self.assertEqual(result["reasons"]["lifecycle_daily_reference_fallback"], 1)
            self.assertTrue(pl.read_parquet(root / f"kline_minute/date={DAY}/part.parquet").select(bars().columns).equals(bars()))

    def test_opening_record_is_validated_without_inventing_a_bar(self):
        opening = bars().head(1).with_columns(pl.lit(datetime(2026, 5, 8, 9, 30)).alias("datetime"))
        source = pl.concat([opening, bars()])
        ref = reference() | {"volume": 241., "amount": 241000.}
        delta, reason = checked_additions(source, source.clear(), ref, DAY)
        self.assertEqual((delta.height, reason), (241, "accepted"))
        rejected, _ = checked_additions(source.head(240), source.clear(), ref, DAY)
        self.assertEqual(rejected.height, 0)

    def test_conflict_replacement_requires_opt_in_and_preserves_other_stocks(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            mp = root / f"kline_minute/date={DAY}/part.parquet"
            dp = root / f"kline_daily/date={DAY}/part.parquet"
            mp.parent.mkdir(parents=True)
            dp.parent.mkdir(parents=True)
            corrupt = bars("000001").head(10).with_columns(pl.lit(9.).alias("close"))
            unrelated = bars("000002.SZ")
            old = pl.concat([corrupt, unrelated])
            old.write_parquet(mp)
            pl.DataFrame([reference()]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            source = SimpleNamespace(get_minute=lambda *a, **k: bars())
            with patch("app.services.minute_repair.kline_sync._resolve_minute_provider", return_value=(source, False, None)):
                rejected = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test")
                self.assertEqual(rejected["added_rows"], 0)
                self.assertTrue(pl.read_parquet(mp).equals(old))
                repaired = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test", repair_conflicts=True, task_id="explicit-replacement")
                repeated = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test", repair_conflicts=True, task_id="explicit-replacement")
            self.assertEqual(repaired["added_rows"], 230)
            self.assertEqual(repaired["replaced_rows"], 10)
            self.assertEqual(repaired["repaired_stock_days"], 1)
            self.assertEqual(repeated["replaced_rows"], 0)
            after = pl.read_parquet(mp)
            self.assertTrue(after.filter(pl.col("symbol") == "000002.SZ").select(unrelated.columns).equals(unrelated))
            self.assertTrue(after.filter(pl.col("symbol") == "000001.SZ").select(bars().columns).equals(bars()))
            backup = next((root / "import_backups").glob("*/date=*/part.parquet"))
            self.assertTrue(pl.read_parquet(backup).equals(old))

    def test_opt_in_does_not_replace_when_daily_reference_disagrees(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            mp = root / f"kline_minute/date={DAY}/part.parquet"
            dp = root / f"kline_daily/date={DAY}/part.parquet"
            mp.parent.mkdir(parents=True)
            dp.parent.mkdir(parents=True)
            old = bars().head(10).with_columns(pl.lit(9.).alias("close"))
            old.write_parquet(mp)
            pl.DataFrame([reference() | {"close": 11.}]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            source = SimpleNamespace(get_minute=lambda *a, **k: bars())
            with patch("app.services.minute_repair.kline_sync._resolve_minute_provider", return_value=(source, False, None)):
                result = repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test", repair_conflicts=True)
            self.assertEqual(result["replaced_rows"], 0)
            self.assertTrue(pl.read_parquet(mp).equals(old))

    def test_failed_readback_restores_original_partition(self):
        from app.services import kline_sync

        with TemporaryDirectory() as folder:
            root = Path(folder)
            mp = root / f"kline_minute/date={DAY}/part.parquet"
            dp = root / f"kline_daily/date={DAY}/part.parquet"
            mp.parent.mkdir(parents=True)
            dp.parent.mkdir(parents=True)
            old = bars().head(10).with_columns(pl.lit(9.).alias("close"))
            old.write_parquet(mp)
            pl.DataFrame([reference()]).write_parquet(dp)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            source = SimpleNamespace(get_minute=lambda *a, **k: bars())
            writer = kline_sync._atomic_write_parquet
            calls = []
            def faulty_writer(frame, path):
                calls.append(path)
                writer(frame.head(1) if len(calls) == 1 else frame, path)
            with (
                patch("app.services.minute_repair.kline_sync._resolve_minute_provider", return_value=(source, False, None)),
                patch("app.services.minute_repair.kline_sync._atomic_write_parquet", side_effect=faulty_writer),
                self.assertRaisesRegex(RuntimeError, "original partition restored"),
            ):
                repair_minute_gaps(repo, start_date=DAY, end_date=DAY, provider_name="test", repair_conflicts=True)
            self.assertTrue(pl.read_parquet(mp).equals(old))

    def test_day_limit_defers_older_gaps_without_reporting_complete(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for day in (DAY - timedelta(days=1), DAY):
                path = root / f"kline_daily/date={day}/part.parquet"
                path.parent.mkdir(parents=True)
                pl.DataFrame([reference()]).write_parquet(path)
            repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                                   rebuild_views=lambda: None, clear_cache=lambda: None)
            source = SimpleNamespace(get_minute=lambda *a, **k: bars())
            with patch("app.services.minute_repair.kline_sync._resolve_minute_provider", return_value=(source, False, None)):
                result = repair_minute_gaps(repo, start_date=DAY - timedelta(days=1), end_date=DAY,
                                            provider_name="test", max_missing_days=1)
            self.assertEqual(result["requests_planned"], 1)
            self.assertEqual(result["touched_dates"], [str(DAY)])
            self.assertEqual(result["reasons"]["deferred_stock_days"], 1)
            self.assertFalse(result["complete"])

    def test_tdx_historical_request_expands_past_800_bars(self):
        provider = TdxProvider()
        seen = []
        with (
            patch.object(provider, "_request", side_effect=lambda path, params: seen.append(params) or []),
            patch("app.market_time.cn_today", return_value=date(2026, 9, 4)),
        ):
            provider.get_minute(["000001.SZ"], datetime(2026, 8, 24), datetime(2026, 9, 3))
        self.assertTrue(seen)
        self.assertTrue(all(p["limit"] == 2880 for p in seen))

    def test_tdx_older_gap_fetch_is_bounded_but_not_truncated_at_5000(self):
        provider = TdxProvider()
        seen = []
        with (
            patch.object(provider, "_request", side_effect=lambda path, params: seen.append(params) or []),
            patch("app.market_time.cn_today", return_value=date(2026, 9, 6)),
        ):
            provider.get_minute(["920002.BJ"], datetime(2026, 5, 7), datetime(2026, 5, 8, 15))
        self.assertTrue(seen)
        self.assertTrue(all(p["limit"] == 24000 for p in seen))


if __name__ == "__main__":
    unittest.main()
