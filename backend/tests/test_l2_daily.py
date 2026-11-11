import io
from datetime import date

import polars as pl
import pytest

from app.data_providers.l2_archive import (
    CODE,
    DAY,
    QUANTITY,
    SIDE,
    TIME,
    TRADE_FLAG,
    TRADE_PRICE,
    csv_batches,
    member_symbol,
)
from app.services.l2_daily import QUOTE_FIELDS, DailyAccumulator


def test_concatenated_stream_is_batched_and_exact():
    header = f"{CODE},{DAY},{TIME}\n"
    raw = header + "000001.SZ,20260105,93000000\n" * 3
    raw += header + "600000.SH,20260105,093100000\n"
    frames = list(csv_batches(io.StringIO(raw), {"000001.SZ", "600000.SH"}, 2))
    assert [f.height for f in frames] == [2, 2]
    assert pl.concat(frames)[CODE].n_unique() == 2


@pytest.mark.parametrize(
    "raw",
    [
        f"{CODE},{DAY},{TIME}\n000001.SZ,20260105\n",
        f"{CODE},{DAY},{TIME}\n000002.SZ,20260105,093000000\n",
        f"{CODE},{DAY},{TIME}\n",
    ],
)
def test_malformed_stream_rejected(raw):
    with pytest.raises(ValueError):
        list(csv_batches(io.StringIO(raw), {"000001.SZ"}))


def test_duplicate_member_rejected():
    raw = f"{CODE},{DAY},{TIME}\n000001.SZ,20260105,093000000\n"
    with pytest.raises(ValueError, match="Duplicate"):
        list(csv_batches(io.StringIO(raw * 2), {"000001.SZ"}))


def test_archive_crc_failure_is_not_hidden_as_missing_members(tmp_path, monkeypatch):
    import subprocess
    import sys

    from app.data_providers import l2_archive

    original = subprocess.Popen
    def broken_reader(*args, **kwargs):
        return original([sys.executable, "-c", "import sys; sys.stderr.write('CRC error'); sys.exit(2)"], **kwargs)

    monkeypatch.setattr(l2_archive.subprocess, "Popen", broken_reader)
    member = "20260105/000001.SZ/\u9010\u7b14\u6210\u4ea4.csv"
    with pytest.raises(ValueError, match=r"reader failed \(2\): CRC error"):
        list(l2_archive.stream_archive(tmp_path / "test.7z", [(member, 123)], {"000001.SZ"}, "trade", {}))


def test_paths_are_not_extracted_or_guessed():
    assert member_symbol("20260105/600000.SH/quotes.csv") == "600000.SH"
    with pytest.raises(ValueError):
        member_symbol("../../600000.SH/quotes.csv")
    with pytest.raises(ValueError):
        member_symbol("600000.SH/000001.SZ/quotes.csv")


def test_archive_catalog_only_selects_supported_csvs(tmp_path, monkeypatch):
    import subprocess

    from app.data_providers import l2_archive

    member = "20250731/000001.SZ/\u884c\u60c5.csv"
    listing = "archive\n----------\n" + "\n\n".join(
        f"Path = {name}\nSize = 100\nAttributes = A"
        for name in ["20250731/run.bat", "20250731/D", member]
    )
    monkeypatch.setattr(l2_archive.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, listing.encode()))
    assert l2_archive.archive_catalog(tmp_path / "x.7z", "7zr") == [(member, 100)]
    bad = listing + "\n\nPath = ../000001.SZ/\u884c\u60c5.csv\nSize = 100\nAttributes = A"
    monkeypatch.setattr(l2_archive.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, bad.encode()))
    with pytest.raises(ValueError, match="Unsafe"):
        l2_archive.archive_catalog(tmp_path / "x.7z", "7zr")


def fixtures():
    symbol = "000001.SZ"
    trade = pl.DataFrame(
        {
            CODE: [symbol] * 2,
            DAY: ["20260105"] * 2,
            TIME: ["93000000", "143000000"],
            TRADE_PRICE: ["100000", "101000"],
            QUANTITY: ["100", "100"],
            SIDE: ["B", "S"],
            TRADE_FLAG: ["0", "0"],
        }
    )
    clocks = [
        f"{m // 60:02d}{m % 60:02d}00000" for m in [*range(570, 690, 5), *range(780, 900, 5), 900]
    ]
    rows = []
    for clock in clocks:
        row = {CODE: symbol, DAY: "20260105", TIME: clock}
        for name, value in zip(
            QUOTE_FIELDS.values(), [100000, 101000, 100000, 101000, 200, 2010], strict=True
        ):
            row[name] = str(value)
        for stem, px, qty in [("\u7533\u4e70", 99900, 20), ("\u7533\u5356", 101100, 10)]:
            for level in range(1, 11):
                row[stem + "\u4ef7" + str(level)] = str(px)
                row[stem + "\u91cf" + str(level)] = str(qty)
        rows.append(row)
    reference = dict(
        symbol=symbol, open=10.0, high=10.1, low=10.0, close=10.1, volume=2.0, amount=2010.0
    )
    return trade, pl.DataFrame(rows), reference


def aggregate(trade, quotes, reference, batch_rows=8192):
    accumulator = DailyAccumulator(date(2026, 1, 5))
    for frame in trade.iter_slices(batch_rows):
        accumulator.add_trades(frame)
    for frame in quotes.iter_slices(batch_rows):
        accumulator.add_quotes(frame)
    return accumulator.finish(reference["symbol"], reference)


def test_numerical_daily_features_are_batch_invariant():
    trade, quotes, reference = fixtures()
    row, reason = aggregate(trade, quotes, reference)
    assert reason == "validated"
    split_row, split_reason = aggregate(trade, quotes, reference, 1)
    assert split_reason == reason
    for key, value in row.items():
        assert split_row[key] == (pytest.approx(value) if isinstance(value, float) else value)
    assert row["l2_trade_volume"] == 200
    assert row["l2_trade_amount"] == 2010
    assert row["l2_buy_amount_ratio"] == pytest.approx(1000 / 2010)
    assert row["l2_net_buy_amount_ratio"] == pytest.approx(-10 / 2010)
    assert row["l2_depth_imbalance_mean"] == pytest.approx(1 / 3)
    assert row["l2_quote_coverage"] == 1
    assert row["available_at"].date() == date(2026, 1, 6)


@pytest.mark.parametrize("key,value", [("close", 99.0), ("volume", 200.0), ("amount", 20.1)])
def test_daily_unit_and_price_conflicts_rejected(key, value):
    trade, quotes, reference = fixtures()
    reference[key] = value
    assert aggregate(trade, quotes, reference)[0] is None


def test_daily_share_volume_requires_explicit_contract():
    trade, quotes, reference = fixtures()
    expected = aggregate(trade, quotes, reference)
    reference["volume"] *= 100
    acc = DailyAccumulator(date(2026, 1, 5))
    acc.add_trades(trade)
    acc.add_quotes(quotes)
    assert acc.finish(reference["symbol"], reference)[1] == "daily_turnover_mismatch"
    assert acc.finish(reference["symbol"], reference, daily_volume_unit="shares") == expected
    with pytest.raises(ValueError, match="daily_volume_unit"):
        acc.finish(reference["symbol"], reference, daily_volume_unit="auto")


def test_cancel_records_do_not_inflate_trade_features():
    trade, quotes, reference = fixtures()
    cancel = trade.head(1).with_columns(
        pl.lit("0").alias(TRADE_PRICE), pl.lit("C").alias(TRADE_FLAG)
    )
    incoming = pl.concat([cancel, trade])
    assert aggregate(incoming, quotes, reference) == aggregate(trade, quotes, reference)


def test_legacy_zero_event_padding_is_audited_not_a_trade():
    trade, quotes, reference = fixtures()
    padding = trade.head(1).with_columns(
        pl.lit("0").alias(DAY), pl.lit("92600000").alias(TIME),
        pl.lit("0").alias(TRADE_PRICE), pl.lit("0").alias(QUANTITY),
        pl.lit("\x00").alias(SIDE), pl.lit(" ").alias(TRADE_FLAG),
        pl.lit(" ").alias("\u59d4\u6258\u4ee3\u7801"),
        *[pl.lit("0").alias(c) for c in ["\u6210\u4ea4\u7f16\u53f7", "\u53eb\u5356\u5e8f\u53f7", "\u53eb\u4e70\u5e8f\u53f7"]],
    )
    acc = DailyAccumulator(date(2026, 1, 5))
    acc.add_trades(padding)
    acc.add_trades(trade)
    acc.add_quotes(quotes)
    assert acc.finish(reference["symbol"], reference) == aggregate(trade, quotes, reference)
    assert acc.excluded_padding_rows == {"000001.SZ": 1}
    assert acc.input_rows == 1 + trade.height + quotes.height
    for field, value in [(QUANTITY, "100"), (TIME, "93000000"), (SIDE, "B"), (DAY, "20260105")]:
        with pytest.raises(ValueError):
            DailyAccumulator(date(2026, 1, 5)).add_trades(padding.with_columns(pl.lit(value).alias(field)))


@pytest.mark.parametrize("clock", ["000000000", "120000000", "235959000", "096000000"])
def test_out_of_session_and_invalid_clocks_rejected(clock):
    trade, quotes, reference = fixtures()
    trade = trade.head(1).with_columns(pl.lit(clock).alias(TIME))
    with pytest.raises(ValueError):
        aggregate(trade, quotes, reference)


def test_unknown_event_flag_rejected():
    trade, quotes, reference = fixtures()
    trade = trade.with_columns(pl.lit("X").alias(TRADE_FLAG))
    with pytest.raises(ValueError):
        aggregate(trade, quotes, reference)


def test_sh_zero_flag_and_audited_close_offset_are_reconciled():
    trade, quotes, reference = fixtures()
    reference["symbol"] = "600000.SH"
    trade = trade.with_columns(
        pl.lit("600000.SH").alias(CODE), pl.Series(TIME, ["093000000", "150002670"])
    )
    quotes = quotes.with_columns(pl.lit("600000.SH").alias(CODE))
    assert aggregate(trade, quotes, reference)[1] == "validated"
    trade = trade.with_columns(pl.Series(TIME, ["150001100", "150002670"]))
    assert aggregate(trade, quotes, reference)[1] == "closing_auction_report_price_mismatch"


def test_nonfinite_snapshot_cannot_pass_reconciliation():
    trade, quotes, reference = fixtures()
    quotes = quotes.with_columns(pl.lit("NaN").alias(QUOTE_FIELDS["q_amount"]))
    assert aggregate(trade, quotes, reference)[1] == "trade_quote_turnover_mismatch"


def test_incomplete_quote_session_rejected():
    trade, quotes, reference = fixtures()
    assert aggregate(trade, quotes.tail(20), reference)[1] == "incomplete_quote_session"


def test_cross_batch_reversed_clock_and_identity_rejected():
    trade, quotes, reference = fixtures()
    with pytest.raises(ValueError, match="across batches"):
        aggregate(trade.reverse(), quotes, reference, 1)
    trade = trade.with_columns(pl.lit("20260106").alias(DAY))
    with pytest.raises(ValueError, match="date"):
        aggregate(trade, quotes, reference)


@pytest.mark.parametrize("reference_symbol", ["000001.SZ", "000001", "SZ000001"])
def test_extraction_resume_and_reference_change_are_fail_closed(tmp_path, monkeypatch, reference_symbol):
    from app.parquet import atomic_write_parquet
    from app.services import l2_daily

    trade, quotes, reference = fixtures()
    archive = tmp_path / "source.7z"
    archive.write_bytes(b"test catalog, not executable")
    daily = tmp_path / "kline_daily/date=2026-01-05/part.parquet"
    atomic_write_parquet(pl.DataFrame([{**reference, "symbol": reference_symbol}]), daily)
    monkeypatch.setattr(l2_daily, "archive_catalog", lambda *args: [])

    def stream(*args, **kwargs):
        yield trade if args[3] == "trade" else quotes

    monkeypatch.setattr(l2_daily, "stream_archive", stream)
    result = l2_daily.extract_day(tmp_path, archive, date(2026, 1, 5), [reference["symbol"]])
    assert result["accepted"] == 1
    assert l2_daily.extract_day(tmp_path, archive, date(2026, 1, 5), [reference["symbol"]])[
        "resumed"
    ]
    reference["amount"] = 0
    atomic_write_parquet(pl.DataFrame([reference]), daily)
    with pytest.raises(ValueError, match="differs"):
        l2_daily.extract_day(tmp_path, archive, date(2026, 1, 5), [reference["symbol"]])
    assert not (tmp_path / "l2_research/.2026-01-05.lock").exists()


def test_daily_reference_canonicalization_does_not_alias_indices():
    from app.services.l2_daily import canonical_equity_rows

    frame = pl.DataFrame({"symbol": ["000001", "000001.SH", "SH600000", "600000.SH", "920001.BJ"]})
    assert canonical_equity_rows(frame)["symbol"].to_list() == [
        "000001.SZ", "600000.SH", "600000.SH",
    ]


def test_duplicate_reference_aliases_fail_before_extraction(tmp_path, monkeypatch):
    from app.parquet import atomic_write_parquet
    from app.services import l2_daily

    _, _, reference = fixtures()
    archive = tmp_path / "source.7z"
    archive.write_bytes(b"test")
    daily = tmp_path / "kline_daily/date=2026-01-05/part.parquet"
    atomic_write_parquet(pl.DataFrame([reference, {**reference, "symbol": "000001"}]), daily)
    monkeypatch.setattr(l2_daily, "archive_catalog", lambda *args: pytest.fail("must not extract"))
    with pytest.raises(ValueError, match="Ambiguous daily references"):
        l2_daily.extract_day(tmp_path, archive, date(2026, 1, 5), ["000001.SZ"])


@pytest.mark.parametrize("duplicate", [False, True])
def test_universe_selection_handles_legacy_keys_without_duplicate_votes(tmp_path, duplicate):
    import runpy
    from pathlib import Path

    from app.parquet import atomic_write_parquet

    select = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/extract_l2_daily.py"))["select_universe"]
    rows = [{"symbol": s, "amount": 1000.0} for s in ["000001", "300001", "600000", "688001"]]
    if duplicate:
        rows.append({"symbol": "000001.SZ", "amount": 1000.0})
    for day in range(1, 21):
        atomic_write_parquet(pl.DataFrame(rows), tmp_path / f"kline_daily/date=2025-06-{day:02d}/part.parquet")
    if duplicate:
        with pytest.raises(ValueError, match="Ambiguous pre-sample"):
            select(tmp_path, date(2025, 7, 1), 4, min_history_days=18)
    else:
        result = select(tmp_path, date(2025, 7, 1), 4, min_history_days=18)
        assert result["symbols"] == ["000001.SZ", "300001.SZ", "600000.SH", "688001.SH"]
        assert result["min_history_days"] == 18


def test_universe_selection_requires_training_history_before_scanning_archives(tmp_path):
    import runpy
    from pathlib import Path

    from app.parquet import atomic_write_parquet

    select = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/extract_l2_daily.py"))["select_universe"]
    rows = [{"symbol": s, "amount": 1000.0} for s in ["000001", "300001", "600000", "688001"]]
    for day in range(1, 21):
        atomic_write_parquet(
            pl.DataFrame(rows),
            tmp_path / f"kline_daily/date=2025-06-{day:02d}/part.parquet",
        )
    with pytest.raises(ValueError, match="at least 60 pre-sample market dates"):
        select(tmp_path, date(2025, 7, 1), 4)


def test_json_atomic_replace_retries_without_truncating(tmp_path, monkeypatch):
    from pathlib import Path

    from app.services.l2_daily import write_json

    output = tmp_path / "summary.json"
    output.write_text('{"previous": true}')
    original = Path.replace
    attempts = []

    def replace(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            assert output.read_text() == '{"previous": true}'
            raise PermissionError("temporary Windows sharing violation")
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    write_json(output, {"complete": True})
    assert len(attempts) == 3
    assert '"complete": true' in output.read_text()
