"""Verified minute backfill and explicitly authorized conflict repair."""
from __future__ import annotations

import json
import shutil
import time as time_module
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time

import polars as pl

from app.market_time import cn_now, cn_today
from app.services import kline_sync
from app.services.security_identity import IdentityBatchError, exchange_for_code, parse_identity, validate_frame_identity

COLS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]
PRICE = ["open", "high", "low", "close"]

# QMT's local 1m files use the close-minute convention: the regular session
# normally has 120 morning bars and 119 afternoon bars, with 15:00 replacing
# the 14:59 bar.  Some files also contain the full 240-minute axis.  Do not
# keep retrying a validated QMT day merely because it is not exactly 240 rows.
QMT_CANONICAL_CLOCKS = frozenset(
    set(range(9 * 60 + 31, 11 * 60 + 31))
    | (set(range(13 * 60 + 1, 15 * 60 + 1)) - {14 * 60 + 59})
)
QMT_FULL_CLOCKS = QMT_CANONICAL_CLOCKS | {14 * 60 + 59}

# The Beijing Stock Exchange has a 15:00-15:30 after-hours fixed-price
# session. Its turnover is part of the daily bar, while the regular intraday
# feature window remains the normal 09:31-11:30 / 13:01-15:00 session.
BJ_CODE_PATTERN = r"^(43|83|87|88|92)"

# These sources use the canonical end-labelled session and may expose an
# additional 09:30 auction/opening snapshot. Tushare/Promax, DataHub and
# StockDB instead expose a traded 09:30 bar, so that row must participate in
# their daily OHLC validation.
OPENING_SNAPSHOT_SOURCES = frozenset({
    "qmt", "tdx", "local", "canonical", "stored", "staged_minute",
    "hf_public_snapshot",
})


def _source_name(source: str | None) -> str:
    return str(source or "").strip().lower()


def _has_explicit_daily_identity(code: str, reference: dict | None) -> bool:
    """Allow a verified daily row to bridge a missing lifecycle dimension.

    A historical daily partition may contain either a market-qualified symbol
    or a bare code.  A bare code is accepted only when the code prefix has a
    deterministic SH/SZ/BJ mapping; unknown codes are still isolated instead
    of being silently assigned to Shenzhen.
    """
    symbol = str((reference or {}).get("symbol") or "").strip().upper()
    expected_code = str(code).zfill(6)
    if not symbol or expected_code != symbol.split(".", 1)[0].zfill(6):
        return False
    expected = exchange_for_code(expected_code, "stock")
    if not expected:
        return False
    if "." not in symbol:
        return symbol == expected_code
    _, ref_exchange = symbol.rsplit(".", 1)
    return ref_exchange in {"SH", "SZ", "BJ"} and ref_exchange == expected


def _matches_identity_hint(code: str, identity_lookup: dict[str, str] | None) -> bool:
    """Validate an explicit market-qualified identity supplied by an audit."""
    symbol = str((identity_lookup or {}).get(str(code).zfill(6)) or "").strip().upper()
    if "." not in symbol:
        return False
    hint_code, hint_exchange = symbol.rsplit(".", 1)
    expected = exchange_for_code(code, "stock")
    return (
        hint_code == str(code).zfill(6)
        and hint_exchange in {"SH", "SZ", "BJ"}
        and bool(expected)
        and hint_exchange == expected
    )


def _bj_symbol_expr() -> pl.Expr:
    symbol = pl.col("symbol").cast(pl.String).str.to_uppercase()
    code = symbol.str.extract(r"(\d{6})", 1)
    return symbol.str.ends_with(".BJ") | code.str.contains(BJ_CODE_PATTERN)


def _minute_window(
    frame: pl.DataFrame,
    day: date,
    *,
    include_opening: bool = False,
    include_bj_after_hours: bool = False,
) -> pl.DataFrame:
    stamp = pl.col("datetime")
    clock = stamp.dt.hour().cast(pl.Int32) * 60 + stamp.dt.minute().cast(pl.Int32)
    clean = (
        (stamp.dt.date() == day)
        & (stamp.dt.second() == 0)
        & (stamp.dt.microsecond() == 0)
    )
    core = clock.is_between(570 if include_opening else 571, 690) | clock.is_between(781, 900)
    if include_bj_after_hours:
        core = core | (_bj_symbol_expr() & clock.is_between(901, 930))
    return frame.filter(clean & core)


def _has_extra_opening_snapshot(frame: pl.DataFrame, day: date) -> bool:
    """Infer the legacy local layout only when no source contract was given."""
    if frame.is_empty():
        return False
    stamp = pl.col("datetime")
    opening = frame.filter(
        (stamp.dt.date() == day)
        & (stamp.dt.hour() == 9)
        & (stamp.dt.minute() == 30)
    )
    return not opening.is_empty() and regular(frame, day)["datetime"].n_unique() >= 240


def _opening_is_snapshot(frame: pl.DataFrame, day: date, source: str | None) -> bool:
    name = _source_name(source)
    if name:
        return name in OPENING_SNAPSHOT_SOURCES
    return _has_extra_opening_snapshot(frame, day)


def _daily_aggregate_exprs(*, opening_is_snapshot: bool = False) -> list[pl.Expr]:
    # 09:30 is the start of continuous trading, so a traded row participates
    # in OHLC by default.  Only source contracts explicitly known to expose an
    # extra auction/opening snapshot exclude that row from OHLC.  Turnover is
    # retained in either case.  A zero-turnover 15:00 terminal row can still
    # carry the source's official close.
    stamp = pl.col("datetime")
    opening = (stamp.dt.hour() == 9) & (stamp.dt.minute() == 30)
    traded = (pl.col("volume") > 0) | (pl.col("amount") > 0)
    comparable_traded = traded & (~opening if opening_is_snapshot else pl.lit(True))
    return [
        pl.col("open").filter(comparable_traded).first(),
        pl.col("high").filter(comparable_traded).max(),
        pl.col("low").filter(comparable_traded).min(),
        pl.col("close").last(),
        pl.col("volume").sum(), pl.col("amount").sum(),
    ]


def keyed(frame: pl.DataFrame) -> pl.DataFrame:
    text = pl.col("symbol").cast(pl.String).str.to_uppercase()
    return frame.with_columns(
        pl.when(text.str.contains(r"^\d{1,5}$")).then(text.str.pad_start(6, "0"))
        .otherwise(text.str.extract(r"(\d{6})", 1)).alias("_code")
    )


def canonical_minute_aliases(frame: pl.DataFrame, day: date) -> pl.DataFrame:
    """Choose one stored symbol identity per stock-day without merging aliases."""
    if frame.is_empty() or "symbol" not in frame.columns:
        return frame
    had_code = "_code" in frame.columns
    tagged = frame if had_code else keyed(frame)
    symbol = pl.col("symbol").cast(pl.String).str.to_uppercase()
    stamp = pl.col("datetime")
    clock = stamp.dt.hour().cast(pl.Int32) * 60 + stamp.dt.minute().cast(pl.Int32)
    clean = (
        (stamp.dt.date() == day)
        & (stamp.dt.second() == 0)
        & (stamp.dt.microsecond() == 0)
    )
    session = clean & (clock.is_between(571, 690) | clock.is_between(781, 900))
    allowed = clean & (clock.is_between(570, 690) | clock.is_between(781, 900))
    choices = (
        tagged.filter(pl.col("_code").is_not_null())
        .group_by(["_code", "symbol"])
        .agg(
            pl.col("datetime").filter(session).n_unique().alias("_session_rows"),
            pl.col("datetime").filter(allowed).n_unique().alias("_allowed_rows"),
            symbol.str.contains(r"\.(SZ|SH|BJ)$").any().alias("_native"),
        )
        .sort(
            ["_code", "_session_rows", "_allowed_rows", "_native", "symbol"],
            descending=[False, True, True, True, False],
        )
        .unique("_code", keep="first", maintain_order=True)
        .select("_code", "symbol")
    )
    selected = tagged.join(choices, on=["_code", "symbol"], how="inner")
    return selected if had_code else selected.drop("_code")


def regular(
    frame: pl.DataFrame,
    day: date,
    *,
    include_opening: bool = False,
    include_bj_after_hours: bool = False,
) -> pl.DataFrame:
    return _minute_window(
        frame,
        day,
        include_opening=include_opening,
        include_bj_after_hours=include_bj_after_hours,
    )


def qmt_session_is_valid(frame: pl.DataFrame, day: date, reference: dict | None) -> bool:
    """Return whether an existing QMT day is already source-valid.

    QMT may omit zero-trade minutes for illiquid securities, so row count is
    not a completeness proof.  Reuse the same independent daily OHLCV checks
    as an import and treat a validated sparse day as terminal for this source.
    """
    if frame.is_empty() or reference is None:
        return False
    session = regular(frame.select(COLS), day)
    if session.is_empty() or session["datetime"].n_unique() != session.height:
        return False
    clocks = set(
        (hour * 60 + minute)
        for hour, minute in session.select(
            pl.col("datetime").dt.hour().alias("_hour"),
            pl.col("datetime").dt.minute().alias("_minute"),
        ).iter_rows()
    )
    if not clocks or not clocks.issubset(QMT_FULL_CLOCKS):
        return False
    _, reason = checked_additions(
        session, session.clear(), reference, day, source="qmt"
    )
    return reason == "accepted"


def daily_reference(frame: pl.DataFrame, *, data_dir=None) -> dict[str, dict]:
    frame = keyed(frame).filter(pl.col("_code").str.contains(r"^(00|30|60|68|43|83|87|88|92)"))
    if "quote_ts" in frame.columns:
        # Batch history has null quote_ts (the data_integrity contract). Choose
        # it before alias preference: a qualified close snapshot may be stale.
        authoritative = pl.col("quote_ts").is_null()
        frame = frame.filter(authoritative == authoritative.max().over("_code"))
    frame = frame.with_columns(pl.col("symbol").str.contains(r"\.(SZ|SH|BJ)$").alias("_native"))
    frame = frame.filter(pl.col("_native") == pl.col("_native").max().over("_code"))
    frame = frame.filter(pl.len().over("_code") == 1)
    frame = frame.filter(pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in COLS[2:]]))
    references = {r["_code"]: r for r in frame.iter_rows(named=True)}
    if data_dir is not None:
        from app.services.minute_reference import apply_reference_evidence
        references = apply_reference_evidence(references, data_dir)
    return references


def conflicting_complete_codes(
    frame: pl.DataFrame,
    references: dict[str, dict],
    day: date,
    *,
    source: str | None = "local",
) -> set[str]:
    """A full row count must not hide invalid prices or inconsistent units."""
    if frame.is_empty() or not references:
        return set()
    frame = keyed(canonical_minute_aliases(frame, day)).sort("datetime")
    frame = _minute_window(
        frame, day, include_opening=True, include_bj_after_hours=True
    )
    valid = pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in PRICE])
    valid &= (pl.col("high") >= pl.max_horizontal(PRICE)) & (pl.col("low") <= pl.min_horizontal(PRICE))
    valid &= pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) >= 0) for c in ["volume", "amount"]])
    opening_snapshot = _opening_is_snapshot(frame, day, source)
    aggregate = frame.group_by("_code").agg(
        *_daily_aggregate_exprs(opening_is_snapshot=opening_snapshot),
        valid.fill_null(False).all().alias("_valid"),
        (pl.len() == pl.col("datetime").n_unique()).alias("_unique"),
    )
    counts = regular(frame, day).group_by("_code").agg(pl.col("datetime").n_unique().alias("_count"))
    refs = pl.DataFrame([{c: row[c] for c in ["_code", *COLS[2:]]} for row in references.values()])
    joined = aggregate.join(counts, on="_code").filter(pl.col("_count") >= 240).join(refs, on="_code", suffix="_daily")
    price_tolerance = pl.max_horizontal(pl.lit(0.011), pl.col("close_daily").abs() * 1e-6)
    if opening_snapshot:
        price_match = (
            (pl.col("close") - pl.col("close_daily")).abs() <= price_tolerance
        ) & (
            pl.col("high") <= pl.col("high_daily") + price_tolerance
        ) & (
            pl.col("low") >= pl.col("low_daily") - price_tolerance
        ) & pl.col("open").is_between(
            pl.col("low_daily") - price_tolerance,
            pl.col("high_daily") + price_tolerance,
        )
    else:
        price_match = pl.all_horizontal([
            (pl.col(c) - pl.col(f"{c}_daily")).abs()
            <= pl.max_horizontal(pl.lit(0.011), pl.col(f"{c}_daily").abs() * 1e-6)
            for c in PRICE
        ])
    volume_match = pl.any_horizontal([
        (pl.col("volume") * scale - pl.col("volume_daily")).abs()
        <= pl.max_horizontal(pl.lit(1.0), pl.col("volume_daily") * 0.002)
        for scale in (0.01, 1, 100)
    ])
    amount_match = (pl.col("amount") - pl.col("amount_daily")).abs() <= pl.max_horizontal(pl.lit(1.0), pl.col("amount_daily") * 0.002)
    tolerance = pl.max_horizontal(pl.lit(0.02), pl.col("high") * 0.002)
    units_match = (pl.col("volume") > 0) & pl.any_horizontal([
        (pl.col("amount") / (pl.col("volume") * unit)).is_between(
            pl.col("low") - tolerance, pl.col("high") + tolerance,
        )
        for unit in (1.0, 100.0)
    ])
    valid = pl.col("_valid") & pl.col("_unique") & price_match & volume_match & amount_match & units_match
    return set(joined.filter(~valid.fill_null(False))["_code"])


def checked_additions(
    incoming: pl.DataFrame,
    existing: pl.DataFrame,
    reference: dict | None,
    day: date,
    *,
    source: str | None = None,
):
    """Return missing keys only; never change values on an existing key."""
    if incoming.is_empty():
        return incoming, "source_empty"
    incoming = regular(
        incoming.select(COLS), day, include_opening=True, include_bj_after_hours=True
    ).sort("datetime")
    session = regular(incoming, day)
    source_name = _source_name(source)
    is_qmt = source_name == "qmt"
    opening_snapshot = _opening_is_snapshot(incoming, day, source)
    unique_rows = incoming["datetime"].n_unique() == incoming.height
    # QMT stores only minutes with an effective trade for some illiquid BSE
    # securities.  Such a day can legitimately contain fewer than 240 rows.
    # Its daily OHLC, volume and amount still have to pass the independent
    # reference checks below, which rejects a truncated morning/afternoon.
    complete_session = (
        0 < session.height <= 240
        if is_qmt
        else session.height == 240 and session["datetime"].n_unique() == 240
    )
    if not complete_session or not unique_rows:
        return incoming.clear(), "source_incomplete"
    good = pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in PRICE])
    coherent_bar = (pl.col("high") >= pl.max_horizontal(PRICE)) & (pl.col("low") <= pl.min_horizontal(PRICE))
    if is_qmt:
        # Some sparse QMT DAT exports omit zero-turnover bars and encode the
        # auction/open snapshot in the first effective trade minute. Its open
        # can precede that minute's reported low. Permit only the first stored
        # row; the independent daily OHLC/turnover gate below remains strict.
        first_qmt_open = (
            (pl.col("datetime") == pl.col("datetime").min())
            & (pl.col("high") >= pl.col("low"))
            & pl.col("close").is_between(pl.col("low"), pl.col("high"))
        )
        good &= coherent_bar | first_qmt_open
    else:
        good &= coherent_bar
    good &= pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) >= 0) for c in ["volume", "amount"]])
    if not incoming.select(good.fill_null(False).all()).item():
        return incoming.clear(), "source_invalid"
    if reference is None:
        return incoming.clear(), "daily_reference_missing"
    aggregate = incoming.select(
        _daily_aggregate_exprs(opening_is_snapshot=opening_snapshot)
    ).row(0, named=True)
    if aggregate["volume"] <= 0:
        return incoming.clear(), "minute_turnover_unit_conflict"
    # QMT exposes minute volume in lots while its minute amount includes the
    # auction/odd-lot rounding used by the client.  The resulting aggregate
    # VWAP can sit slightly outside the OHLC range even when the price and
    # amount are correct.  Keep this sanity gate for remote sources, but use
    # the explicit QMT volume-unit check below for local QMT data.
    if not is_qmt:
        tolerance = max(0.02, aggregate["high"] * 0.002)
        vwaps = (
            aggregate["amount"] / aggregate["volume"],
            aggregate["amount"] / (aggregate["volume"] * 100.0),
        )
        if not any(
            aggregate["low"] - tolerance <= vwap <= aggregate["high"] + tolerance
            for vwap in vwaps
        ):
            return incoming.clear(), "minute_turnover_unit_conflict"
    price_tolerance = max(0.011, abs(reference["close"]) * 1e-6)
    if opening_snapshot:
        # End-labelled QMT/TDX and canonical local data may omit the 09:25
        # auction from comparable minute OHLC. Their regular-session range must
        # stay inside the daily range and their terminal close must match.
        regular_price_ok = (
            abs(aggregate["close"] - reference["close"]) <= price_tolerance
            and aggregate["high"] <= reference["high"] + price_tolerance
            and aggregate["low"] >= reference["low"] - price_tolerance
            and reference["low"] - price_tolerance
            <= aggregate["open"]
            <= reference["high"] + price_tolerance
        )
    else:
        regular_price_ok = all(
            abs(aggregate[c] - reference[c]) <= max(0.011, abs(reference[c]) * 1e-6)
            for c in PRICE
        )
    if not regular_price_ok:
        return incoming.clear(), "daily_price_conflict"
    # QMT minute bars start at 09:31 while the daily volume commonly includes
    # the opening auction. Keep price/amount as hard gates, but allow the
    # source-specific volume gap caused by that auction. Other sources retain
    # the strict 0.2% contract.
    volume_tolerance = 0.08 if is_qmt else 0.002
    volume_matches = any(abs(aggregate["volume"] * scale - reference["volume"]) <= max(1.0, reference["volume"] * volume_tolerance) for scale in (0.01, 1, 100))
    amount_tolerance = 0.005 if is_qmt else 0.002
    if not volume_matches or abs(aggregate["amount"] - reference["amount"]) > max(1.0, reference["amount"] * amount_tolerance):
        return incoming.clear(), "daily_turnover_conflict"
    if not existing.is_empty():
        if existing["symbol"].n_unique() != 1:
            return incoming.clear(), "existing_symbol_alias_conflict"
        overlap = incoming.join(existing, on="datetime", suffix="_old")
        for c in COLS[2:]:
            absolute, relative = (0.001, 1e-6) if c in PRICE else (1.0, 0.002)
            valid = ((pl.col(c) - pl.col(c + "_old")).abs() <= pl.max_horizontal(pl.lit(absolute), pl.col(c + "_old").abs() * relative)).fill_null(False)
            if not overlap.select(valid.all()).item():
                return incoming.clear(), "existing_values_conflict"
        incoming = incoming.with_columns(pl.lit(existing["symbol"][0]).alias("symbol"))
        incoming = incoming.join(existing.select("datetime"), on="datetime", how="anti")
        # Small tolerable differences on individual old bars can accumulate
        # into an invalid full day; validate what will actually be persisted.
        merged = pl.concat([existing.select(COLS), incoming]).sort("datetime")
        _, merged_reason = checked_additions(merged, merged.clear(), reference, day, source=source)
        if merged_reason != "accepted":
            return incoming.clear(), "existing_values_conflict"
    return incoming, "accepted" if incoming.height else "already_complete"


def repair_minute_gaps(repo, *, start_date: date, end_date: date, provider_name: str, progress=None, workers: int = 2, symbols: list[str] | None = None, max_missing_days: int | None = None, newest_first: bool = True, repair_conflicts: bool = False, stock_days: list[tuple[str, date]] | None = None, max_requests: int | None = None, single_day_requests: bool = False, source_provider=None, bypass_failover: bool = False, identity_lookup: dict[str, str] | None = None, task_id: str | None = None) -> dict:
    # A same-day after-close repair is valid once the final daily reference is
    # published.  Before 15:10 it is still an intraday stream and must use the
    # live tail path instead of pretending the session is complete.
    wall_now = cn_now().replace(tzinfo=None)
    if end_date > cn_today() or (end_date == cn_today()
                                 and wall_now.time() < time(15, 10)):
        raise ValueError("Verified minute repair only accepts closed trading dates")
    provider = source_provider
    if provider is None:
        provider, fallback, error = kline_sync._resolve_minute_provider(provider_name)
        if fallback or provider is None:
            raise ValueError(error or "Configured minute provider unavailable")
    root = repo.store.data_dir
    report_dir = root / "import_reports" / ("online_minute_" + uuid.uuid4().hex[:12])
    report_dir.mkdir(parents=True, exist_ok=False)
    minute_dir = root / "kline_minute"
    references, plans = {}, defaultdict(list)
    reasons = Counter()
    details = []
    from app.services.security_lifecycle import load_lifecycle_index, validate_security_lifecycle
    lifecycle_index = load_lifecycle_index(root)
    allowed = None if symbols is None else set(keyed(pl.DataFrame({"symbol": symbols}, schema={"symbol": pl.String}))["_code"].drop_nulls())
    selected_pairs = None
    requested_symbol_by_code_day: dict[tuple[str, date], str] = {}
    if stock_days is not None:
        selected_pairs = {(symbol.split(".")[0], day) for symbol, day in stock_days}
        requested_symbol_by_code_day = {
            (symbol.split(".")[0], day): symbol
            for symbol, day in stock_days
            if "." in symbol
        }
    dates = sorted(set(kline_sync._partition_dates(root / "kline_daily")) | set(kline_sync._partition_dates(minute_dir)))
    dates = [d for d in dates if start_date <= d <= end_date]
    if selected_pairs is not None:
        dates = [d for d in dates if d in {day for _, day in selected_pairs}]
    for day in dates:
        dp = root / "kline_daily" / f"date={day}" / "part.parquet"
        if not dp.exists():
            reasons["daily_partition_missing"] += 1
            continue
        refs = daily_reference(pl.read_parquet(dp), data_dir=root)
        references[day] = refs
        mp = minute_dir / f"date={day}" / "part.parquet"
        counts = {}
        conflicts = set()
        local = pl.DataFrame()
        if mp.exists():
            # QMT needs the value columns even when conflict replacement is
            # disabled: a validated sparse day is terminal for QMT and should
            # not be queued again solely because it has fewer than 240 rows.
            columns = COLS if repair_conflicts or provider_name == "qmt" else ["symbol", "datetime"]
            if selected_pairs is not None:
                # An explicit durable stock-day queue must not materialize the
                # whole market partition merely to inspect a few target rows.
                # Keep the filter in the lazy parquet scan so only target
                # symbols reach memory.
                day_codes = [code for code, pair_day in selected_pairs if pair_day == day]
                symbol_code = pl.col("symbol").cast(pl.String).str.extract(r"(\d{6})", 1)
                stored = (
                    pl.scan_parquet(mp)
                    .select(columns)
                    .filter(symbol_code.is_in(day_codes))
                    .collect()
                )
            else:
                stored = pl.read_parquet(mp, columns=columns)
            local = keyed(regular(stored, day))
            counts = dict(local.group_by("_code").agg(pl.col("datetime").n_unique()).iter_rows())
            if repair_conflicts:
                conflicts = conflicting_complete_codes(stored, refs, day)
        qmt_valid_sparse = set()
        if provider_name == "qmt" and not local.is_empty() and set(COLS).issubset(local.columns):
            for code in counts:
                if code in conflicts or counts.get(code, 0) >= 240:
                    continue
                current = local.filter(pl.col("_code") == code).drop("_code")
                if qmt_session_is_valid(current, day, refs.get(code)):
                    qmt_valid_sparse.add(code)
        for code in refs:
            if selected_pairs is not None and (code, day) not in selected_pairs:
                continue
            if allowed is not None and code not in allowed:
                continue
            if lifecycle_index:
                lifecycle = validate_security_lifecycle(
                    code, day, root, index=lifecycle_index,
                )
                if not lifecycle.ok:
                    # A live-only lifecycle dimension can omit an otherwise
                    # valid historical symbol. An exact market-qualified
                    # daily row is sufficient identity evidence for this
                    # stock-day; it is never a substitute for a market or
                    # lifecycle conflict.
                    if (
                        lifecycle.reason == "identity_history_unknown"
                        and (
                            _has_explicit_daily_identity(code, refs.get(code))
                            or _matches_identity_hint(code, identity_lookup)
                        )
                    ):
                        reasons["lifecycle_daily_reference_fallback"] += 1
                    else:
                        reasons[f"lifecycle_{lifecycle.reason}"] += 1
                        details.append({
                            "date": str(day), "symbol": code,
                            "reason": f"lifecycle_{lifecycle.reason}",
                            "lifecycle": lifecycle.as_dict(),
                        })
                        continue
            if (counts.get(code, 0) >= 240 or code in qmt_valid_sparse) and code not in conflicts:
                continue
            plans[code].append(day)
    if max_missing_days is not None:
        selected = set(sorted({d for ds in plans.values() for d in ds}, reverse=newest_first)[:max(1, max_missing_days)])
        reasons["deferred_stock_days"] = sum(d not in selected for ds in plans.values() for d in ds)
        plans = {code: [d for d in ds if d in selected] for code, ds in plans.items()}
        plans = {code: ds for code, ds in plans.items() if ds}
    work = []
    for code, ds in sorted(plans.items()):
        # TDX pages recent bars once per stock; historical vendors receive
        # short contiguous windows rather than a repeated two-year request.
        # TDX and QMT are local stores. Fetch each stock's requested span once;
        # splitting it into remote-vendor-sized windows multiplies disk scans.
        local_history_provider = provider_name in {"tdx", "qmt"}
        groups = [ds] if local_history_provider else [[ds[0]]]
        if not local_history_provider:
            for day in ds[1:]:
                same_identity = (
                    requested_symbol_by_code_day.get((code, day), code)
                    == requested_symbol_by_code_day.get((code, groups[-1][-1]), code)
                )
                if (
                    not single_day_requests
                    and same_identity
                    and (day - groups[-1][-1]).days <= 7
                    and (day - groups[-1][0]).days <= 25
                ):
                    groups[-1].append(day)
                else:
                    groups.append([day])
        work.extend((code, days) for days in groups)
    if max_requests is not None:
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        reasons["deferred_request_stock_days"] += sum(len(ds) for _, ds in work[max_requests:])
        work = work[:max_requests]
    planned_days = sum(len(ds) for _, ds in work)
    if progress:
        progress(0, max(1, len(work)), f"{provider_name}: {len(work)} requests, {planned_days} stock-days")

    def fetch(item):
        code, days = item
        try:
            request_symbol = requested_symbol_by_code_day.get((code, days[0]), code)
            args = ([request_symbol], datetime.combine(min(days), time(9, 30)), datetime.combine(max(days), time(15)))
            diagnostic = getattr(provider, "get_minute_diagnostics", None)
            from app.services import minute_failover
            if not bypass_failover and minute_failover.enabled(provider_name, provider):
                df, issues = minute_failover.fetch_minutes(
                    provider, *args, freq="1m", data_dir=root, allow_replacement=repair_conflicts,
                )
            elif callable(diagnostic):
                df, issues = diagnostic(*args, freq="1m")
            else:
                df, issues = provider.get_minute(*args, freq="1m"), []
            if not df.is_empty():
                expected_exchange = exchange_for_code(code, "stock")
                expected_symbol = f"{code}.{expected_exchange}" if expected_exchange else code
                try:
                    validate_frame_identity(
                        df,
                        asset_type="stock",
                        lookup={},
                        expected_symbols=[expected_symbol],
                        require_lookup=False,
                    )
                except IdentityBatchError:
                    return item, pl.DataFrame(), [{"reason": "source_identity_conflict", "retryable": False}]
                selected = keyed(df).filter(pl.col("_code") == code).drop("_code")
                if selected.height != df.height:
                    return item, pl.DataFrame(), [{"reason": "source_symbol_conflict", "retryable": False}]
            return item, df, issues
        except Exception as exc:
            from app.plugins.tushare.provider import minute_error_details
            return item, pl.DataFrame(), [minute_error_details(exc)]

    backups, touched = set(), set()
    pending = defaultdict(list)
    added = checked = replaced = removed = repaired_days = 0
    duplicate_rows = updated_rows = rejected_rows = 0

    # A current instrument dimension is intentionally live-only in some
    # deployments. Historical daily bars, however, already carry a verified
    # native exchange suffix. Use that historical identity for this repair
    # batch, while still rejecting code/exchange mismatches at write time.
    historical_lookup: dict[str, str] = {}
    ambiguous_codes: set[str] = set()
    for day_refs in references.values():
        for code, row in day_refs.items():
            symbol = str(row.get("symbol") or "")
            exchange = exchange_for_code(code, "stock")
            if symbol.upper().endswith((".SH", ".SZ", ".BJ")):
                exchange = symbol.rsplit(".", 1)[-1].upper()
            if not exchange:
                continue
            canonical = f"{code}.{exchange}"
            previous = historical_lookup.get(code)
            if previous and previous != canonical:
                ambiguous_codes.add(code)
            else:
                historical_lookup[code] = canonical
    for code in ambiguous_codes:
        historical_lookup.pop(code, None)
    write_identity_lookup = identity_lookup or historical_lookup or None

    from app.services import minute_repair_authorization as authorization
    authorization_task = task_id or authorization.scope_id(
        start_date, end_date, symbols=symbols, stock_days=stock_days,
    )
    first_authorization_scope = authorization.open_scope(
        root, authorization_task, provider_name,
    )
    planned_authorization: dict[tuple[str, date], str] = {}
    approval_required: list[dict] = []
    identity_pending: list[dict] = []
    authorization_reused = 0
    planned_pairs = []
    for code, ds in plans.items():
        for day in ds:
            symbol = requested_symbol_by_code_day.get((code, day)) or historical_lookup.get(code)
            if symbol:
                try:
                    identity = parse_identity(symbol, asset_type="stock", lookup=write_identity_lookup, require_lookup=True)
                    planned_pairs.append((authorization.canonical(identity.symbol), day))
                except ValueError:
                    # Identity validation remains authoritative; no replacement
                    # authorization is created for an unqualified/ambiguous row.
                    continue
    existing_grants = authorization.grants(root, authorization_task)
    if repair_conflicts and planned_pairs and first_authorization_scope and not existing_grants:
        # The explicit repair_conflicts flag is the user's authorization for
        # this first exact task scope. Persist it so retries do not ask again.
        authorization.grant(root, authorization_task, provider_name, planned_pairs,
                            actor="explicit_task_request")
        existing_grants = authorization.grants(root, authorization_task)
    for symbol, day in planned_pairs:
        grant_id = next((row["id"] for row in existing_grants
                         if row["source"] == provider_name and row["operation"] == authorization.OPERATION
                         and row["symbol"] == symbol and row["trade_date"] == str(day)), None)
        if grant_id:
            planned_authorization[(symbol.split(".")[0], day)] = grant_id
            authorization_reused += 1

    def flush():
        nonlocal added, replaced, removed, repaired_days
        nonlocal duplicate_rows, updated_rows, rejected_rows
        for day, frames in pending.items():
            incoming = pl.concat(frames, how="diagonal_relaxed")
            path = minute_dir / f"date={day}" / "part.parquet"
            with repo._write_lock:
                old = pl.read_parquet(path) if path.exists() else incoming.clear()
                old_by = keyed(old)
                additions = []
                replacements = []
                replacement_authorizations = []
                replacement_codes = []
                replacement_added = replacement_removed = replacement_old = 0
                for block in keyed(incoming).partition_by("_code"):
                    code = block["_code"][0]
                    current = old_by.filter(pl.col("_code") == code).drop("_code")
                    source = block.drop("_code")
                    # Legacy partitions and providers may use bare codes.
                    # Resolve both sides against the verified repair identity
                    # before checking overlaps or recording authorization.
                    current = kline_sync._ensure_minute_identity_columns(
                        current, asset_type="stock", identity_lookup=write_identity_lookup,
                        allow_existing_identity=True,
                    )
                    source = kline_sync._ensure_minute_identity_columns(
                        source, asset_type="stock", identity_lookup=write_identity_lookup,
                    )
                    accepted, reason = checked_additions(
                        source, current, references[day].get(code), day,
                        source=provider_name,
                    )
                    overlap_rows = source.join(current.select("datetime"), on="datetime", how="semi").height
                    grant_id = planned_authorization.get((code, day))
                    if reason in {"accepted", "already_complete"}:
                        candidate = current["symbol"][0] if current.height else source["symbol"][0]
                        try:
                            qualified = authorization.canonical(candidate)
                        except ValueError:
                            # A successful-looking frame must never turn an
                            # unknown identity into an automatic write.
                            identity_pending.append({
                                "task_id": authorization_task, "source": provider_name,
                                "symbol": code, "date": str(day),
                                "reason": "identity_pending",
                            })
                            reasons["identity_pending"] += 1
                            rejected_rows += source.height
                            details.append({"date": str(day), "symbol": code,
                                            "reason": "identity_pending", "added_rows": 0,
                                            "rejected_rows": source.height})
                            continue
                        authorization.resolve(root, authorization_task, provider_name,
                                              qualified, day)
                    if reason == "existing_values_conflict" and grant_id:
                        # Independently validate the entire replacement against daily
                        # OHLC/turnover; old disagreements never lower the threshold.
                        verified, validation = checked_additions(
                            source, source.clear(), references[day].get(code), day,
                            source=provider_name,
                        )
                        if validation == "accepted" and current["symbol"].n_unique() == 1:
                            verified = verified.with_columns(pl.lit(current["symbol"][0]).alias("symbol"))
                            new_keys = verified.join(current.select("datetime"), on="datetime", how="anti").height
                            old_keys = current.join(verified.select("datetime"), on="datetime", how="anti").height
                            replacements.append(verified)
                            replacement_codes.append(code)
                            replacement_added += new_keys
                            replacement_removed += old_keys
                            replacement_old += current.height
                            updated_rows += current.height
                            reasons["repaired_existing_conflict"] += 1
                            details.append({"date": str(day), "symbol": code, "reason": "repaired_existing_conflict", "added_rows": new_keys, "replaced_rows": current.height, "removed_rows": old_keys})
                            replacement_authorizations.append(
                                (grant_id, str(current["symbol"][0]), day)
                            )
                            continue
                    if reason == "existing_values_conflict":
                        candidate = current["symbol"][0] if current.height else source["symbol"][0]
                        try:
                            request_id = authorization.request_replacement(
                                root, authorization_task, provider_name, str(candidate), day,
                            )
                        except ValueError:
                            # Identity-unknown conflicts belong to validation,
                            # never to the replacement confirmation flow.
                            identity_pending.append({
                                "task_id": authorization_task, "source": provider_name,
                                "symbol": code, "date": str(day),
                                "reason": "identity_pending",
                            })
                            reasons["identity_pending"] += 1
                            rejected_rows += source.height
                            details.append({"date": str(day), "symbol": code,
                                            "reason": "identity_pending", "added_rows": 0,
                                            "rejected_rows": source.height})
                            continue
                        approval_required.append({
                            "id": request_id, "task_id": authorization_task,
                            "source": provider_name, "symbol": code,
                            "qualified_symbol": str(candidate), "date": str(day),
                            "reason": "existing_values_conflict",
                        })
                    reasons[reason] += 1
                    if reason == "already_complete":
                        duplicate_rows += source.height
                    elif reason != "accepted":
                        rejected_rows += source.height
                    elif overlap_rows:
                        duplicate_rows += overlap_rows
                    details.append({"date": str(day), "symbol": code, "reason": reason,
                                    "added_rows": accepted.height, "duplicate_rows": overlap_rows,
                                    "rejected_rows": source.height if reason not in {"accepted", "already_complete"} else 0})
                    if not accepted.is_empty():
                        additions.append(accepted)
                if not additions and not replacements:
                    continue
                delta = pl.concat(additions + replacements, how="diagonal_relaxed")
                delta = kline_sync._ensure_minute_identity_columns(
                    delta,
                    asset_type="stock",
                    identity_lookup=write_identity_lookup,
                )
                untouched = old_by.filter(~pl.col("_code").is_in(replacement_codes)).drop("_code")
                untouched = kline_sync._ensure_minute_identity_columns(
                    untouched,
                    asset_type="stock",
                    identity_lookup=write_identity_lookup,
                    allow_existing_identity=True,
                )
                expected = pl.concat([untouched, delta], how="diagonal_relaxed").sort(["symbol", "datetime"])
                if path.exists() and day not in backups:
                    backup = root / "import_backups" / report_dir.name / f"date={day}" / "part.parquet"
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup)
                    backups.add(day)
                if replacements:
                    kline_sync._atomic_write_parquet(expected, path)
                else:
                    kline_sync._write_minute_partition(
                        delta,
                        minute_dir,
                        asset_type="stock",
                        identity_lookup=write_identity_lookup,
                    )
                after = pl.read_parquet(path)
                if not after.sort(["symbol", "datetime"]).equals(expected):
                    # Restore this transaction's original partition on failed verification.
                    kline_sync._atomic_write_parquet(old, path)
                    raise RuntimeError(f"Readback verification failed; original partition restored: {day}")
                if replacement_authorizations:
                    backup = root / "import_backups" / report_dir.name / f"date={day}" / "part.parquet"
                    for grant_id, symbol, grant_day in replacement_authorizations:
                        authorization.used(
                            root, grant_id, symbol, grant_day,
                            backup=str(backup.relative_to(root)),
                        )
                added += sum(frame.height for frame in additions) + replacement_added
                replaced += replacement_old
                removed += replacement_removed
                repaired_days += len(replacements)
                touched.add(day)
        pending.clear()

    retryable_reasons = {
        "source_pending", "rate_limited", "upstream_unavailable", "request_timeout",
        "transport_error", "backup_unavailable", "not_requested",
    }
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(16, int(workers)))) as pool:
            # HTTP 202 is per-request preparation, not a source-wide outage.
            # Queue each request once, then retry only the pending subset after
            # the vendor's bounded wait. This keeps thousands of stock-days
            # moving without parking one worker per pending symbol.
            round_work = list(work)
            for attempt in range(4):
                retry_work = []
                retry_after = 0
                empty_waves = 0
                for offset in range(0, len(round_work), 40):
                    before_written = added + repaired_days
                    before_verified = (
                        reasons["accepted"]
                        + reasons["already_complete"]
                        + reasons["repaired_existing_conflict"]
                    )
                    wave_retry_count = 0
                    futures = [pool.submit(fetch, item) for item in round_work[offset:offset + 40]]
                    for future in as_completed(futures):
                        item, frame, issues = future.result()
                        code, ds = item
                        retry_item = False
                        for day in ds:
                            part = regular(frame, day, include_opening=True) if not frame.is_empty() else frame
                            if part.is_empty():
                                matched = [r for r in issues if r.get("date") == str(day)]
                                if not matched:
                                    matched = [r for r in issues if not r.get("date")]
                                issue = next((r for r in reversed(matched) if r.get("reason") != "accepted"),
                                             {"reason": "source_empty", "retryable": False})
                                reason = issue["reason"]
                                if (attempt < 3 and issue.get("retryable")
                                        and reason in retryable_reasons):
                                    retry_item = True
                                    retry_after = max(retry_after, int(issue.get("retry_after_seconds") or 30))
                                    continue
                                reasons[reason] += 1
                                rejected_rows += part.height
                                details.append({**{k: issue[k] for k in (
                                    "retryable", "http_status", "retry_after_seconds", "evidence", "source_symbol",
                                ) if k in issue}, "date": str(day), "symbol": code,
                                    "reason": reason, "added_rows": 0})
                            else:
                                # Reject a provider accidentally returning another ticker.
                                part = keyed(part).filter(pl.col("_code") == code).drop("_code")
                                pending[day].append(part)
                        if retry_item:
                            retry_work.append(item)
                            wave_retry_count += 1
                        checked += 1
                        if progress:
                            progress(
                                min(checked, len(work)), len(work),
                                f"{provider_name}: round {attempt + 1}, checked {checked}, "
                                f"pending {len(retry_work)}, added {added} rows, repaired {repaired_days} stock-days",
                            )
                    flush()
                    verified_progress = (
                        reasons["accepted"]
                        + reasons["already_complete"]
                        + reasons["repaired_existing_conflict"]
                    ) > before_verified
                    if added + repaired_days == before_written:
                        empty_waves = 0 if wave_retry_count or verified_progress else empty_waves + 1
                    else:
                        empty_waves = 0
                    # A validated QMT sparse day commonly has no new keys: the
                    # local partition already contains every effective trade
                    # minute.  Treat ``already_complete`` as successful
                    # verification instead of aborting the whole repair wave.
                    verified_noop = (
                        reasons["accepted"]
                        + reasons["already_complete"]
                        + reasons["repaired_existing_conflict"]
                    )
                    if (not bypass_failover and offset == 0 and len(round_work) > 40 and not added and not repaired_days
                            and not wave_retry_count and verified_noop == 0):
                        raise RuntimeError("First 40 requests yielded no verifiable additions; stopped to avoid pointless bulk requests")
                    if (not bypass_failover and empty_waves >= 2
                            and checked < len(work) and not retry_work):
                        raise RuntimeError("Two request waves yielded no verifiable additions; stopped for source review")
                if not retry_work:
                    break
                if progress:
                    progress(
                        min(checked, len(work)), len(work),
                        f"{provider_name}: waiting to retry {len(retry_work)} pending requests (round {attempt + 2}/4)",
                    )
                time_module.sleep(min(max(retry_after, 2), 120))
                round_work = retry_work
    finally:
        report = {"provider": provider_name, "start": str(start_date), "end": str(end_date),
                  "requests_planned": len(work), "requests_checked": checked,
                  "stock_days_planned": planned_days, "added_rows": added,
                  "repair_conflicts": repair_conflicts, "replaced_rows": replaced,
                  "removed_rows": removed, "updated_rows": updated_rows,
                  "duplicate_rows": duplicate_rows, "rejected_rows": rejected_rows,
                  "repaired_stock_days": repaired_days,
                  "approval_required": approval_required,
                  "identity_pending": identity_pending,
                  "authorization_task_id": authorization_task,
                  "authorization_reused": authorization_reused,
                  "touched_dates": sorted(str(d) for d in touched), "reasons": dict(reasons), "details": details}
        (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if touched:
            repo.rebuild_views()
            repo.clear_cache()
    return {k: v for k, v in report.items() if k != "details"} | {
        "written_rows": added, "inserted_rows": added,
        "updated_rows": updated_rows, "duplicate_rows": duplicate_rows,
        "rejected_rows": rejected_rows,
        "identity_pending": identity_pending,
        "complete": all(k in {"accepted", "already_complete", "repaired_existing_conflict"} or not v for k, v in reasons.items()),
        # Keep source/validation evidence available to the durable stock-day
        # queue.  The full report remains on disk; this compact projection is
        # safe to pass through the maintenance worker result.
        "source_errors": [item for item in details if item.get("reason") not in {
            "accepted", "already_complete"}],
        "report_path": str(report_dir / "report.json"),
    }
