"""Daily L2 research features, separately versioned from production market data."""

from __future__ import annotations

import hashlib
import json
import math
import time as wall_time
import uuid
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl

from app.data_providers.l2_archive import (
    CODE,
    CONTRACT_VERSION,
    DAY,
    QUANTITY,
    SIDE,
    TIME,
    TRADE_FLAG,
    TRADE_PRICE,
    archive_catalog,
    stream_archive,
)
from app.parquet import atomic_write_parquet

FEATURE_VERSION = "l2_daily_research_v1"
FEATURES = [
    "l2_trade_count",
    "l2_trade_volume",
    "l2_trade_amount",
    "l2_buy_amount_ratio",
    "l2_net_buy_amount_ratio",
    "l2_large_trade_amount_ratio",
    "l2_late_net_buy_ratio",
    "l2_late_amount_ratio",
    "l2_average_trade_amount",
    "l2_vwap_close_gap",
    "l2_depth_imbalance_mean",
    "l2_depth_imbalance_positive_fraction",
    "l2_spread_bps_mean",
]
QUOTE_FIELDS = {
    "q_open": "\u5f00\u76d8\u4ef7",
    "q_high": "\u6700\u9ad8\u4ef7",
    "q_low": "\u6700\u4f4e\u4ef7",
    "q_close": "\u6210\u4ea4\u4ef7",
    "q_volume": "\u5f53\u65e5\u7d2f\u8ba1\u6210\u4ea4\u91cf",
    "q_amount": "\u5f53\u65e5\u6210\u4ea4\u989d",
}


def canonical_equity_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Match bare and exchange-qualified SH/SZ equity keys without aliasing indices."""
    valid = pl.col("symbol").str.contains(
        r"^(?:(?:00|30)\d{4}(?:\.SZ)?|(?:60|68)\d{4}(?:\.SH)?|"
        r"SZ(?:00|30)\d{4}|SH(?:60|68)\d{4})$"
    )
    code = pl.col("symbol").str.extract(r"(\d{6})", 1)
    suffix = pl.when(code.str.starts_with("6")).then(pl.lit(".SH")).otherwise(pl.lit(".SZ"))
    return frame.filter(valid).with_columns((code + suffix).alias("symbol"))


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(
            json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False, default=str),
            encoding="utf-8",
        )
        # Windows readers/indexers can briefly hold the destination without
        # delete-sharing. Keep replacement atomic; never truncate the old file.
        for attempt in range(6):
            try:
                temp.replace(path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                wall_time.sleep(0.05 * (attempt + 1))
    finally:
        temp.unlink(missing_ok=True)


class DailyAccumulator:
    """Memory grows with selected instruments, not the number of tick records."""

    def __init__(self, day: date):
        self.day = day
        self.trades = {}
        self.quotes = {}
        self.clocks = {}
        self.bins = defaultdict(set)
        self.close_prices = {}
        self.event_flags = set()
        self.input_rows = 0
        self.max_batch_rows = 0
        self.excluded_padding_rows = defaultdict(int)

    def normalize(self, frame: pl.DataFrame, kind: str) -> pl.DataFrame:
        self.input_rows += frame.height
        self.max_batch_rows = max(self.max_batch_rows, frame.height)
        frame = frame.with_columns(pl.col(TIME).str.zfill(9).alias("clock"))
        valid = pl.col("clock").str.contains(r"^(?:[01]\d|2[0-3])[0-5]\d[0-5]\d\d{3}$") & (
            pl.col(DAY) == self.day.strftime("%Y%m%d")
        )
        if not frame.select(valid.all()).item():
            raise ValueError("Invalid L2 date or timestamp")
        frame = frame.with_columns(pl.col("clock").cast(pl.Int64))
        backward = (pl.col("clock").diff().over(CODE) < 0).fill_null(False)
        if frame.select(backward.any()).item():
            raise ValueError("L2 timestamps go backwards")
        for symbol, first, last in (
            frame.group_by(CODE, maintain_order=True)
            .agg(pl.col("clock").first().alias("first"), pl.col("clock").last().alias("last"))
            .iter_rows()
        ):
            if first < self.clocks.get((kind, symbol), -1):
                raise ValueError("L2 timestamp reversal across batches")
            self.clocks[kind, symbol] = last
        return frame

    def add_trades(self, frame: pl.DataFrame):
        # Older archives contain a 09:26 non-trade sentinel. Require every
        # event identifier and economic field to be empty/zero, not just date=0.
        zero_fields = [TRADE_PRICE, QUANTITY, "\u6210\u4ea4\u7f16\u53f7", "\u53eb\u5356\u5e8f\u53f7", "\u53eb\u4e70\u5e8f\u53f7"]
        blank_fields = [SIDE, TRADE_FLAG, "\u59d4\u6258\u4ee3\u7801"]
        if set(zero_fields + blank_fields) <= set(frame.columns):
            padding = (
                (pl.col(DAY) == "0")
                & (pl.col(TIME).str.zfill(9) == "092600000")
                & pl.all_horizontal([pl.col(c) == "0" for c in zero_fields])
                & pl.all_horizontal([pl.col(c).fill_null("").str.strip_chars(" \x00") == "" for c in blank_fields])
            ).fill_null(False)
            skipped = frame.filter(padding)
            for symbol, count in skipped.group_by(CODE).len().iter_rows():
                self.excluded_padding_rows[symbol] += count
            self.input_rows += skipped.height
            frame = frame.filter(~padding)
            if frame.is_empty():
                return
        frame = self.normalize(frame, "trade").with_columns(
            pl.col(TRADE_PRICE).cast(pl.Int64).alias("px"),
            pl.col(QUANTITY).cast(pl.Int64).alias("qty"),
        )
        self.event_flags.update(frame.select(CODE, TRADE_FLAG).unique().iter_rows())
        if frame.select(((pl.col("px") < 0) | (pl.col("qty") <= 0)).any()).item():
            raise ValueError("Invalid L2 trade price/quantity")
        allowed = (
            pl.when(pl.col(CODE).str.ends_with(".SZ"))
            .then(
                ((pl.col(TRADE_FLAG) == "0") & (pl.col("px") > 0))
                | ((pl.col(TRADE_FLAG) == "C") & (pl.col("px") == 0))
            )
            .otherwise(
                pl.col(TRADE_FLAG).fill_null("").is_in(["", "\x00", "0"]) & (pl.col("px") > 0)
            )
        )
        if not frame.select(allowed.fill_null(False).all()).item():
            raise ValueError("Unrecognized exchange event contract")
        frame = frame.filter(pl.col("px") > 0)
        if frame.is_empty():
            return
        # SH closing-auction reports can arrive a few seconds after 15:00.
        # Accept only this bounded reporting interval, and reconcile its price
        # with the final snapshot in finish(); never shift or fabricate ticks.
        close_bound = (
            pl.when(pl.col(CODE).str.ends_with(".SH")).then(150005000).otherwise(150000999)
        )
        in_session = pl.col("clock").is_between(92500000, 113000999) | pl.col("clock").is_between(
            130000000, close_bound
        )
        if not frame.select(in_session.all()).item():
            raise ValueError("Executed trades outside supported auction/trading sessions")
        for symbol, lo, hi in (
            frame.filter(pl.col("clock") > 150000000)
            .group_by(CODE)
            .agg(pl.col("px").min().alias("lo"), pl.col("px").max().alias("hi"))
            .iter_rows()
        ):
            old = self.close_prices.get(symbol, (lo, hi))
            self.close_prices[symbol] = (min(lo, old[0]), max(hi, old[1]))
        frame = frame.with_columns(
            (pl.col("px").cast(pl.Float64) * pl.col("qty") / 10000).alias("amt"),
        ).with_columns(
            pl.when(pl.col(SIDE) == "B").then(pl.col("amt")).otherwise(0.0).alias("buy"),
            pl.when(pl.col(SIDE) == "S").then(pl.col("amt")).otherwise(0.0).alias("sell"),
        )
        late = pl.col("clock") >= 143000000
        grouped = frame.group_by(CODE, maintain_order=True).agg(
            pl.len().alias("count"),
            pl.col("qty").sum().alias("volume"),
            pl.col("amt").sum().alias("amount"),
            pl.col("buy").sum(),
            pl.col("sell").sum(),
            pl.col("amt").filter(pl.col("amt") >= 200000).sum().alias("large"),
            pl.col("amt").filter(late).sum().alias("late_amount"),
            (pl.col("buy") - pl.col("sell")).filter(late).sum().alias("late_net"),
            pl.col("px").first().alias("open"),
            pl.col("px").last().alias("close"),
            pl.col("px").max().alias("high"),
            pl.col("px").min().alias("low"),
        )
        for row in grouped.iter_rows(named=True):
            old = self.trades.get(row[CODE])
            if old is None:
                self.trades[row[CODE]] = row
                continue
            for key in [
                "count",
                "volume",
                "amount",
                "buy",
                "sell",
                "large",
                "late_amount",
                "late_net",
            ]:
                old[key] += row[key]
            old["close"] = row["close"]
            old["high"] = max(old["high"], row["high"])
            old["low"] = min(old["low"], row["low"])

    def add_quotes(self, frame: pl.DataFrame):
        frame = self.normalize(frame, "quote")
        aliases = [pl.col(v).cast(pl.Float64).alias(k) for k, v in QUOTE_FIELDS.items()]
        for side, stem in [("bid", "\u7533\u4e70"), ("ask", "\u7533\u5356")]:
            for level in range(1, 11):
                aliases += [
                    pl.col(stem + "\u4ef7" + str(level)).cast(pl.Float64).alias(f"{side}_p{level}"),
                    pl.col(stem + "\u91cf" + str(level)).cast(pl.Float64).alias(f"{side}_v{level}"),
                ]
        frame = frame.with_columns(aliases)
        for row in (
            frame.group_by(CODE, maintain_order=True)
            .agg(*[pl.col(k).last() for k in QUOTE_FIELDS], pl.col("clock").last())
            .iter_rows(named=True)
        ):
            old = self.quotes.setdefault(
                row[CODE],
                {
                    "samples": 0,
                    "depth_sum": 0.0,
                    "depth_count": 0,
                    "positive": 0,
                    "spread_sum": 0.0,
                    "spread_count": 0,
                },
            )
            old.update({k: row[k] for k in [*QUOTE_FIELDS, "clock"]})
        regular = pl.col("clock").is_between(93000000, 113000000) | pl.col("clock").is_between(
            130000000, 150000000
        )
        frame = frame.filter(regular)
        if frame.is_empty():
            return
        qty_cols = [f"{side}_v{i}" for side in ["bid", "ask"] for i in range(1, 11)]
        price_cols = [f"{side}_p{i}" for side in ["bid", "ask"] for i in range(1, 11)]
        if not frame.select(
            pl.all_horizontal(
                [pl.col(c).is_finite() & (pl.col(c) >= 0) for c in [*qty_cols, *price_cols]]
            ).all()
        ).item():
            raise ValueError("Invalid depth prices/quantities")
        frame = frame.with_columns(
            *[
                pl.sum_horizontal(
                    [
                        pl.when(pl.col(f"{s}_p{i}") > 0).then(pl.col(f"{s}_v{i}")).otherwise(0)
                        for i in range(1, 11)
                    ]
                ).alias(s)
                for s in ["bid", "ask"]
            ]
        )
        depth_valid = pl.col("bid") + pl.col("ask") > 0
        spread_valid = (pl.col("bid_p1") > 0) & (pl.col("ask_p1") >= pl.col("bid_p1"))
        frame = frame.with_columns(
            pl.when(depth_valid)
            .then((pl.col("bid") - pl.col("ask")) / (pl.col("bid") + pl.col("ask")))
            .alias("depth"),
            pl.when(spread_valid)
            .then(
                (pl.col("ask_p1") - pl.col("bid_p1"))
                * 20000
                / (pl.col("ask_p1") + pl.col("bid_p1"))
            )
            .alias("spread"),
            ((pl.col("clock") // 10000000) * 60 + (pl.col("clock") // 100000) % 100).alias(
                "minute"
            ),
        )
        for symbol, minutes in frame.group_by(CODE).agg(pl.col("minute").unique()).iter_rows():
            self.bins[symbol].update(
                (m - 570) // 5 if 570 <= m < 690 else 24 + (m - 780) // 5
                for m in minutes
                if 570 <= m < 690 or 780 <= m < 900
            )
        for row in (
            frame.group_by(CODE)
            .agg(
                pl.len().alias("samples"),
                pl.col("depth").sum().alias("depth_sum"),
                pl.col("depth").count().alias("depth_count"),
                (pl.col("depth") > 0).sum().alias("positive"),
                pl.col("spread").sum().alias("spread_sum"),
                pl.col("spread").count().alias("spread_count"),
            )
            .iter_rows(named=True)
        ):
            for key in [
                "samples",
                "depth_sum",
                "depth_count",
                "positive",
                "spread_sum",
                "spread_count",
            ]:
                self.quotes[row[CODE]][key] += row[key]

    def finish(self, symbol: str, reference: dict | None, *, daily_volume_unit="lots") -> tuple[dict | None, str]:
        if daily_volume_unit not in {"shares", "lots"}:
            raise ValueError("daily_volume_unit must be shares or lots")
        volume_scale = 1 if daily_volume_unit == "shares" else 100
        t, q = self.trades.get(symbol), self.quotes.get(symbol)
        if not t or not q or reference is None:
            return None, "missing_trade_quote_or_daily_reference"
        if q["clock"] < 150000000 or len(self.bins[symbol]) < 46:
            return None, "incomplete_quote_session"
        if symbol in self.close_prices and self.close_prices[symbol] != (
            q["q_close"],
            q["q_close"],
        ):
            return None, "closing_auction_report_price_mismatch"
        for key in ["open", "high", "low", "close"]:
            if not math.isfinite(q["q_" + key]) or abs(t[key] - q["q_" + key]) > 1:
                return None, "trade_quote_price_mismatch"
            if not math.isfinite(reference[key]) or abs(t[key] / 10000 - reference[key]) > 0.011:
                return None, "daily_price_mismatch"
        if (
            not math.isfinite(q["q_amount"])
            or not math.isfinite(q["q_volume"])
            or t["volume"] != q["q_volume"]
            or abs(t["amount"] - q["q_amount"]) > 1.0
        ):
            return None, "trade_quote_turnover_mismatch"
        # Explicit reference contract; never guess share/lot units by magnitude.
        if (
            not math.isfinite(reference["volume"])
            or not math.isfinite(reference["amount"])
            or abs(t["volume"] - reference["volume"] * volume_scale) > max(100, t["volume"] * 0.002)
            or abs(t["amount"] - reference["amount"]) > max(1, t["amount"] * 0.002)
        ):
            return None, "daily_turnover_mismatch"
        amount = t["amount"]
        if amount <= 0 or t["volume"] <= 0 or q["depth_count"] < q["samples"] * 0.95:
            return None, "insufficient_valid_depth_or_trades"
        side_coverage = (t["buy"] + t["sell"]) / amount
        if side_coverage < 0.98:
            return None, "unclassified_trade_side"
        row = {
            "symbol": symbol,
            "date": self.day,
            "available_at": datetime.combine(self.day + timedelta(days=1), time.min),
            "quality_status": "validated",
            "feature_version": FEATURE_VERSION,
            "source_contract": CONTRACT_VERSION,
            "l2_trade_count": float(t["count"]),
            "l2_trade_volume": float(t["volume"]),
            "l2_trade_amount": amount,
            "l2_buy_amount_ratio": t["buy"] / amount,
            "l2_net_buy_amount_ratio": (t["buy"] - t["sell"]) / amount,
            "l2_large_trade_amount_ratio": t["large"] / amount,
            "l2_late_net_buy_ratio": t["late_net"] / amount,
            "l2_late_amount_ratio": t["late_amount"] / amount,
            "l2_average_trade_amount": amount / t["count"],
            "l2_vwap_close_gap": t["close"] / 10000 / (amount / t["volume"]) - 1,
            "l2_depth_imbalance_mean": q["depth_sum"] / q["depth_count"],
            "l2_depth_imbalance_positive_fraction": q["positive"] / q["depth_count"],
            "l2_spread_bps_mean": q["spread_sum"] / q["spread_count"]
            if q["spread_count"]
            else None,
            "l2_side_coverage": side_coverage,
            "l2_quote_coverage": len(self.bins[symbol]) / 48,
        }
        return row, "validated"


def extract_day(
    data_dir: Path,
    archive: Path,
    day: date,
    symbols: list[str],
    *,
    batch_rows=8192,
    timeout=600,
    tar="tar",
    research_dir: Path | None = None,
    daily_volume_unit="lots",
) -> dict:
    if daily_volume_unit not in {"shares", "lots"}:
        raise ValueError("daily_volume_unit must be shares or lots")
    root = research_dir or data_dir / "l2_research"
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f".{day}.lock"
    with lock.open("x", encoding="ascii"):
        pass
    try:
        return _extract_day(
            data_dir,
            archive,
            day,
            symbols,
            batch_rows=batch_rows,
            timeout=timeout,
            tar=tar,
            root=root,
            daily_volume_unit=daily_volume_unit,
        )
    finally:
        lock.unlink(missing_ok=True)


def _extract_day(data_dir, archive, day, symbols, *, batch_rows, timeout, tar, root, daily_volume_unit):
    report_path = root / "extraction" / f"{day}.json"
    daily_path = data_dir / "kline_daily" / f"date={day}" / "part.parquet"
    output_path = root / "daily" / f"date={day}" / "part.parquet"
    if not daily_path.exists():
        return {"date": str(day), "status": "daily_reference_missing"}
    stat = archive.stat()
    signature = {
        "archive": str(archive.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "daily_sha256": hashlib.sha256(daily_path.read_bytes()).hexdigest(),
        "symbols": sorted(symbols),
        "feature_version": FEATURE_VERSION,
    }
    if daily_volume_unit != "lots":
        signature["daily_volume_unit"] = daily_volume_unit
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            previous.get("signature") == signature
            and previous.get("status") == "complete"
            and (
                not previous.get("accepted")
                or (
                    output_path.exists()
                    and previous.get("output_sha256")
                    == hashlib.sha256(output_path.read_bytes()).hexdigest()
                )
            )
        ):
            return {**previous, "resumed": True}
        # Research partitions are immutable unless explicitly moved to another run.
        if output_path.exists():
            raise ValueError(
                "Existing L2 research partition differs; use a new research data directory"
            )
    elif output_path.exists():
        raise ValueError("Unmanifested L2 research partition exists")
    refs = canonical_equity_rows(pl.read_parquet(daily_path)).filter(pl.col("symbol").is_in(symbols))
    if refs["symbol"].n_unique() != refs.height:
        raise ValueError("Ambiguous daily references")
    references = {r["symbol"]: r for r in refs.iter_rows(named=True)}
    active_symbols = sorted(references)
    if not active_symbols:
        return {"date": str(day), "status": "daily_reference_missing"}
    entries = archive_catalog(archive, tar)
    accumulator = DailyAccumulator(day)
    evidence = []
    for kind in ["trade", "quote"]:
        stream_info = {}
        for batch in stream_archive(
            archive,
            entries,
            set(active_symbols),
            kind,
            stream_info,
            batch_rows=batch_rows,
            timeout=timeout,
            tar=tar,
        ):
            (accumulator.add_trades if kind == "trade" else accumulator.add_quotes)(batch)
        evidence.append(stream_info)
    rows, checks = [], []
    for symbol in symbols:
        row, reason = accumulator.finish(symbol, references.get(symbol), daily_volume_unit=daily_volume_unit)
        checks.append({"symbol": symbol, "reason": reason})
        if row is not None:
            rows.append(row)
    # No data changes are allowed between reference validation and publication.
    if hashlib.sha256(daily_path.read_bytes()).hexdigest() != signature["daily_sha256"]:
        raise ValueError("Daily reference changed during extraction")
    if (
        archive.stat().st_size != signature["size"]
        or archive.stat().st_mtime_ns != signature["mtime_ns"]
    ):
        raise ValueError("Archive changed during extraction")
    report = {
        "date": str(day),
        "status": "complete",
        "signature": signature,
        "checks": checks,
        "accepted": len(rows),
        "rejected": len(symbols) - len(rows),
        "streams": evidence,
        "input_rows": accumulator.input_rows,
        "max_batch_rows": accumulator.max_batch_rows,
        "observed_event_flags": sorted([list(x) for x in accumulator.event_flags], key=str),
        "excluded_padding_rows": dict(accumulator.excluded_padding_rows),
        "closing_report_prices": accumulator.close_prices,
        "production_writes": 0,
        "daily_volume_unit": daily_volume_unit,
        "source_dictionary_confirmed": False,
        "available_at_policy": "historical research assumption: next calendar day 00:00 Asia/Shanghai",
        "limitations": [
            "research only",
            "supplier B/S proxy",
            "no order or cancellation attribution",
            "snapshot-weighted depth, not time-weighted",
            "no live delivery contract",
        ],
    }
    if rows:
        atomic_write_parquet(
            pl.DataFrame(rows).with_columns([pl.col(c).cast(pl.Float64) for c in FEATURES]),
            output_path,
        )
        report["output_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
        report["output_bytes"] = output_path.stat().st_size
    write_json(report_path, report)
    return report
