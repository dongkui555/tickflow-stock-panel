from __future__ import annotations

import argparse
import json
import ssl
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import polars as pl

from app.services.minute_repair import daily_reference
from app.services.security_identity import IdentityError, parse_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit A-share minute coverage against traded daily bars."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--markets", nargs="+", choices=("BJ", "SH", "SZ"), default=("BJ", "SH"))
    return parser.parse_args()


def normalized_code_expr() -> pl.Expr:
    return pl.col("symbol").cast(pl.String).str.extract(r"(\d{6})", 1).alias("code")


def market_expr() -> pl.Expr:
    symbol = pl.col("symbol").cast(pl.String).str.to_uppercase()
    code = pl.col("code")
    return (
        pl.when(symbol.str.ends_with(".BJ") | code.str.contains(r"^(4|8|92)"))
        .then(pl.lit("BJ"))
        .when(symbol.str.ends_with(".SH") | code.str.starts_with("6"))
        .then(pl.lit("SH"))
        .when(symbol.str.ends_with(".SZ") | code.str.contains(r"^(00|30)"))
        .then(pl.lit("SZ"))
        .otherwise(None)
        .alias("market")
    )


def a_share_expr() -> pl.Expr:
    code = pl.col("code").cast(pl.String)
    market = pl.col("market")
    return (
        ((market == "BJ") & code.str.contains(r"^(43|83|87|88|92)"))
        | ((market == "SH") & code.str.contains(r"^(60|68)"))
        | ((market == "SZ") & code.str.contains(r"^(00|30)"))
    )


def _dotenv_value(path: Path, name: str) -> str:
    if not path.exists():
        return ""
    prefix = f"{name}="
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip().strip('"').strip("'")
        return value
    return ""


def _online_trade_dates(data_dir: Path, start: date, end: date) -> list[date]:
    """Read an authoritative open-day calendar without logging credentials."""
    dotenv = data_dir.parent / ".env"
    base_url = _dotenv_value(dotenv, "TUSHARE_PROMAX_URL").rstrip("/")
    api_key = _dotenv_value(dotenv, "TUSHARE_PROMAX_KEY")
    if not base_url or not api_key:
        return []
    base_params = {
        "start_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "is_open": 1,
        "fields": "cal_date,is_open",
    }
    payload = None
    last_error: Exception | None = None
    verify = _dotenv_value(dotenv, "TUSHARE_PROMAX_VERIFY_TLS").lower() not in {"0", "false", "no"}
    context = None if verify else ssl._create_unverified_context()
    for exchange in ("", "SSE"):
        params = urllib.parse.urlencode({**base_params, "exchange": exchange})
        request = urllib.request.Request(
            f"{base_url}/trade_cal?{params}",
            headers={"X-API-Key": api_key, "User-Agent": "TickFlow-readonly-audit/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30, context=context) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if payload is None:
        print(f"online trade calendar unavailable: {type(last_error).__name__}: {last_error}", flush=True)
        return []

    data = payload.get("data") if isinstance(payload, dict) else None
    rows = data if isinstance(data, list) else (data.get("items", []) if isinstance(data, dict) else [])
    if not rows and isinstance(payload, dict):
        rows = payload.get("rows", []) or payload.get("result", [])
    dates: set[date] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("is_open", 1)) not in {"1", "1.0", "True", "true"}:
            continue
        raw = str(row.get("cal_date") or row.get("date") or "").replace("-", "")[:8]
        try:
            value = date.fromisoformat(f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}")
        except ValueError:
            continue
        if start <= value <= end:
            dates.add(value)
    return sorted(dates)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    daily_root = data_dir / "kline_daily"
    minute_root = data_dir / "kline_minute"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or data_dir / "import_reports" / f"minute_audit_sh_bj_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    markets = tuple(args.markets)

    aggregate: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_date: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    sparse_rows: list[dict[str, object]] = []
    index_root = data_dir / "kline_index_daily"
    index_dates: set[date] = {
        date.fromisoformat(path.parent.name.removeprefix("date="))
        for path in index_root.glob("date=*/part.parquet")
        if args.start <= date.fromisoformat(path.parent.name.removeprefix("date=")) <= args.end
    }
    online_dates = set(_online_trade_dates(data_dir, args.start, args.end))
    reference_dates: set[date] = online_dates or index_dates
    calendar_source = "promax_trade_cal" if online_dates else "local_kline_index_partitions"
    daily_market_dates: dict[str, set[date]] = {market: set() for market in markets}
    minute_market_codes: dict[tuple[date, str], int] = defaultdict(int)
    processed_dates = 0

    for daily_path in sorted(daily_root.glob("date=*/part.parquet")):
        trade_date = date.fromisoformat(daily_path.parent.name.removeprefix("date="))
        if trade_date < args.start or trade_date > args.end:
            continue

        raw_daily = pl.read_parquet(daily_path)
        daily = (
            raw_daily
            .with_columns(normalized_code_expr())
            .with_columns(market_expr())
            .filter(pl.col("market").is_in(markets) & a_share_expr())
        )
        if "quote_ts" in daily.columns:
            # Match the importer's daily_reference authority before selecting
            # aliases; a qualified intraday snapshot may predate final history.
            authoritative = pl.col("quote_ts").is_null()
            daily = daily.filter(authoritative == authoritative.max().over(["market", "code"]))
        daily = (
            daily.select("symbol", "open", "high", "low", "close", "volume", "amount", "code", "market")
            .sort("symbol")
            .unique(["market", "code"], keep="last")
        )
        if daily.is_empty():
            continue

        # 北交所的原始日K金额可能包含 15:01-15:30 盘后固定价格成交，
        # 而分钟K只覆盖连续交易时段。已有 minute_reference_evidence 时，
        # 用带来源证据的正规交易时段基准审计，避免把真实的盘后成交误报为分钟数据错误。
        references = daily_reference(raw_daily, data_dir=data_dir)
        evidence_rows = []
        for code, reference in references.items():
            try:
                identity = parse_identity(reference.get("symbol") or code, asset_type="stock")
            except IdentityError:
                continue
            market = identity.exchange
            if market not in markets:
                continue
            evidence_rows.append({
                "market": market,
                "code": code,
                "_reference_open": reference.get("open"),
                "_reference_high": reference.get("high"),
                "_reference_low": reference.get("low"),
                "_reference_close": reference.get("close"),
                "_reference_volume": reference.get("volume"),
                "_reference_amount": reference.get("amount"),
                "_reference_scope": "regular_session" if reference.get("_volume_scope") else "daily_total",
            })
        if evidence_rows:
            daily = daily.join(pl.DataFrame(evidence_rows), on=["market", "code"], how="left")
        else:
            daily = daily.with_columns(
                *[pl.col(column).alias(f"_reference_{column}") for column in ("open", "high", "low", "close", "volume", "amount")],
                pl.lit("daily_total").alias("_reference_scope"),
            )
        daily = daily.with_columns(
            *[
                pl.coalesce([pl.col(f"_reference_{column}"), pl.col(column)]).alias(f"_reference_{column}")
                for column in ("open", "high", "low", "close", "volume", "amount")
            ],
            pl.col("_reference_scope").fill_null("daily_total").alias("_reference_scope"),
        )

        active_daily = daily.filter((pl.col("volume") > 0) | (pl.col("amount") > 0))
        if not active_daily.is_empty():
            reference_dates.add(trade_date)
            for market in markets:
                if active_daily.filter(pl.col("market") == market).height:
                    daily_market_dates[market].add(trade_date)

        active = active_daily
        inactive = daily.filter((pl.col("volume") <= 0) & (pl.col("amount") <= 0))

        minute_path = minute_root / daily_path.parent.name / "part.parquet"
        if minute_path.exists():
            stamp = pl.col("datetime")
            clock = stamp.dt.hour().cast(pl.Int32) * 60 + stamp.dt.minute().cast(pl.Int32)
            clean_stamp = (stamp.dt.second() == 0) & (stamp.dt.microsecond() == 0)
            bj_after_hours = (pl.col("market") == "BJ") & clock.is_between(901, 930)
            allowed = clean_stamp & (
                clock.is_between(570, 690) | clock.is_between(781, 900) | bj_after_hours
            )
            session = clean_stamp & (clock.is_between(571, 690) | clock.is_between(781, 900))
            turnover_session = session | bj_after_hours
            traded = turnover_session & ((pl.col("volume") > 0) | (pl.col("amount") > 0))
            minute_rows = (
                pl.read_parquet(
                    minute_path,
                    columns=["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"],
                )
                .with_columns(normalized_code_expr())
                .with_columns(market_expr())
                .filter(pl.col("market").is_in(markets) & a_share_expr())
                .with_columns(
                    (
                        (pl.col("market") == "BJ") & pl.col("symbol").cast(pl.String).str.to_uppercase().str.ends_with(".BJ")
                        | (pl.col("market") == "SH") & pl.col("symbol").cast(pl.String).str.to_uppercase().str.ends_with(".SH")
                        | (pl.col("market") == "SZ") & pl.col("symbol").cast(pl.String).str.to_uppercase().str.ends_with(".SZ")
                    ).alias("_native")
                )
            )
            alias_choices = (
                minute_rows.group_by(["market", "code", "symbol"])
                .agg(
                    pl.col("datetime").filter(session).n_unique().alias("_session_rows"),
                    pl.col("datetime").filter(allowed).n_unique().alias("_allowed_rows"),
                    pl.col("_native").any().alias("_native"),
                )
                .with_columns(pl.len().over(["market", "code"]).alias("alias_count"))
                .sort(
                    ["market", "code", "_session_rows", "_allowed_rows", "_native", "symbol"],
                    descending=[False, False, True, True, True, False],
                )
                .unique(["market", "code"], keep="first", maintain_order=True)
                .select("market", "code", "symbol", "alias_count")
            )
            minute_rows = minute_rows.join(alias_choices, on=["market", "code", "symbol"], how="inner")
            minute = (
                minute_rows.sort(["market", "code", "datetime"])
                .group_by(["market", "code"])
                .agg(
                    pl.col("symbol").first().alias("selected_symbol"),
                    pl.col("alias_count").first(),
                    pl.col("datetime").filter(session).n_unique().alias("bars"),
                    pl.col("datetime").filter(allowed).n_unique().alias("rows"),
                    pl.col("datetime").filter(bj_after_hours).n_unique().alias("after_hours_rows"),
                    pl.col("open").filter(traded).first().alias("minute_open"),
                    pl.col("high").filter(traded).max().alias("minute_high"),
                    pl.col("low").filter(traded).min().alias("minute_low"),
                    # A terminal 15:00 bar may have zero turnover while still
                    # carrying the valid session close. Do not use the last
                    # traded bar as a proxy for the daily close.
                    pl.col("close").filter(turnover_session).last().alias("minute_close"),
                    pl.col("volume").filter(allowed).sum().alias("minute_volume"),
                    pl.col("amount").filter(allowed).sum().alias("minute_amount"),
                )
            )
        else:
            minute = pl.DataFrame(
                schema={
                    "market": pl.String,
                    "code": pl.String,
                    "bars": pl.UInt32,
                    "rows": pl.UInt32,
                    "minute_open": pl.Float64,
                    "minute_high": pl.Float64,
                    "minute_low": pl.Float64,
                    "minute_close": pl.Float64,
                    "minute_volume": pl.Float64,
                    "minute_amount": pl.Float64,
                    "after_hours_rows": pl.UInt32,
                    "selected_symbol": pl.String,
                    "alias_count": pl.UInt32,
                }
            )

        for market in markets:
            minute_market_codes[(trade_date, market)] = minute.filter(pl.col("market") == market).height

        price_tolerance = pl.max_horizontal(pl.lit(0.011), pl.col("_reference_close").abs() * 1e-6)
        # The canonical local minute contract starts at 09:31 and may omit
        # auction prints included in the daily OHLC. Require the regular
        # session close to match, and keep its open/high/low within the daily
        # range instead of requiring auction-inclusive equality.
        price_match = (
            ((pl.col("minute_close") - pl.col("_reference_close")).abs() <= price_tolerance)
            & (pl.col("minute_high") <= pl.col("_reference_high") + price_tolerance)
            & (pl.col("minute_low") >= pl.col("_reference_low") - price_tolerance)
            & pl.col("minute_open").is_between(
                pl.col("_reference_low") - price_tolerance,
                pl.col("_reference_high") + price_tolerance,
            )
        )
        volume_match = pl.any_horizontal(
            [
                (pl.col("minute_volume") * scale - pl.col("_reference_volume")).abs()
                <= pl.max_horizontal(pl.lit(1.0), pl.col("_reference_volume").abs() * 0.08)
                for scale in (0.01, 1, 100)
            ]
        )
        amount_match = (pl.col("minute_amount") - pl.col("_reference_amount")).abs() <= pl.max_horizontal(
            pl.lit(1.0), pl.col("_reference_amount").abs() * 0.005
        )
        audited = (
            active.join(minute, on=["market", "code"], how="left")
            .with_columns(
                pl.col("bars").fill_null(0),
                pl.col("rows").fill_null(0),
            )
            .with_columns(
                price_match.fill_null(False).alias("price_match"),
                volume_match.fill_null(False).alias("volume_match"),
                amount_match.fill_null(False).alias("amount_match"),
            )
            .with_columns(
                (pl.col("price_match") & pl.col("volume_match") & pl.col("amount_match"))
                .alias("daily_match")
            )
            .with_columns(
                pl.when(pl.col("bars") == 0)
                .then(pl.lit("missing_0"))
                .when((pl.col("bars") < 240) & pl.col("daily_match"))
                .then(pl.lit("valid_sparse"))
                .when(pl.col("bars") < 240)
                .then(pl.lit("invalid_partial"))
                .when((pl.col("bars") == 240) & pl.col("daily_match"))
                .then(pl.lit("complete_valid"))
                .when(pl.col("bars") == 240)
                .then(pl.lit("complete_conflict"))
                .otherwise(pl.lit("invalid_time_axis"))
                .alias("status")
            )
        )

        for market in markets:
            market_rows = audited.filter(pl.col("market") == market)
            if market_rows.is_empty():
                continue
            counts = {
                row["status"]: row["len"]
                for row in market_rows.group_by("status").len().to_dicts()
            }
            expected = market_rows.height
            zero_trade = inactive.filter(pl.col("market") == market).height
            row = {
                "date": trade_date.isoformat(),
                "year": trade_date.year,
                "market": market,
                "expected_active_stock_days": expected,
                "zero_trade_excluded": zero_trade,
                "missing_0": counts.get("missing_0", 0),
                "invalid_partial": counts.get("invalid_partial", 0),
                "valid_sparse": counts.get("valid_sparse", 0),
                "complete_valid": counts.get("complete_valid", 0),
                "complete_conflict": counts.get("complete_conflict", 0),
                "invalid_time_axis": counts.get("invalid_time_axis", 0),
            }
            by_date.append(row)
            key = (trade_date.year, market)
            aggregate[key]["trading_dates"] += 1
            for field in row:
                if field not in {"date", "year", "market"}:
                    aggregate[key][field] += int(row[field])

        issues = audited.filter(
            pl.col("status").is_in(
                ["missing_0", "invalid_partial", "complete_conflict", "invalid_time_axis"]
            )
        )
        issue_rows.extend(
            {
                "date": trade_date.isoformat(),
                "year": trade_date.year,
                **row,
            }
            for row in issues.select(
                "market", "code", "bars", "rows", "after_hours_rows", "status", "daily_match", "volume", "amount",
                "minute_volume", "minute_amount", "price_match", "volume_match",
                "amount_match", "selected_symbol", "alias_count", "_reference_scope",
                "_reference_volume", "_reference_amount",
            ).to_dicts()
        )
        sparse = audited.filter(pl.col("status") == "valid_sparse")
        sparse_rows.extend(
            {
                "date": trade_date.isoformat(),
                "year": trade_date.year,
                **row,
            }
            for row in sparse.select(
                "market", "code", "bars", "rows", "status", "volume", "amount",
                "minute_volume", "minute_amount",
            ).to_dicts()
        )

        processed_dates += 1
        if processed_dates % 25 == 0:
            print(f"audited {processed_dates} trading dates, latest={trade_date}", flush=True)

    summary_rows: list[dict[str, object]] = []
    for (year, market), values in sorted(aggregate.items()):
        expected = values["expected_active_stock_days"]
        usable = values["valid_sparse"] + values["complete_valid"]
        strict = values["complete_valid"]
        summary_rows.append(
            {
                "year": year,
                "market": market,
                **values,
                "strict_valid_240_coverage_pct": round(strict * 100 / expected, 4) if expected else 0,
                "validated_usable_coverage_pct": round(usable * 100 / expected, 4) if expected else 0,
            }
        )

    pl.DataFrame(summary_rows).write_csv(output_dir / "summary.csv")
    pl.DataFrame(by_date).write_csv(output_dir / "by_date.csv")
    daily_gap_rows = []
    for trade_date in sorted(reference_dates):
        if trade_date < args.start or trade_date > args.end:
            continue
        for market in markets:
            if trade_date not in daily_market_dates[market]:
                daily_gap_rows.append(
                    {
                        "date": trade_date.isoformat(),
                        "year": trade_date.year,
                        "market": market,
                        "daily_status": "daily_missing_for_market",
                        "minute_symbols_present": minute_market_codes[(trade_date, market)],
                    }
                )
    if daily_gap_rows:
        pl.DataFrame(daily_gap_rows).write_csv(output_dir / "daily_market_missing_dates.csv")
    else:
        (output_dir / "daily_market_missing_dates.csv").write_text("", encoding="utf-8")
    if issue_rows:
        pl.DataFrame(issue_rows, infer_schema_length=None).sort(["market", "date", "code"]).write_csv(output_dir / "missing_or_partial.csv")
    else:
        (output_dir / "missing_or_partial.csv").write_text("", encoding="utf-8")
    if sparse_rows:
        pl.DataFrame(sparse_rows, infer_schema_length=None).sort(["market", "date", "code"]).write_csv(output_dir / "validated_sparse.csv")
    else:
        (output_dir / "validated_sparse.csv").write_text("", encoding="utf-8")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "calendar_source": calendar_source,
        "definition": {
            "expected": "An audited-market stock-day with local daily volume or amount greater than zero.",
            "missing": "No unique minute timestamps.",
            "minute_session": "沪深: 09:31-11:30 and 13:01-15:00; 北交所 additionally includes 15:01-15:30 after-hours fixed-price trading for daily turnover reconciliation.",
            "valid_sparse": "Fewer than 240 timestamps but daily OHLC, volume and amount pass QMT tolerances.",
            "invalid_partial": "Fewer than 240 timestamps and daily aggregate validation fails.",
            "complete": "Exactly 240 timestamps and daily aggregate validation passes.",
            "excluded": "Daily bars with both volume and amount equal to zero; listing dates are implicit in daily data.",
        },
        "summary": summary_rows,
        "files": {
            "summary": "summary.csv",
            "by_date": "by_date.csv",
            "missing_or_partial": "missing_or_partial.csv",
            "validated_sparse": "validated_sparse.csv",
            "daily_market_missing_dates": "daily_market_missing_dates.csv",
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"report_dir={output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
