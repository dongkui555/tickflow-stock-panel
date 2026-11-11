"""Stage and publish one trade day using the panel's existing import services."""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import math
import os
import shutil
import stat
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from zipfile import BadZipFile, ZipFile

import polars as pl

from app.services.financial_sync import _write_table
from app.services.market_calendar_sync import CALENDAR, _normalize
from app.services.minute_features import FEATURE_DEFAULTS, build_daily_features
from app.services.minute_gap_queue import record_cross_source_results, record_statuses
from app.services.minute_repair import (
    COLS,
    _daily_aggregate_exprs,
    canonical_minute_aliases,
    checked_additions,
    daily_reference,
    keyed,
    repair_minute_gaps,
)
from app.services.security_identity import canonicalize_identity_frame, exchange_for_code

from source import LC1, sha, verify_package

sys.path.insert(0, "/app/scripts")
from import_staged_minute_gaps import StagedMinuteProvider  # noqa: E402

ROOT = Path("/app/data")
CODE = Path(__file__).resolve().parent
CLOCKS = list(range(571, 691)) + list(range(781, 901))
SCHEMA = {"symbol": pl.String, "datetime": pl.Datetime("us"), **dict.fromkeys(COLS[2:], pl.Float64)}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, payload) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, default=str, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def partition(name: str, day: date) -> Path:
    return ROOT / name / f"date={day}" / "part.parquet"


def plan(day: date) -> dict:
    path = ROOT / CALENDAR
    day_state = _normalize(pl.read_parquet(path).to_dicts()).get(day) if path.is_file() else None
    if day_state != 1:
        return {"status": "closed" if day_state == 0 else "calendar_unavailable", "date": str(day)}
    daily = partition("kline_daily", day)
    if not daily.is_file():
        return {"status": "waiting_daily", "date": str(day)}
    refs = daily_reference(pl.read_parquet(daily), data_dir=ROOT)
    symbols = [f"{code}.{exchange_for_code(code, 'stock')}" for code in sorted(refs)]
    if not symbols or any(symbol.endswith(".None") for symbol in symbols):
        raise ValueError("Empty or unresolved daily stock universe")
    return {"status": "ready", "date": str(day), "symbols": symbols, "daily_sha256": sha(daily)}


def minute_frame(symbol: str, day: date, data: bytes):
    if not data:
        return pl.DataFrame(schema=SCHEMA), "source_empty"
    wire = list(LC1.iter_unpack(data))
    expected = (day.year - 2004) * 2048 + day.month * 100 + day.day
    if any(row[0] != expected or not 0 <= row[1] < 1440 for row in wire):
        raise ValueError("LC1 date or clock mismatch")
    rows = [(symbol, datetime.combine(day, time(m // 60, m % 60)), opening, high, low, close, shares / 100, amount)
            for _, m, opening, high, low, close, amount, shares, _ in wire]
    sample = pl.DataFrame(rows, schema=SCHEMA, orient="row")
    return sample, "valid_axis" if [row[1] for row in wire] == CLOCKS else "source_incomplete"


def stage_minute(day: date, out: Path, current: dict) -> dict:
    daily, minute = (partition(name, day) for name in ("kline_daily", "kline_minute"))
    refs = daily_reference(pl.read_parquet(daily), data_dir=ROOT)
    old_sha = sha(minute) if minute.exists() else None
    old = canonical_minute_aliases(pl.read_parquet(minute, columns=COLS), day) if old_sha else pl.DataFrame(schema=SCHEMA)
    by_code = {(key[0] if isinstance(key, tuple) else key): frame.drop("_code")
               for key, frame in keyed(old).partition_by("_code", as_dict=True).items()}
    with (out / "before-coverage/missing_or_partial.csv").open(encoding="utf-8", newline="") as stream:
        targets = {f"{row['code']}.{row['market']}" for row in csv.DictReader(stream)}
    universe = set(current["symbols"])
    if not targets <= universe:
        raise ValueError("Coverage audit contains a stock outside the daily universe")
    stage = out / "staged"
    stage.mkdir()
    accepted, failures, seen = [], [], set()
    with gzip.open(out / "minutes.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            symbol = item["symbol"]
            if symbol in seen or symbol not in universe:
                raise ValueError("Unexpected or duplicate LC1 identity")
            seen.add(symbol)
            if symbol not in targets:
                continue
            code = symbol[:6]
            ref = refs[code]
            sample, reason = minute_frame(symbol, day, base64.b64decode(item["records"], validate=True))
            aggregate = None
            if sample.height:
                aggregate = sample.select(_daily_aggregate_exprs()).row(0, named=True)
                aggregate.update(volume_unit="lots", rows=sample.height, symbol=symbol, date=str(day))
            if reason == "valid_axis":
                existing = by_code.get(code, sample.clear())
                delta, reason = checked_additions(sample, existing, ref, day, source="tdx")
                if reason == "accepted" and abs(aggregate["volume"] * 100 - ref["volume"]) > 0.002 * max(1, ref["volume"]):
                    reason = "explicit_source_unit_conflict"
                if reason == "accepted":
                    path = stage / f"{symbol}_{day}.parquet"
                    sample.write_parquet(path)
                    accepted.append({"symbol": symbol, "date": str(day), "reason": reason, "file": path.name,
                                     "sha256": sha(path), "added_rows": delta.height})
                elif reason == "already_complete":
                    # The independent coverage audit disagreed with this exact duplicate.
                    reason = "coverage_audit_conflict"
            elif not sample.height:
                reason = item["status"]
            if reason != "accepted":
                failures.append({"symbol": symbol, "date": str(day), "reason": reason, "source_rows": sample.height,
                                 "local": {**ref, "symbol": symbol, "date": str(day)}, "minute": aggregate})
    if seen != universe or sha(daily) != current["daily_sha256"] or (
        sha(minute) if minute.exists() else None
    ) != old_sha:
        raise ValueError("Minute plan or partition changed during validation")
    save(stage / "report.json", accepted)
    save(out / "minute-failures.json", failures)
    return {"accepted": len(accepted), "failures": len(failures), "expected": len(current["symbols"]),
            "daily_sha256": current["daily_sha256"], "minute_sha256": old_sha}


def financial_rows(out: Path, package: dict, day: date, fields: dict) -> list[dict]:
    period = date.fromisoformat(package["period"])
    name = f"gpcw{period:%Y%m%d}"
    if not period < day or package["name"] != name + ".zip":
        raise ValueError("Invalid financial report period")
    path = out / package["name"]
    blob = path.read_bytes()
    verify_package(blob, package)
    if sha(path) != package["sha256"]:
        raise ValueError("Financial package changed in transit")
    with ZipFile(path) as archive:
        if archive.namelist() != [name + ".dat"] or archive.infolist()[0].file_size > 25_000_000:
            raise ValueError("Unexpected financial archive member")
        data = archive.read(name + ".dat")
    version, stamp, count, _, width, _ = struct.unpack_from("<hIHIII", data)
    begin = 20 + 11 * count
    if version != 1 or stamp != int(period.strftime("%Y%m%d")) or width != 658 * 4 or not 5000 <= count <= 6500 or begin + count * width != len(data):
        raise ValueError("Unexpected professional finance layout")
    instruments = pl.read_parquet(ROOT / "instruments/instruments.parquet")
    if "asset_type" in instruments.columns:
        instruments = instruments.filter(pl.col("asset_type") == "stock")
    identities = canonicalize_identity_frame(
        instruments.select([c for c in ("symbol", "exchange") if c in instruments.columns]), asset_type="stock"
    ).accepted.select("symbol", "exchange", "code", "asset_type", "security_key").unique()
    distinct = identities.filter(~pl.col("code").is_duplicated())
    roster = {row["code"]: row for row in distinct.to_dicts()}
    index, offsets, rows, rejected = set(), set(), [], []
    for pos in range(count):
        code, marker, offset = struct.unpack_from("<6scI", data, 20 + 11 * pos)
        code = code.decode("ascii")
        if not code.isascii() or not code.isdigit() or marker != b"\x00" or code in index or offset in offsets or not begin <= offset <= len(data) - width or (offset - begin) % width:
            raise ValueError("Invalid financial identity index")
        index.add(code)
        offsets.add(offset)
        identity = roster.get(code)
        if not identity:
            rejected.append({"code": code, "reason": "unresolved_instrument"})
            continue
        values = {}
        for field in set(fields.values()) | {314, 40, 63, 72}:
            value = struct.unpack_from("<f", data, offset + 4 * (field - 1))[0]
            values[field] = value if math.isfinite(value) and 0 < abs(value) < 1e16 else None
        announced = values[314]
        try:
            published = date.fromisoformat(f"20{int(announced):06d}") if announced and announced == int(announced) else None
        except ValueError:
            published = None
        asset, liability, equity = (values[i] for i in (40, 63, 72))
        if not published or not period < published <= day or None in (asset, liability, equity) or asset <= 0 or abs(asset - liability - equity) > max(100, asset * 5e-6):
            rejected.append({"symbol": identity["symbol"], "reason": "announcement_or_balance_unverified"})
            continue
        rows.append({**identity, "period_end": period, "announce_date": published,
                     **{f: values[n] for f, n in fields.items()}})
    save(out / "financial-quarantine.json", rejected)
    if not rows:
        raise ValueError("No financial rows satisfy the announcement and identity contract")
    return rows


def stage_finance(day: date, out: Path, package: dict | None) -> dict:
    if package is None:
        return {"status": "source_unavailable", "tables": {}}
    mapping = read(CODE / "financial-fields.json")
    rows = financial_rows(out, package, day, {f: n for table in mapping.values() for f, n in table.items()})
    frame = pl.DataFrame(rows)
    tables = {}
    for table, fields in mapping.items():
        path = ROOT / "financials" / table / "part.parquet"
        digest = sha(path)
        stored = pl.read_parquet(path)
        known = stored.filter(pl.col("period_end") == date.fromisoformat(package["period"]))
        known = known.with_columns(pl.col("symbol").str.split(".").list.first().alias("_bare"))
        fresh = frame.filter(pl.any_horizontal([pl.col(f).is_not_null() for f in fields]))
        candidates = fresh.join(known.select("_bare").unique(), left_on="code", right_on="_bare", how="anti")
        cols = set(fields) | {"symbol", "period_end", "announce_date", "code", "asset_type", "exchange", "security_key"}
        additions = candidates.select(
            pl.col(c).cast(dtype) if c in cols else pl.lit(None, dtype=dtype).alias(c) for c, dtype in stored.schema.items()
        )
        staged_path = out / f"financial-{table}.parquet"
        additions.write_parquet(staged_path)
        # A newer reported announcement for an existing period is a review
        # candidate. Source revisions retaining the old date need manual proof.
        new_dates = fresh.select("code", "announce_date").join(
            known.group_by("_bare").agg(pl.col("announce_date").max().alias("stored_date")),
            left_on="code", right_on="_bare", how="inner",
        ).filter(pl.col("announce_date") > pl.col("stored_date"))
        prior = known.filter(~pl.col("_bare").is_duplicated()).select("_bare", *fields)
        matched = fresh.select("code", *fields).join(prior, left_on="code", right_on="_bare", suffix="_stored")
        material = []
        for field in fields:
            current, stored_value = pl.col(field), pl.col(f"{field}_stored")
            material.append(
                current.is_not_null() & stored_value.is_not_null()
                & ((current - stored_value).abs() > pl.max_horizontal(pl.lit(0.02), stored_value.abs() * 0.001))
            )
        differences = matched.filter(pl.any_horizontal(material))["code"].to_list()
        tables[table] = {"rows": additions.height, "before_sha256": digest, "staged_sha256": sha(staged_path),
                         "sample_symbol": additions["symbol"][0] if additions.height else None,
                         "review_later_announcement": new_dates["code"].to_list(),
                         "review_existing_value_differences": differences}
    return {"status": "staged", "package": package, "stocks_validated": len(rows), "tables": tables}


def reconcile(failures: list[dict]) -> list[dict]:
    from app.services.minute_cross_source import collect_daily_snapshots, verify_stock_day

    def check(item):
        day = date.fromisoformat(item["date"])
        snapshots, errors = collect_daily_snapshots(item["symbol"], day)
        decision = verify_stock_day(symbol=item["symbol"], trade_date=day, local_daily=item["local"],
                                    minute_aggregate=item["minute"], external_snapshots=snapshots, source_errors=errors)
        if decision["write_allowed"]:
            decision.update(status="quarantine", write_allowed=False)
        return {**decision, "source_validation_reason": item["reason"], "actual_minutes": 0,
                "daily_evidence": snapshots, "source_errors": errors}

    with ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(check, failures))


def publish_finance(out: Path, tables: dict) -> dict:
    targets = {}
    for table, detail in tables.items():
        if detail["rows"]:
            path = ROOT / "financials" / table / "part.parquet"
            if sha(path) != detail["before_sha256"]:
                raise ValueError("Financial table changed after staging")
            staged_path = out / f"financial-{table}.parquet"
            assert sha(staged_path) == detail["staged_sha256"]
            additions = pl.read_parquet(staged_path)
            assert additions.height == detail["rows"]
            targets[table] = (path, path.stat(), additions)
    rollback = out / "rollback"
    written = []
    try:
        for table, (path, _, _) in targets.items():
            shutil.copy2(path, rollback / f"financial-{table}.parquet")
            assert sha(rollback / f"financial-{table}.parquet") == tables[table]["before_sha256"]
        for table, (path, metadata, additions) in targets.items():
            existing = pl.read_parquet(path)
            frame = pl.concat([existing, additions])
            written.append(table)
            assert _write_table(table, frame, ROOT) == frame.height
            os.chown(path, metadata.st_uid, metadata.st_gid)
            path.chmod(stat.S_IMODE(metadata.st_mode))
            assert pl.read_parquet(path).equals(frame)
    except BaseException:
        for table in reversed(written):
            path, metadata, _ = targets[table]
            temporary = path.with_name(".tdx-finance-rollback.parquet")
            shutil.copy2(rollback / f"financial-{table}.parquet", temporary)
            os.chown(temporary, metadata.st_uid, metadata.st_gid)
            temporary.replace(path)
        raise
    return {table: detail["rows"] for table, detail in tables.items()}


def publish(day: date, out: Path) -> None:
    staged = read(out / "staging.json")
    minute = partition("kline_minute", day)
    old_sha = sha(minute) if minute.exists() else None
    assert old_sha == staged["minute"]["minute_sha256"]
    assert sha(partition("kline_daily", day)) == staged["minute"]["daily_sha256"]
    entries = read(out / "staged/report.json")
    provider = StagedMinuteProvider([out / "staged"])
    provider.name = "tdx"
    assert len(entries) == len(provider.files) and {
        (item["symbol"], item["sha256"]) for item in entries
    } == {(item["symbol"], item["sha256"]) for item in provider.inventory}
    (out / "rollback").mkdir()
    if old_sha and entries:
        shutil.copy2(minute, out / "rollback/minute.parquet")
        assert sha(out / "rollback/minute.parquet") == old_sha
    minute_result = {"added_rows": 0, "stock_days": 0}
    if entries:
        repo = SimpleNamespace(store=SimpleNamespace(data_dir=ROOT), _write_lock=RLock(),
                               rebuild_views=lambda: None, clear_cache=lambda: None)
        result = repair_minute_gaps(
            repo, start_date=day, end_date=day, provider_name="tdx", source_provider=provider,
            stock_days=list(provider.files), workers=1, single_day_requests=True, bypass_failover=True,
            repair_conflicts=False,
            progress=lambda done, total, message: print(message, flush=True) if done % 250 == 0 or done == total else None,
        )
        save(out / "minute-import.json", result)
        assert result["added_rows"] == sum(item["added_rows"] for item in entries)
        assert all(result[k] == 0 for k in ("replaced_rows", "removed_rows", "updated_rows", "rejected_rows"))
        assert not result["identity_pending"] and not result["approval_required"]
        frame = pl.read_parquet(minute, columns=COLS)
        if old_sha:
            previous = pl.read_parquet(out / "rollback/minute.parquet", columns=COLS)
            assert frame.join(previous.select("symbol", "datetime").unique(), on=["symbol", "datetime"]).sort(COLS[:2]).equals(previous.sort(COLS[:2]))
        features = build_daily_features(ROOT, [row["symbol"] for row in entries], day, day)
        assert features.height == len(entries)
        assert features.select(pl.all_horizontal([pl.col(c).is_finite() for c in FEATURE_DEFAULTS]).all()).item()
        assert features.filter((pl.col("minute_bar_count") != 240) | (pl.col("minute_session_coverage") != 1)).is_empty()
        record_statuses(ROOT, [{**row, "status": "complete", "actual_minutes": 240,
                                "expected_minutes": 240, "reason": "tdx_lc1_readback"} for row in entries])
        minute_result = {"added_rows": result["added_rows"], "stock_days": len(entries)}
    decisions = read(out / "peer-evidence.json")
    if decisions:
        record_cross_source_results(ROOT, decisions)
    save(out / "minute-published.json", minute_result)
    finance = publish_finance(out, staged["financial"]["tables"])
    save(out / "published.json", {"minute": minute_result, "financial_rows_added": finance})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "stage", "publish", "reconcile"))
    parser.add_argument("day", type=date.fromisoformat)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        print(json.dumps(plan(args.day)))
    elif args.mode == "reconcile":
        print(json.dumps(reconcile(json.load(sys.stdin)), ensure_ascii=False, default=str))
    elif args.mode == "stage":
        out = args.out
        current = plan(args.day)
        manifest = read(out / "upload.json")
        assert current["status"] == "ready" and current["date"] == manifest["date"]
        assert sha(out / "minutes.jsonl.gz") == manifest["minutes_sha256"]
        minute = stage_minute(args.day, out, current)
        try:
            finance = stage_finance(args.day, out, manifest.get("financial"))
        except (ValueError, OSError, BadZipFile, struct.error, pl.exceptions.PolarsError) as exc:
            finance = {"status": "validation_failed", "error": type(exc).__name__, "tables": {}}
        save(out / "staging.json", {"minute": minute, "financial": finance})
    else:
        publish(args.day, args.out)


if __name__ == "__main__":
    main()
