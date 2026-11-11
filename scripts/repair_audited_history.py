"""Backup-first local repair and bounded online minute verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time as timer
from datetime import date, datetime, time
from pathlib import Path

import polars as pl

from app.services.daily_aliases import deduplicate_verified_aliases
from app.services.minute_repair import checked_additions, daily_reference, keyed, regular


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def replace_verified(root, output, relative, frame):
    path = root / relative
    old_hash = digest(path) if path.exists() else None
    backup = output / "backups" / relative
    if path.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists() and digest(backup) != old_hash:
            raise RuntimeError("Backup already exists for different content")
        if not backup.exists():
            shutil.copy2(path, backup)
        if digest(backup) != old_hash:
            raise RuntimeError("Backup checksum mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".repair-tmp")
    frame.write_parquet(tmp)
    if not pl.read_parquet(tmp).equals(frame):
        raise RuntimeError("Repair readback mismatch")
    if (digest(path) if path.exists() else None) != old_hash:
        raise RuntimeError("Source changed during repair")
    tmp.replace(path)
    return {"file": str(relative), "before_sha256": old_hash, "after_sha256": digest(path),
            "rows": frame.height}


def aliases(root, output, apply=False):
    results = []
    for day in ("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21", "2026-08-24"):
        rel = Path("kline_daily") / f"date={day}" / "part.parquet"
        old = pl.read_parquet(root / rel)
        new, report = deduplicate_verified_aliases(old)
        item = {"date": day, "removed": old.height - new.height,
                "unresolved": sum(r["status"] != "verified_duplicate" for r in report)}
        if apply and new.height < old.height:
            item.update(replace_verified(root, output, rel, new))
        results.append(item)
    (output / "alias-repair.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"aliases": results, "applied": apply}), flush=True)


def minute_probe(root, output):
    import httpx
    from app.config import settings
    from app.plugins.tdx.provider import TdxProvider
    from app.plugins.tushare.provider import TushareProvider, minute_error_details

    settings.data_dir = output / "provider_journal"
    pro = TushareProvider()
    pro._ensure_api()._client.timeout = httpx.Timeout(15)
    providers = [("promax", pro), ("tdx", TdxProvider())]
    cases = [("000001.SZ", "2026-09-07"), ("600000.SH", "2026-09-08"),
             ("000001.SZ", "2026-09-09"), ("000002.SZ", "2026-07-30")]
    results = []
    for name, provider in providers:
        failure_streak = 0
        for symbol, day_text in cases:
            day = date.fromisoformat(day_text)
            result = {"source": name, "symbol": symbol, "date": day_text}
            try:
                frame, issues = provider.get_minute_diagnostics(
                    [symbol], datetime.combine(day, time(9, 30)), datetime.combine(day, time(15)))
                result.update(rows=frame.height,
                              issues=[{k: i[k] for k in ("reason", "retryable", "http_status") if k in i}
                                      for i in issues])
                refs = daily_reference(pl.read_parquet(root / "kline_daily" / f"date={day}" / "part.parquet"), data_dir=root)
                accepted, reason = checked_additions(frame, frame.clear(), refs.get(symbol[:6]), day)
                result["validation"] = reason
                if reason == "accepted":
                    target = output / "minutes" / f"{symbol}_{day}_{name}.parquet"
                    target.parent.mkdir(exist_ok=True)
                    accepted.write_parquet(target)
                    result["file"] = str(target.relative_to(output))
                failure_streak = failure_streak + 1 if frame.is_empty() else 0
            except Exception as exc:
                result.update(minute_error_details(exc))
                failure_streak += 1
            results.append(result)
            (output / "minute-probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
            if failure_streak >= 3:
                break
    pro._ensure_api().close()


def import_minutes(root, output, apply=True):
    results = []
    partitions, references, originals = {}, {}, {}
    for staged in sorted((output / "minutes").glob("*.parquet")):
        frame = pl.read_parquet(staged)
        dates = frame["datetime"].dt.date().unique().to_list()
        codes = keyed(frame)["_code"].unique().to_list()
        if len(dates) != 1 or len(codes) != 1:
            raise ValueError("Only verified single stock-day staging is accepted")
        day, code = dates[0], codes[0]
        if day not in references:
            references[day] = daily_reference(pl.read_parquet(root / "kline_daily" / f"date={day}" / "part.parquet"), data_dir=root)
        refs = references[day]
        candidate, reason = checked_additions(frame, frame.clear(), refs.get(code), day)
        result = {"symbol": code, "date": str(day), "validation": reason}
        if reason != "accepted":
            results.append(result)
            continue
        rel = Path("kline_minute") / f"date={day}" / "part.parquet"
        if day not in partitions:
            partitions[day] = pl.read_parquet(root / rel) if (root / rel).exists() else candidate.clear()
            originals[day] = digest(root / rel) if (root / rel).exists() else None
        full = partitions[day]
        old = keyed(full).filter(pl.col("_code") == code).drop("_code") if not full.is_empty() else candidate.clear()
        additions, overlap = checked_additions(candidate, old, refs.get(code), day)
        if overlap == "already_complete":
            result["action"] = "unchanged"
        elif overlap == "accepted":
            partitions[day] = pl.concat([full, additions], how="diagonal_relaxed")
            result.update(action="append", added=additions.height)
        elif overlap == "existing_values_conflict":
            _, old_status = checked_additions(old, old.clear(), refs.get(code), day)
            if old_status == "accepted":
                result.update(action="preserve_valid_existing", old_validation=old_status)
            else:
                rest = keyed(full).filter(pl.col("_code") != code).drop("_code")
                partitions[day] = pl.concat([rest, candidate], how="diagonal_relaxed")
                result.update(action="replace_invalid",
                              old_validation=old_status, added=candidate.height - old.height)
        else:
            result.update(action="deferred", reason=overlap)
        results.append(result)
    publications = []
    for day, full in sorted(partitions.items()):
        if not any(r["date"] == str(day) and r.get("action") in {"append", "replace_invalid"} for r in results):
            continue
        rel = Path("kline_minute") / f"date={day}" / "part.parquet"
        if (digest(root / rel) if (root / rel).exists() else None) != originals[day]:
            raise RuntimeError("Source changed before publishing minute repairs")
        if not apply:
            publications.append({"file": str(rel), "rows": full.height, "before_sha256": originals[day]})
            continue
        publications.append(replace_verified(root, output, rel, full.sort(["symbol", "datetime"])))
        (output / "minute-publications.json").write_text(json.dumps(publications, indent=2), encoding="utf-8")
    (output / ("minute-import.json" if apply else "minute-import-plan.json")).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"stock_days_checked": len(results), "stock_days_changed": sum(r.get("action") in {"append", "replace_invalid"} for r in results),
                      "rows_added": sum(r.get("added", 0) for r in results), "partitions_published": len(publications) if apply else 0,
                      "dry_run": not apply}), flush=True)


def stage_recent_minutes(root, output, max_stock_days, minute_api="stk_mins"):
    import httpx
    from app.config import settings
    from app.indicators.pipeline import _canonicalize_and_dedupe_daily, _normalize_raw_volume_unit
    from app.plugins.tushare.provider import TushareProvider, minute_error_details
    from app.plugins.tushare.minute_contract import normalize_history

    settings.data_dir = output / "provider_journal"
    pro = TushareProvider()
    api = pro._ensure_api()
    api._client.timeout = httpx.Timeout(20)
    results, plans, requested = [], [], 0
    failures = 0
    for text in ("2026-09-07", "2026-09-08", "2026-09-09"):
        day = date.fromisoformat(text)
        daily = pl.read_parquet(root / "kline_daily" / f"date={day}" / "part.parquet")
        daily = _normalize_raw_volume_unit(_canonicalize_and_dedupe_daily(daily)).filter(
            (pl.col("volume") > 0) & (pl.col("amount") > 0)
            & pl.col("symbol").str.contains(r"^(?:00|30|60|68|4|8|92)\d+"))
        minute_path = root / "kline_minute" / f"date={day}" / "part.parquet"
        counts = (regular(keyed(pl.read_parquet(minute_path)), day)
                  .group_by("_code").agg(pl.col("datetime").n_unique().alias("minutes")))
        complete = set(counts.filter(pl.col("minutes") == 240)["_code"].to_list())
        missing = sorted(set(daily["symbol"].to_list()) - complete)
        plans.append({"date": text, "missing_or_partial": len(missing)})
        daily_pd = daily.select(
            pl.when(pl.col("symbol").str.starts_with("6")).then(pl.col("symbol") + ".SH")
            .when(pl.col("symbol").str.contains(r"^(4|8|92)")).then(pl.col("symbol") + ".BJ")
            .otherwise(pl.col("symbol") + ".SZ").alias("ts_code"),
            pl.col("date").dt.strftime("%Y%m%d").alias("trade_date"),
            "open", "high", "low", "close", (pl.col("volume") / 100).alias("vol"),
            (pl.col("amount") / 1000).alias("amount"),
        ).to_pandas()
        batch_size = 1 if minute_api == "a_share_mins" else 10
        for index in range(0, len(missing), batch_size):
            if requested >= max_stock_days or failures >= 3:
                break
            codes = missing[index:index + min(batch_size, max_stock_days - requested)]
            symbols = sorted(daily_pd[daily_pd.ts_code.str[:6].isin(codes)].ts_code.tolist())
            # One closed day x ten stocks fits below the gateway's 5000-row cap.
            requested += len(symbols)
            try:
                raw = getattr(api, minute_api)(ts_code=",".join(symbols), freq="1MIN" if minute_api == "a_share_mins" else "1min", limit=5000,
                                   start_date=text + " 09:25:00", end_date=text + " 15:10:00")
                if raw.empty or "ts_code" not in raw:
                    failures += 1
                    results.append({"date": text, "symbols": symbols, "reason": "source_empty"})
                    continue
                if not set(raw.ts_code.unique()) <= set(symbols):
                    raise ValueError("Unexpected stock in batch response")
                failures = 0
                for symbol in symbols:
                    part = raw[raw.ts_code.eq(symbol)]
                    if part.empty:
                        results.append({"date": text, "symbol": symbol, "reason": "source_empty"})
                        continue
                    outcomes = []
                    accepted = normalize_history(part, daily_pd, symbol,
                        output / "gateway-batch-evidence", outcomes=outcomes)
                    results.extend({k: v for k, v in row.items() if k != "evidence"} for row in outcomes)
                    if not accepted.is_empty():
                        folder = output / "minutes"
                        folder.mkdir(exist_ok=True)
                        accepted.write_parquet(folder / f"{symbol}_{text}_batch.parquet")
            except Exception as exc:
                detail = minute_error_details(exc)
                results.append({"date": text, "symbols": symbols, **detail})
                failures += 1
                if detail.get("reason") in {"permission_denied", "rate_limited", "source_pending"}:
                    failures = 3
            finally:
                (output / "recent-minute-stage.json").write_text(json.dumps(
                    {"plans": plans, "requested_stock_days": requested, "results": results}, indent=2), encoding="utf-8")
                print(json.dumps({"requested_stock_days": requested, "recent": results[-len(symbols):]}), flush=True)
            timer.sleep(1)
    api.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["aliases", "minute-probe", "minute-import", "minute-plan", "minute-stage"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--offline-confirmed", action="store_true")
    parser.add_argument("--minute-api", choices=["stk_mins", "a_share_mins"], default="stk_mins")
    parser.add_argument("--max-stock-days", type=int, default=300)
    args = parser.parse_args()
    if args.apply and not args.offline_confirmed:
        raise ValueError("Stop all production writers before applying audited repairs")
    root, output = args.data_dir.resolve(), args.output.resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Backups and staging must be separate from production data")
    output.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.CRITICAL)
    if args.phase == "aliases":
        aliases(root, output, args.apply)
    elif args.phase == "minute-probe":
        minute_probe(root, output)
    elif args.phase == "minute-stage":
        stage_recent_minutes(root, output, args.max_stock_days, args.minute_api)
    elif args.phase == "minute-plan":
        import_minutes(root, output, apply=False)
    elif args.apply:
        import_minutes(root, output)
    else:
        raise ValueError("Minute import requires --apply --offline-confirmed")


if __name__ == "__main__":
    main()
