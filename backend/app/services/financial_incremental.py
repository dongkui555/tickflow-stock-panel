"""Market-wide financial announcements with bounded catch-up and preserved history."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import date, timedelta

import polars as pl

from app.parquet import atomic_write_parquet


def sync_announcements(repo, today: date, emit) -> dict:
    from app.data_providers.custom import get_provider
    from app.services.financial_sync import _financial_provider_name
    from app.services.tushare_ext_sync import _temporary_promax_timeout

    root = repo.store.data_dir
    state_path = root / "user_data" / "financial_incremental.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    provider = get_provider(_financial_provider_name())
    fetch = getattr(provider, "get_financial_updates", None)
    if fetch is None:
        return {
            "rows": 0,
            "unavailable": ["Financial provider has no market-wide announcement endpoint"],
        }
    api = getattr(provider, "_ensure_api", lambda: None)()
    result = {"rows": 0, "unavailable": [], "tables": {}}
    run_id = "financial_tail_" + uuid.uuid4().hex[:12]
    with _temporary_promax_timeout(api, 8):
        for table in ("metrics", "income", "balance_sheet", "cash_flow", "shares"):
            latest = (
                date.fromisoformat(state[table]) if state.get(table) else today - timedelta(days=3)
            )
            begin = max(latest, today - timedelta(days=14))
            if latest < begin:
                result["unavailable"].append(
                    f"{table}: older announcements require history backfill"
                )
            frames, good_days = [], []
            for offset in range((today - begin).days + 1):
                day = begin + timedelta(days=offset)
                emit("sync_financial_tail", 90, f"全市场财务公告 {table} {day}")
                try:
                    frame = fetch(table, day)
                    if table == "shares" and frame.is_empty() and (root / "kline_daily" / f"date={day}").exists():
                        raise ValueError("shares snapshot missing for a locally recorded market day")
                    if not frame.is_empty():
                        frames.append(frame)
                    good_days.append(day)
                except Exception as exc:
                    result["unavailable"].append(f"{table} {day}: {type(exc).__name__}")
                    break
            if frames:
                incoming = pl.concat(frames, how="diagonal_relaxed")
                path = root / "financials" / table / "part.parquet"
                with repo._write_lock:
                    old = pl.read_parquet(path) if path.exists() else incoming.clear()
                    merged = pl.concat([old, incoming], how="diagonal_relaxed")
                    keys = [
                        c for c in ("symbol", "period_end", "announce_date") if c in merged.columns
                    ]
                    merged = merged.unique(subset=keys, keep="last").sort(keys)
                    if path.exists():
                        backup = root / "import_backups" / run_id / table / "part.parquet"
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, backup)
                    atomic_write_parquet(merged, path)
                    if not pl.read_parquet(path).equals(merged):
                        if not old.is_empty():
                            atomic_write_parquet(old, path)
                        raise IOError("financial readback mismatch")
                result["rows"] += incoming.height
            path = root / "financials" / table / "part.parquet"
            if not path.exists():
                result["unavailable"].append(f"{table}: no historical initialization")
            if good_days:
                state[table] = str(max(good_days))
            result["tables"][table] = {
                "checked_through": state.get(table),
                "responses": len(frames),
            }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(state_path)
    report = root / "import_reports" / run_id
    report.mkdir(parents=True)
    (report / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
