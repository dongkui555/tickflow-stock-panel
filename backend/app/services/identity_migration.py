"""Read-only inventory and migration bookkeeping for canonical identities.

This module deliberately keeps the first migration step non-destructive. It
counts the current identity shapes and records enough metadata to decide
whether a later, explicitly approved batch migration is safe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import polars as pl
import pyarrow.parquet as pq
import duckdb

from app.services.identity_schema import (
    CANONICAL_VERSION,
    migration_lock,
    publish_identity_version,
)
from app.services.security_identity import (
    IdentityBatch,
    build_instrument_lookup,
    canonicalize_identity_frame,
)


QUALIFIED_SYMBOL = re.compile(r"^\d{6}\.(SH|SZ|BJ)$", re.IGNORECASE)
BARE_SYMBOL = re.compile(r"^\d{6}$")
DATA_SUFFIXES = {".parquet", ".sqlite", ".sqlite3", ".json", ".jsonl", ".csv"}
EXCLUDED_DIRS = {"migration_backups", ".git", ".superpowers"}
DEFAULT_DATASETS = (
    "instruments", "instruments_index", "instruments_etf", "instruments_history",
    "instruments_lifecycle", "kline_daily", "kline_minute", "kline_daily_enriched",
    "kline_index_daily", "kline_etf_daily", "moneyflow", "auction_open", "financials",
    "news", "adj_factor", "user_data",
)


@dataclass
class FileInventory:
    path: str
    dataset: str
    size_bytes: int
    row_count: int = 0
    has_symbol: bool = False
    bare_symbol_rows: int = 0
    qualified_symbol_rows: int = 0
    invalid_symbol_rows: int = 0


@dataclass
class DatasetStats:
    dataset: str
    file_count: int = 0
    row_count: int = 0
    bare_symbol_rows: int = 0
    qualified_symbol_rows: int = 0
    invalid_symbol_rows: int | None = 0
    sampled_symbol_rows: int = 0
    bytes: int = 0


@dataclass
class MigrationManifest:
    schema_version: str
    generated_at: str
    root: str
    file_count: int
    total_bytes: int
    estimated_backup_bytes: int
    free_bytes: int
    identity_scan: str
    sample_rows_per_file: int
    sample_files_per_dataset: int
    files: list[FileInventory] = field(default_factory=list)
    dataset_stats: dict[str, DatasetStats] = field(default_factory=dict)


@dataclass
class MigrationResult:
    migration_id: str
    replaced_files: int = 0
    skipped_files: int = 0
    quarantined_rows: int = 0
    accepted_rows: int = 0
    backup_root: str = ""
    journal_path: str = ""
    version_path: str = ""
    sqlite_backups: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


STOCK_DATASETS = frozenset({
    "instruments", "instruments_history", "instruments_lifecycle",
    "kline_daily", "kline_minute", "kline_daily_enriched", "moneyflow",
    "auction_open", "financials", "adj_factor",
})
INDEX_DATASETS = frozenset({"instruments_index", "kline_index_daily", "kline_index_enriched"})
ETF_DATASETS = frozenset({"instruments_etf", "kline_etf_daily", "kline_etf_enriched", "kline_etf_minute"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_type_for_dataset(dataset: str) -> str | None:
    if dataset in STOCK_DATASETS:
        return "stock"
    if dataset in INDEX_DATASETS:
        return "index"
    if dataset in ETF_DATASETS:
        return "etf"
    return None


def _dimension_frames(root: Path, dataset: str) -> list[pl.DataFrame]:
    directory = root / dataset
    if not directory.exists():
        return []
    frames: list[pl.DataFrame] = []
    for path in sorted(directory.rglob("*.parquet")):
        try:
            frame = pl.read_parquet(path)
        except Exception:
            continue
        if "code" in frame.columns or "symbol" in frame.columns:
            frames.append(frame)
    return frames


def _build_lookups(root: Path) -> dict[str, dict[str, str]]:
    return {
        "stock": build_instrument_lookup(
            _dimension_frames(root, "instruments")
            + _dimension_frames(root, "instruments_history")
            + _dimension_frames(root, "instruments_lifecycle")
        ),
        "index": build_instrument_lookup(_dimension_frames(root, "instruments_index")),
        "etf": build_instrument_lookup(_dimension_frames(root, "instruments_etf")),
    }


def _write_quarantine(path: Path, batch: IdentityBatch) -> Path | None:
    if batch.quarantined.is_empty():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    batch.quarantined.write_parquet(path)
    return path


def _backup_sqlite(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_conn = sqlite3.connect(str(source))
    target_conn = sqlite3.connect(str(target))
    try:
        source_conn.backup(target_conn)
        target_conn.commit()
    finally:
        target_conn.close()
        source_conn.close()


def _normalize_exchange_expr(expr: pl.Expr) -> pl.Expr:
    text = expr.cast(pl.String).str.strip_chars().str.to_uppercase().str.replace(r"^\.", "")
    return (
        pl.when(text == "SSE").then(pl.lit("SH"))
        .when(text == "SZSE").then(pl.lit("SZ"))
        .when(text == "BSE").then(pl.lit("BJ"))
        .otherwise(text)
    )


def _canonicalize_frame_columnar(
    frame: pl.DataFrame,
    *,
    asset_type: str,
    lookup: dict[str, str],
) -> IdentityBatch:
    """Canonicalize one parquet partition without materializing row dicts."""
    if frame.is_empty():
        return IdentityBatch(frame, frame, {})
    if "symbol" not in frame.columns:
        quarantine = frame.with_columns(
            pl.lit("").alias("original_symbol"),
            pl.lit("symbol_column_missing").alias("identity_reason"),
        )
        return IdentityBatch(frame.clear(), quarantine, {"symbol_column_missing": frame.height})

    mapping = pl.DataFrame(
        {
            "__code": list(lookup),
            "__mapped_symbol": list(lookup.values()),
        },
        schema={"__code": pl.String, "__mapped_symbol": pl.String},
    ).with_columns(
        pl.col("__mapped_symbol").str.extract(r"\.([A-Za-z]+)$", 1).str.to_uppercase().alias("__mapped_exchange")
    )
    raw = pl.col("symbol").cast(pl.String).str.strip_chars()
    digits = raw.str.replace_all(r"\D", "")
    code = (
        pl.when((digits.str.len_chars() > 0) & (digits.str.len_chars() <= 6))
        .then(digits.str.zfill(6))
        .otherwise(pl.lit(None).cast(pl.String))
    )
    suffix_exchange = raw.str.extract(r"\.([A-Za-z]+)$", 1).str.to_uppercase()
    prefix_exchange = raw.str.extract(r"^(SH|SZ|BJ|SSE|SZSE|BSE)[._-]?\d{1,6}$", 1).str.to_uppercase()
    explicit_exchange = pl.when(suffix_exchange.is_not_null()).then(suffix_exchange).otherwise(prefix_exchange)
    explicit_exchange = _normalize_exchange_expr(explicit_exchange)

    enriched = frame.with_row_index("__identity_row").with_columns(
        raw.alias("__raw_symbol"),
        code.alias("__code"),
        explicit_exchange.alias("__explicit_exchange"),
    )
    if "exchange" in enriched.columns:
        enriched = enriched.with_columns(
            _normalize_exchange_expr(pl.col("exchange")).alias("__source_exchange")
        )
    else:
        enriched = enriched.with_columns(pl.lit(None).cast(pl.String).alias("__source_exchange"))
    enriched = enriched.join(mapping, on="__code", how="left")
    chosen_exchange = pl.coalesce(
        [pl.col("__mapped_exchange"), pl.col("__explicit_exchange"), pl.col("__source_exchange")]
    ).alias("__chosen_exchange")
    enriched = enriched.with_columns(chosen_exchange)
    structural_exchange = (
        pl.when(pl.col("__code").str.starts_with("60") | pl.col("__code").str.starts_with("68"))
        .then(pl.lit("SH"))
        .when(
            pl.col("__code").str.starts_with("00")
            | pl.col("__code").str.starts_with("30")
        )
        .then(pl.lit("SZ"))
        .when(
            pl.col("__code").str.starts_with("43")
            | pl.col("__code").str.starts_with("83")
            | pl.col("__code").str.starts_with("87")
            | pl.col("__code").str.starts_with("88")
            | pl.col("__code").str.starts_with("92")
        )
        .then(pl.lit("BJ"))
        .otherwise(pl.lit(None).cast(pl.String))
    )
    reason = (
        pl.when(pl.col("__code").is_null()).then(pl.lit("code_unparseable"))
        .when(pl.col("__chosen_exchange").is_null()).then(pl.lit("exchange_unknown"))
        .when(
            pl.col("__explicit_exchange").is_not_null()
            & pl.col("__mapped_exchange").is_not_null()
            & (pl.col("__explicit_exchange") != pl.col("__mapped_exchange"))
        ).then(pl.lit("dimension_exchange_conflict"))
        .when(
            pl.col("__source_exchange").is_not_null()
            & pl.col("__mapped_exchange").is_not_null()
            & (pl.col("__source_exchange") != pl.col("__mapped_exchange"))
        ).then(pl.lit("source_exchange_conflict"))
        .when(
            (asset_type == "stock")
            & structural_exchange.is_not_null()
            & (pl.col("__chosen_exchange") != structural_exchange)
        ).then(pl.lit("code_exchange_conflict"))
        .otherwise(pl.lit(None).cast(pl.String))
        .alias("__identity_reason")
    )
    enriched = enriched.with_columns(reason)
    accepted = enriched.filter(pl.col("__identity_reason").is_null()).with_columns(
        pl.lit(asset_type).alias("asset_type"),
        pl.col("__chosen_exchange").alias("exchange"),
        pl.col("__code").alias("code"),
        pl.coalesce(
            [
                pl.col("__mapped_symbol"),
                pl.concat_str([pl.col("__code"), pl.col("__chosen_exchange")], separator="."),
            ]
        ).alias("symbol"),
        pl.concat_str(
            [pl.lit(asset_type), pl.col("__chosen_exchange"), pl.col("__code")],
            separator="|",
        ).alias("security_key"),
    )
    quarantined = enriched.filter(pl.col("__identity_reason").is_not_null()).with_columns(
        pl.col("__raw_symbol").alias("original_symbol"),
        pl.col("__identity_reason").alias("identity_reason"),
    )
    internal = [
        "__raw_symbol", "__code", "__explicit_exchange", "__source_exchange",
        "__mapped_symbol", "__mapped_exchange", "__chosen_exchange",
        "__identity_reason", "__identity_row",
    ]
    accepted = accepted.drop([column for column in internal if column in accepted.columns])
    quarantined = quarantined.drop([
        column for column in internal if column in quarantined.columns and column != "__identity_row"
    ]).rename({"__identity_row": "identity_row"})
    reason_counts = (
        quarantined.group_by("identity_reason").len().to_dicts()
        if not quarantined.is_empty() else []
    )
    return IdentityBatch(
        accepted,
        quarantined,
        {str(row["identity_reason"]): int(row["len"]) for row in reason_counts},
    )


def apply_migration(
    root: Path,
    manifest: MigrationManifest,
    *,
    migration_id: str,
) -> MigrationResult:
    """Apply a reviewed full-scan manifest with backups and an audit journal.

    The function is intentionally fail-closed: it accepts only a full-scan
    manifest, obtains a single migration lock, and never replaces a source
    parquet file before its backup has been written successfully. Rows whose
    security identity cannot be resolved are written to quarantine and are
    excluded from the active replacement file.
    """
    root = Path(root).resolve()
    if manifest.identity_scan != "full":
        raise ValueError("apply_migration requires a full-scan manifest")
    if Path(manifest.root).resolve() != root:
        raise ValueError("manifest root does not match migration root")
    if manifest.estimated_backup_bytes > manifest.free_bytes:
        raise OSError(
            f"insufficient disk space for backup: need {manifest.estimated_backup_bytes}, "
            f"free {manifest.free_bytes}"
        )

    backup_root = root / "migration_backups" / migration_id
    quarantine_root = backup_root / "quarantine"
    journal_path = backup_root / "migration.jsonl"
    result = MigrationResult(
        migration_id=migration_id,
        backup_root=str(backup_root),
        journal_path=str(journal_path),
    )
    backup_root.mkdir(parents=True, exist_ok=False)
    lookups = _build_lookups(root)

    with migration_lock(root):
        for relative in (
            "user_data/minute_gap_queue.sqlite",
            "user_data/minute_repair_authorizations.sqlite",
        ):
            source = root / relative
            if not source.exists():
                continue
            target = backup_root / relative
            _backup_sqlite(source, target)
            result.sqlite_backups.append(str(target.relative_to(root)))
        with journal_path.open("w", encoding="utf-8") as journal:
            for backup in result.sqlite_backups:
                journal.write(json.dumps({
                    "status": "sqlite_backup",
                    "path": backup,
                }, ensure_ascii=False) + "\n")
            for item in manifest.files:
                source = root / item.path
                dataset = item.dataset
                asset_type = _asset_type_for_dataset(dataset)
                entry: dict[str, object] = {
                    "path": item.path,
                    "dataset": dataset,
                    "asset_type": asset_type,
                    "source_sha256_before": None,
                    "backup_path": None,
                    "status": "skipped",
                    "accepted_rows": 0,
                    "quarantined_rows": 0,
                    "reasons": {},
                }
                if asset_type is None or source.suffix.lower() != ".parquet" or not item.has_symbol:
                    result.skipped_files += 1
                    journal.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    continue
                if not source.exists():
                    raise FileNotFoundError(source)

                frame = pl.read_parquet(source)
                batch = _canonicalize_frame_columnar(
                    frame,
                    asset_type=asset_type,
                    lookup=lookups[asset_type],
                )
                backup = backup_root / item.path
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, backup)
                entry["source_sha256_before"] = _sha256(source)
                entry["backup_path"] = str(backup.relative_to(root))
                if not backup.exists():
                    raise OSError(f"backup was not created: {backup}")

                if batch.accepted.is_empty():
                    quarantine = _write_quarantine(
                        quarantine_root / f"{item.path}.parquet",
                        batch,
                    )
                    entry["status"] = "quarantined"
                    entry["quarantine_path"] = str(quarantine.relative_to(root)) if quarantine else None
                    entry["quarantined_rows"] = batch.quarantined.height
                    entry["reasons"] = batch.reasons
                    result.quarantined_rows += batch.quarantined.height
                    result.skipped_files += 1
                    journal.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    continue

                quarantine = _write_quarantine(
                    quarantine_root / f"{item.path}.parquet",
                    batch,
                )
                temporary = source.with_name(f".{source.name}.{migration_id}.tmp")
                batch.accepted.write_parquet(temporary)
                temporary.replace(source)
                entry["status"] = "replaced"
                entry["accepted_rows"] = batch.accepted.height
                entry["quarantined_rows"] = batch.quarantined.height
                entry["reasons"] = batch.reasons
                entry["quarantine_path"] = str(quarantine.relative_to(root)) if quarantine else None
                entry["source_sha256_after"] = _sha256(source)
                result.replaced_files += 1
                result.accepted_rows += batch.accepted.height
                result.quarantined_rows += batch.quarantined.height
                journal.write(json.dumps(entry, ensure_ascii=False) + "\n")

        version_path = publish_identity_version(root, version=CANONICAL_VERSION)
        result.version_path = str(version_path)
    return result


def _iter_files(root: Path, datasets: Iterable[str]) -> Iterable[Path]:
    roots = [root / dataset for dataset in datasets]
    for data_root in roots:
        if not data_root.exists():
            continue
        for path in data_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in DATA_SUFFIXES:
                continue
            if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
                continue
            yield path
    # Keep top-level manifests/configuration files in the inventory without
    # recursively walking unrelated raw archives.
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in DATA_SUFFIXES:
            continue
        yield path


def _dataset_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else "root"


def _parquet_inventory(root: Path, path: Path) -> FileInventory:
    item = FileInventory(
        path=str(path.relative_to(root)),
        dataset=_dataset_name(root, path),
        size_bytes=path.stat().st_size,
    )
    metadata = pq.ParquetFile(path).metadata
    item.row_count = int(metadata.num_rows)
    columns = set(metadata.schema.names)
    item.has_symbol = "symbol" in columns
    return item


def _file_inventory(root: Path, path: Path) -> FileInventory:
    if path.suffix.lower() == ".parquet":
        return _parquet_inventory(root, path)
    return FileInventory(
        path=str(path.relative_to(root)),
        dataset=_dataset_name(root, path),
        size_bytes=path.stat().st_size,
    )


def _count_symbols(path: Path, *, sample_rows: int | None) -> tuple[int, int]:
    parquet = pq.ParquetFile(path)
    qualified = bare = seen = 0
    for batch in parquet.iter_batches(columns=["symbol"], batch_size=100_000):
        values = batch.column("symbol").to_pylist()
        if sample_rows is not None:
            values = values[: max(0, sample_rows - seen)]
        for value in values:
            text = str(value or "")
            qualified += bool(QUALIFIED_SYMBOL.fullmatch(text))
            bare += bool(BARE_SYMBOL.fullmatch(text))
        seen += len(values)
        if sample_rows is not None and seen >= sample_rows:
            break
    return qualified, bare


def inventory(
    root: Path,
    *,
    full_scan: bool = False,
    sample_rows: int = 10_000,
    sample_files_per_dataset: int = 8,
    datasets: Iterable[str] | None = None,
) -> MigrationManifest:
    """Build a read-only manifest without changing any source file."""
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    selected_datasets = tuple(datasets or DEFAULT_DATASETS)
    files = [_file_inventory(root, path) for path in sorted(_iter_files(root, selected_datasets))]
    stats: dict[str, DatasetStats] = {}
    for item in files:
        stat = stats.setdefault(item.dataset, DatasetStats(dataset=item.dataset))
        stat.file_count += 1
        stat.row_count += item.row_count
        stat.bare_symbol_rows += item.bare_symbol_rows
        stat.qualified_symbol_rows += item.qualified_symbol_rows
        stat.invalid_symbol_rows += item.invalid_symbol_rows
        stat.bytes += item.size_bytes
    # The default is a bounded sample per file. A full scan is intentionally
    # opt-in because decompression of the existing minute archive can take
    # more than ten minutes on a Docker bind mount.
    for dataset, stat in stats.items():
        parquet_files = [item for item in files if item.dataset == dataset and item.path.lower().endswith(".parquet")]
        if not parquet_files or not any(item.has_symbol for item in parquet_files):
            continue
        if full_scan:
            parquet_glob = str(root / dataset / "**" / "*.parquet")
            result = duckdb.execute(
                """
                SELECT
                    count(*) FILTER (WHERE regexp_matches(CAST(symbol AS VARCHAR), ?)) AS qualified,
                    count(*) FILTER (WHERE regexp_matches(CAST(symbol AS VARCHAR), ?)) AS bare
                FROM read_parquet(?, union_by_name=true)
                """,
                [QUALIFIED_SYMBOL.pattern, BARE_SYMBOL.pattern, parquet_glob],
            ).fetchone()
            stat.qualified_symbol_rows = int(result[0] or 0)
            stat.bare_symbol_rows = int(result[1] or 0)
        else:
            stat.qualified_symbol_rows = 0
            stat.bare_symbol_rows = 0
            selected = parquet_files if len(parquet_files) <= sample_files_per_dataset else (
                parquet_files[: sample_files_per_dataset // 2]
                + parquet_files[-(sample_files_per_dataset // 2):]
            )
            for item in selected:
                qualified, bare = _count_symbols(root / item.path, sample_rows=sample_rows)
                stat.qualified_symbol_rows += qualified
                stat.bare_symbol_rows += bare
        if full_scan:
            stat.invalid_symbol_rows = max(0, stat.row_count - stat.qualified_symbol_rows - stat.bare_symbol_rows)
            stat.sampled_symbol_rows = stat.row_count
        else:
            stat.invalid_symbol_rows = None
            stat.sampled_symbol_rows = min(
                stat.row_count,
                sample_rows * min(len(parquet_files), sample_files_per_dataset),
            )
    usage = shutil.disk_usage(root)
    total_bytes = sum(item.size_bytes for item in files)
    return MigrationManifest(
        schema_version="canonical-identity-inventory-v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        root=str(root),
        file_count=len(files),
        total_bytes=total_bytes,
        estimated_backup_bytes=total_bytes,
        free_bytes=usage.free,
        identity_scan="full" if full_scan else "sampled",
        sample_rows_per_file=0 if full_scan else sample_rows,
        sample_files_per_dataset=0 if full_scan else sample_files_per_dataset,
        files=files,
        dataset_stats=stats,
    )


def write_manifest(manifest: MigrationManifest, target: Path) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory canonical security identity migration inputs")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="scan only; no source writes")
    parser.add_argument("--full-scan", action="store_true", help="scan every symbol row; slower but required before apply")
    parser.add_argument("--dataset", action="append", dest="datasets", help="limit scan; repeat for multiple datasets")
    parser.add_argument("--apply", action="store_true", help="reserved for the approved migration phase")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.apply and not args.full_scan:
        raise SystemExit("refusing to apply without --full-scan")
    if not args.dry_run and not args.apply:
        raise SystemExit("refusing to scan without --dry-run")
    manifest = inventory(args.data_dir, full_scan=args.full_scan, datasets=args.datasets)
    target = args.report_dir / ("manifest.pre-apply.json" if args.apply else "manifest.json")
    write_manifest(manifest, target)
    if args.apply:
        migration_id = datetime.now(timezone.utc).strftime("identity-%Y%m%dT%H%M%SZ")
        result = apply_migration(args.data_dir, manifest, migration_id=migration_id)
        write_manifest(manifest, args.report_dir / f"{migration_id}.manifest.json")
        print(json.dumps({"apply": asdict(result), "manifest": str(target)}, ensure_ascii=False))
        return 0
    stats = list(manifest.dataset_stats.values())
    summary = {
        "file_count": manifest.file_count,
        "total_bytes": manifest.total_bytes,
        "estimated_backup_bytes": manifest.estimated_backup_bytes,
        "free_bytes": manifest.free_bytes,
        "identity_scan": manifest.identity_scan,
        "bare_symbol_rows": sum(item.bare_symbol_rows for item in stats),
        "qualified_symbol_rows": sum(item.qualified_symbol_rows for item in stats),
        "invalid_symbol_rows": (
            sum(item.invalid_symbol_rows for item in stats if item.invalid_symbol_rows is not None)
            if manifest.identity_scan == "full" else None
        ),
        "report": str(target),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
