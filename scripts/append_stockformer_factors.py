"""Append missing StockFormer prediction-factor dates without fabricating labels.

This is deliberately separate from model training.  It refuses to write a
requested date when the prediction service returns another date, which avoids
silently storing a prior trading session under a new date.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl


FACTOR_COLUMNS = [
    "symbol",
    "date",
    "pred_ret_1d",
    "pred_ret_raw_1d",
    "target_1d",
    "pred_ret_3d",
    "pred_ret_raw_3d",
    "target_3d",
    "pred_ret_5d",
    "pred_ret_raw_5d",
    "target_5d",
    "pred_ret_10d",
    "pred_ret_raw_10d",
    "target_10d",
]


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current = date.fromordinal(current.toordinal() + 1)


def _factor_rows(prediction: dict, target_date: date) -> pl.DataFrame:
    rows = prediction.get("rows") or []
    if not rows:
        return pl.DataFrame(schema={column: pl.Utf8 for column in FACTOR_COLUMNS})
    values = []
    for row in rows:
        item = {column: row.get(column) for column in FACTOR_COLUMNS}
        item["date"] = target_date
        values.append(item)
    frame = pl.DataFrame(values)
    for column in FACTOR_COLUMNS:
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None).alias(column))
    return frame.select(FACTOR_COLUMNS)


def append_factors(data_dir: Path, model_id: str, start: date, end: date) -> dict:
    from app.services.stockformer import _model_meta, _resolve_artifact_path, predict

    if start > end:
        raise ValueError("开始日期必须早于或等于结束日期")
    meta = _model_meta(data_dir, model_id)
    if not meta:
        raise ValueError(f"模型不存在: {model_id}")
    factor_path = _resolve_artifact_path(data_dir, meta.get("factor_file"))
    if factor_path is None:
        raise ValueError("模型缺少样本外因子文件路径")
    factor_path.parent.mkdir(parents=True, exist_ok=True)
    if factor_path.exists():
        existing = pl.read_parquet(factor_path).select(FACTOR_COLUMNS)
    else:
        existing = pl.DataFrame(schema={
            "symbol": pl.String,
            "date": pl.Date,
            **{column: pl.Float64 for column in FACTOR_COLUMNS[2:]},
        })
    if existing.select(pl.struct(["symbol", "date"]).is_duplicated().any()).item():
        raise ValueError(f"因子文件存在重复 symbol/date 键: {factor_path}")

    existing_dates = set(existing["date"].unique().to_list()) if not existing.is_empty() else set()
    added_dates: list[str] = []
    skipped_dates: list[str] = []
    added_frames: list[pl.DataFrame] = []
    for requested in _date_range(start, end):
        if requested in existing_dates:
            skipped_dates.append(requested.isoformat())
            continue
        prediction = predict(
            data_dir,
            model_id=model_id,
            as_of=requested,
            limit=0,
            write_artifact=False,
        )
        returned = prediction.get("as_of")
        if prediction.get("error") or returned != requested.isoformat():
            raise RuntimeError(
                f"{requested.isoformat()} 因子未追加: "
                f"{prediction.get('error') or f'预测日期返回 {returned}'}"
            )
        frame = _factor_rows(prediction, requested)
        if frame.is_empty():
            raise RuntimeError(f"{requested.isoformat()} 预测结果为空，拒绝写入因子")
        added_frames.append(frame)
        added_dates.append(requested.isoformat())

    combined = pl.concat([existing, *added_frames], how="diagonal_relaxed")
    if combined.select(pl.struct(["symbol", "date"]).is_duplicated().any()).item():
        raise ValueError("追加后发现重复 symbol/date 键，拒绝替换因子文件")
    if added_frames:
        combined = combined.select(FACTOR_COLUMNS).sort(["date", "symbol"])
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".parquet", prefix=f".{factor_path.name}.",
            dir=factor_path.parent, delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        try:
            combined.write_parquet(temporary_path)
            os.replace(temporary_path, factor_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    report = {
        "model_id": model_id,
        "factor_file": str(factor_path),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "added_dates": added_dates,
        "skipped_existing_dates": skipped_dates,
        "rows_added": int(sum(frame.height for frame in added_frames)),
        "rows_total": int(combined.height),
        "targets_left_null": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = factor_path.parent / (
        f"factor_append_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    report = append_factors(args.data_dir.resolve(), args.model_id, args.start, args.end)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
