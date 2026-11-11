from __future__ import annotations

from datetime import date

import polars as pl
import probe_bj_gap_sources as probe
import pytest


def test_csv_safe_rows_serializes_nested_diagnostics() -> None:
    try:
        from app.reporting import csv_safe_rows
    except ImportError as exc:
        raise AssertionError("CSV diagnostics serializer is not implemented") from exc

    rows = [{
        "source": "promax",
        "issues": [{"reason": "source_empty", "retryable": False}],
        "identity": {"market": "BJ"},
    }]

    safe = csv_safe_rows(rows)

    assert safe == [{
        "source": "promax",
        "issues": '[{"reason": "source_empty", "retryable": false}]',
        "identity": '{"market": "BJ"}',
    }]


@pytest.mark.parametrize("source", ["tdx", "promax"])
@pytest.mark.parametrize("reason", ["upstream_unavailable", "source_incomplete", "source_pending", "source_empty"])
def test_empty_normalized_frame_preserves_provider_diagnostic(monkeypatch, tmp_path, source, reason):
    issue = {"reason": reason, "retryable": reason in {"upstream_unavailable", "source_pending"}}

    class Provider:
        def get_minute_diagnostics(self, *args, **kwargs):
            return pl.DataFrame(), [issue]

        def close(self):
            pass

    monkeypatch.setattr(probe, "TdxProvider" if source == "tdx" else "TushareProvider", Provider)
    item = ("920701.BJ", "920701", date(2026, 9, 30))
    row = probe.probe_tdx_one(item, {}) if source == "tdx" else probe.probe_promax_one(item, {}, tmp_path)
    assert row["validation"] == reason
    assert row["issues"] == [issue]
    assert probe.summarize([row])[source]["classifications"] == {reason: 1}


def test_empty_frame_without_diagnostics_stays_source_empty():
    result = probe.validate(pl.DataFrame(), "920701", date(2026, 9, 30), {}, "qmt")
    assert result["validation"] == "source_empty"


def test_multiple_source_failures_are_not_reduced_to_source_empty():
    issues = [{"reason": "upstream_unavailable"}, {"reason": "source_incomplete"}]
    result = probe.validate(pl.DataFrame(), "920701", date(2026, 9, 30), {}, "promax", issues=issues)
    assert result["validation"] == "source_multiple_issues"
