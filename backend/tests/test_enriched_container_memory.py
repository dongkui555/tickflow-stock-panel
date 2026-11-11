from types import SimpleNamespace

import psutil
import pytest
from app.indicators import pipeline
from app.services import memory_budget


@pytest.mark.parametrize("v2,v1,expected_gib", [
    (str(4 * 1024**3), None, 4),
    ("max", str(6 * 1024**3), 6),
    ("max", None, 16),
    ("broken", str(2 * 1024**3), 2),
    (str(64 * 1024**3), None, 16),
])
def test_batch_memory_probe_respects_container_limit(tmp_path, monkeypatch, v2, v1, expected_gib):
    host_bytes = 16 * 1024**3
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=host_bytes, available=host_bytes))
    groups = []
    for name, limit, unified in (("memory.max", v2, True), ("memory.limit_in_bytes", v1, False)):
        group = tmp_path / name.removesuffix("_in_bytes")
        group.mkdir()
        if limit is not None:
            (group / name).write_text(limit)
        groups.append((group, group, unified))
    monkeypatch.setattr(memory_budget, "_memory_groups", lambda: groups)
    assert pipeline._total_ram_bytes() == expected_gib * 1024**3


def test_eight_gib_container_uses_bounded_history_batches(monkeypatch):
    monkeypatch.setattr(pipeline, "_total_ram_bytes", lambda: 8 * 1024**3)
    assert pipeline._adaptive_sym_batch(2000, 200) == 750
