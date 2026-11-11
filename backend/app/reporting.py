"""Small helpers for durable machine-readable operator reports."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any


def csv_safe_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Serialize nested diagnostics so tabular report writers cannot drop them."""
    return [
        {
            key: json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, (dict, list, tuple, set))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
