"""Cancellation must address the same minute/regime task as the stream request."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from app.api import backtest as api


@pytest.mark.parametrize("minute_value", [None, "false", "true", "True", "1", "on", "yes", "t", "y"])
@pytest.mark.parametrize("regime_filter", [None, '{"mode":"strict"}'])
def test_cancel_preserves_minute_and_regime_job_identity(monkeypatch, minute_value, regime_filter):
    minute_fill = minute_value not in (None, "false")
    args = (
        "cancel_identity_fixture", "000001", "2026-09-21", "2026-10-08",
        "open_t+1", None, None, 0.0002, 5.0, 10, 1.0, 1_000_000.0,
        "equal", None, None,
    )
    key = api._make_job_key(*args, minute_fill=minute_fill, regime_filter=regime_filter)
    job = api._BacktestJob(key)
    other_key = api._make_job_key(*args, minute_fill=not minute_fill, regime_filter=regime_filter)
    other = api._BacktestJob(other_key)
    monkeypatch.setattr(api, "_running_jobs", {key: job, other_key: other})

    query = {
        "strategy_id": "cancel_identity_fixture", "symbols": "000001",
        "start": "2026-09-21", "end": "2026-10-08",
    }
    if minute_value is not None:
        query["minute_fill"] = minute_value
    if regime_filter is not None:
        query["regime_filter"] = regime_filter

    async def body():
        return {"qs": urlencode(query)}

    result = asyncio.run(api.strategy_cancel(SimpleNamespace(json=body)))

    assert result == {"ok": True}
    assert job.cancel_event.is_set()
    assert not other.cancel_event.is_set()


def test_cancel_rejects_invalid_minute_flag(monkeypatch):
    from fastapi import HTTPException

    job = api._BacktestJob("unrelated")
    monkeypatch.setattr(api, "_running_jobs", {"unrelated": job})

    async def body():
        return {"qs": "strategy_id=cancel_identity_fixture&minute_fill=invalid"}

    with pytest.raises(HTTPException) as error:
        asyncio.run(api.strategy_cancel(SimpleNamespace(json=body)))
    assert error.value.status_code == 400
    assert not job.cancel_event.is_set()
