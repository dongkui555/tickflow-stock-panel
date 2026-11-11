"""回归测试: SDK 批量调用守护 deadline(「分钟K同步卡死空转」持有侧修复)。

TickFlow SDK 自带 timeout=30s×3 重试, 但只保护其覆盖的请求阶段; 真挂死在
不可中断的网络读里时调用永不返回, 持有独占槽的重任务线程被连坐卡死, 后续
重任务全部无限排队。deadline 包装在超时后放弃等待, 调用方按分块失败路径
继续, 心跳与协作取消在分块边界照常生效。纯逻辑, 不触网。
"""
from __future__ import annotations

import functools
import time
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.data_providers.base import ProviderRequestError
from app.services import kline_sync
from app.services.kline_sync import SDKCallTimeoutError, _sdk_call_with_deadline

# ── 包装器单元行为 ──────────────────────────────────────────────────────

def test_result_passthrough():
    assert _sdk_call_with_deadline(lambda: 42) == 42


def test_exception_propagates_unchanged():
    def _boom():
        raise ValueError("sdk error")

    with pytest.raises(ValueError, match="sdk error"):
        _sdk_call_with_deadline(_boom)


def test_deadline_exceeded_abandons_wait():
    started = time.monotonic()
    with pytest.raises(SDKCallTimeoutError):
        _sdk_call_with_deadline(lambda: time.sleep(3), timeout_s=0.1)
    assert time.monotonic() - started < 2  # 不陪着挂死的调用一起等


# ── 分块循环集成: 超时走失败路径且心跳不断 ─────────────────────────────

class _HangingKlines:
    @staticmethod
    def batch(*args, **kwargs):
        time.sleep(3)  # 模拟挂死; worker 是 daemon 线程, 不拖慢进程退出


class _HangingClient:
    klines = _HangingKlines


def _fast_deadline(monkeypatch):
    """循环内 deadline 默认 600s, 换成 0.1s 版本方便测试。"""
    real = kline_sync._sdk_call_with_deadline
    monkeypatch.setattr(
        kline_sync, "_sdk_call_with_deadline",
        functools.partial(real, timeout_s=0.1),
    )


def test_minute_loop_timeout_keeps_heartbeat(monkeypatch):
    """分钟分块超时 → 本批放弃并继续, on_chunk_done 心跳仍被调用
    (保留进度上报与协作取消的检查点, 防僵尸线程滞留独占槽)。"""
    _fast_deadline(monkeypatch)
    monkeypatch.setattr(kline_sync, "_try_custom_minute", lambda *a, **k: (None, True))
    monkeypatch.setattr(kline_sync, "get_client", lambda: _HangingClient())

    beats: list[tuple[int, int, str]] = []
    with pytest.raises(ProviderRequestError, match="未完成"):
        kline_sync.sync_minute_batch(
            ["600000.SH", "000001.SZ"],
            start_time=datetime(2026, 8, 3), end_time=datetime(2026, 8, 7),
            batch_size=1, rpm=None,
            on_chunk_done=lambda cur, total, seg: beats.append((cur, total, seg)),
        )
    assert [b[0] for b in beats] == [1, 2]  # 每个超时分块都推进心跳
    assert beats[0][1] == 2


def test_daily_loop_timeout_marks_failed(monkeypatch):
    """日K分块超时 → failed_out 收编本批(上层可判「部分失败」), 心跳推进。"""
    _fast_deadline(monkeypatch)
    monkeypatch.setattr(kline_sync, "get_client", lambda: _HangingClient())

    beats: list[tuple[int, int]] = []
    failed: list[str] = []
    df = kline_sync.sync_daily_batch(
        ["600000.SH", "000001.SZ"], count=250, batch_size=1, rpm=None,
        on_chunk_done=lambda cur, total: beats.append((cur, total)),
        failed_out=failed,
    )

    assert df.is_empty()
    assert sorted(failed) == ["000001.SZ", "600000.SH"]
    assert [b[0] for b in beats] == [1, 2]


@pytest.mark.parametrize('kind', ['daily_count', 'daily_range', 'minute_count', 'minute_range', 'adj_factor'])
def test_late_sdk_calls_keep_original_batch_and_segment(monkeypatch, kind):
    pending, calls = [], []

    def abandon(fn):
        pending.append(fn)
        raise SDKCallTimeoutError('delayed worker')

    def record(symbols, **kwargs):
        calls.append((list(symbols), kwargs))

    monkeypatch.setattr(kline_sync, '_sdk_call_with_deadline', abandon)
    monkeypatch.setattr(kline_sync, 'get_client', lambda: SimpleNamespace(
        klines=SimpleNamespace(batch=record, ex_factors=record)))
    monkeypatch.setattr(kline_sync, 'sleep_between_batches', lambda *a: None)
    monkeypatch.setattr(kline_sync, '_try_custom_minute', lambda *a, **k: (None, True))
    symbols = ['600000.SH', '000001.SZ']
    window = {'start_time': datetime(2026, 8, 3), 'end_time': datetime(2026, 8, 5)}
    if kind.startswith('daily'):
        kline_sync.sync_daily_batch(symbols, batch_size=1, **(window if kind.endswith('range') else {}))
    elif kind.startswith('minute'):
        from app.data_providers.base import ProviderRequestError

        with pytest.raises(ProviderRequestError, match="分钟同步未完成"):
            kline_sync.sync_minute_batch(symbols, batch_size=1, segment_trading_days=1,
                                        **(window if kind.endswith('range') else {}))
    else:
        monkeypatch.setattr(kline_sync.preferences, 'get_adj_factor_provider', lambda: 'tickflow')
        monkeypatch.setattr(kline_sync, 'resolve_limit', lambda *a, **k: SimpleNamespace(batch=1, rpm=None))
        kline_sync.sync_adj_factor(symbols, None, SimpleNamespace(has=lambda _: True))
    # Execute abandoned workers only after both loops advanced to their end.
    for fn in pending:
        fn()
    assert [call[0] for call in calls] == [[symbols[0]], [symbols[1]]] * (len(pending) // 2)
    if kind == 'minute_range':
        expected = [datetime(2026, 8, 3), datetime(2026, 8, 4)]
        assert [calls[index][1]['start_time'] for index in (0, 2)] == [
            kline_sync._datetime_to_ms(value) for value in expected]
