"""Scoped order-quantity adapter for the single-threaded research worker.

The frozen paper source and its matcher remain unchanged. Only a whole stock
position's odd-lot exit is admitted. Original preflight still checks T+1 and
pending reservations; original matching retains price limits, fees and timing.
Do not install this process-local adapter in the shared HTTP application.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.strategy import paper

POLICY = 'whole_position_odd_lot_exit_v1'


@contextmanager
def environment(data_dir: Path, account_id: str, symbols):
    root = Path(data_dir).resolve()
    allowed = frozenset(symbols)
    with paper.PAPER_LOCK:
        original = paper.create_order
        lot_size = paper.LOT_SIZE
        if getattr(original, '_shadow_quantity_policy', None):
            raise RuntimeError('Quantity policy already installed')
        if lot_size != 100:
            raise ValueError('Unsupported frozen paper quantity contract')

        def create_order(data_dir, symbol, side, **kwargs):
            with paper.PAPER_LOCK:
                if lot_size != paper.LOT_SIZE:
                    raise RuntimeError('Reentrant quantity-policy order call')
                if (Path(data_dir).resolve() != root
                        or kwargs.get('account_id', paper.DEFAULT_ACCOUNT_ID) != account_id):
                    return original(data_dir, symbol, side, **kwargs)
                value = kwargs.get('qty')
                if value is None:
                    return original(data_dir, symbol, side, **kwargs)
                try:
                    qty = Decimal(str(value))
                    valid = not isinstance(value, bool) and qty.is_finite() and qty > 0 and qty == qty.to_integral_value()
                except InvalidOperation:
                    valid = False
                if not valid:
                    return None, '数量必须是正整数股数'
                qty = int(qty)
                if qty % lot_size == 0:
                    return original(data_dir, symbol, side, **kwargs)
                position = paper.load_positions(data_dir, account_id).get(symbol, {})
                if (side != 'sell' or symbol not in allowed
                        or kwargs.get('asset_type') not in {None, 'stock'}
                        or position.get('asset_type') != 'stock' or position.get('qty') != qty):
                    return None, '非整手数量仅允许本研究股票持仓一次性全仓卖出'
                # The original function's sole quantity gate is LOT_SIZE. Its
                # reentrant account lock protects the entire delegated call.
                # Retain its original, atomic order save with the real quantity;
                # never create a 100-share order and rewrite it to 130 afterward.
                try:
                    paper.LOT_SIZE = 1
                    return original(data_dir, symbol, side, **kwargs)
                finally:
                    paper.LOT_SIZE = lot_size

        create_order._shadow_quantity_policy = POLICY
        paper.create_order = create_order
    try:
        yield
    finally:
        with paper.PAPER_LOCK:
            paper.create_order = original
            paper.LOT_SIZE = lot_size
