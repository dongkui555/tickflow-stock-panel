"""Explicit evidence for regular-session turnover; raw daily totals stay intact."""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)
FIELDS = ('open', 'high', 'low', 'close', 'volume', 'amount')


def positive(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a financial value')
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError('Expected positive finite financial value')
    return number


def regular_reference(reference: dict, evidence: dict) -> dict:
    """Use independently evidenced turnover, never a residual inferred from bars."""
    symbol = str(reference['symbol'])
    day = str(reference['date'])
    if not re.fullmatch(r'92\d{4}\.BJ', symbol):
        raise ValueError('Evidence contract only covers canonical Beijing stocks')
    if (evidence['schema_version'] != 1 or evidence['basis'] != 'bse_post_close_block_trade'
            or evidence['symbol'] != symbol or evidence['date'] != day):
        raise ValueError('Evidence scope mismatch')
    sources = evidence['sources']
    if not sources or any(not s['provider'] or not re.fullmatch(r'[a-f0-9]{64}', s['sha256']) for s in sources):
        raise ValueError('Evidence provenance missing')
    for field in FIELDS:
        if positive(reference[field]) != positive(evidence['daily_snapshot'][field]):
            raise ValueError('Daily reference changed since evidence was reviewed')
    unit = evidence['daily_volume_unit']
    if unit not in {'lots', 'shares'}:
        raise ValueError('Daily volume unit must be explicit')
    total = evidence['daily_total']
    shares, amount = positive(total['volume_shares']), positive(total['amount_cny'])
    for field in FIELDS[:4]:
        if abs(positive(total[field]) - reference[field]) > max(.011, reference[field] * 1e-6):
            raise ValueError('Independent daily prices disagree')
    factor = 100 if unit == 'lots' else 1
    if abs(shares - reference['volume'] * factor) > max(factor, shares * .002):
        raise ValueError('Independent daily volume disagrees')
    if abs(amount - reference['amount']) > max(1., amount * .002):
        raise ValueError('Independent daily amount disagrees')
    blocks = evidence['blocks']
    if not blocks:
        raise ValueError('No published post-close block trades')
    block_shares = block_amount = 0.
    for trade in blocks:
        if trade['symbol'] != symbol or trade['date'] != day:
            raise ValueError('Block trade scope mismatch')
        qty, price = positive(trade['volume_shares']), positive(trade['price'])
        cash = positive(trade['amount_cny'])
        if abs(cash - qty * price) > .011:
            raise ValueError('Block trade price/volume/amount mismatch')
        block_shares += qty
        block_amount += cash
    regular_shares = positive(shares - block_shares)
    regular_amount = positive(amount - block_amount)
    avg = regular_amount / regular_shares
    tolerance = max(.02, reference['high'] * .002)
    if not reference['low'] - tolerance <= avg <= reference['high'] + tolerance:
        raise ValueError('Derived regular-session turnover inconsistent with daily range')
    return {**reference, 'volume': regular_shares, 'amount': regular_amount,
            '_volume_scope': 'regular_session'}


def apply_reference_evidence(references: dict[str, dict], data_dir: Path) -> dict[str, dict]:
    """An invalid/stale evidence file excludes its target, without affecting peers."""
    root = Path(data_dir) / 'minute_reference_evidence'
    if not root.is_dir():
        return references
    result = dict(references)
    days = {str(row['date']) for row in references.values() if row.get('date') is not None}
    for day in days:
        # Accept only date-only partitions, never paths supplied in JSON evidence.
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError('Expected date-only daily reference')
        for path in (root / f'date={day}').glob('*.json'):
            code = path.stem
            if code not in result or str(result[code].get('date')) != day:
                continue
            try:
                if path.stat().st_size > 65536:
                    raise ValueError('Evidence exceeds size limit')
                evidence = json.loads(path.read_text(encoding='utf-8'))
                result[code] = regular_reference(result[code], evidence)
            except (OSError, ValueError, KeyError, TypeError, OverflowError):
                result.pop(code, None)
                logger.warning('Invalid regular-session reference evidence: %s %s', code, day)
    return result
