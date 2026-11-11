import struct
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from import_qmt_daily_dat import _read_qmt
from repair_targeted_qmt_dat import QmtDatProvider

DAY = date(2026, 9, 22)
TZ = timezone(timedelta(hours=8))


def _record(stamp, lots, remainder):
    # Pack the independently audited 64-byte layout, including offset 56.
    return struct.pack(
        '<8IQI2f3I', int(stamp.replace(tzinfo=TZ).timestamp()),
        10000, 10000, 10000, 10000, 0, lots, 399,
        (lots * 100 + remainder) * 10, 0, 1.0, 1.0, 0, remainder, 32761,
    )


def _file(root: Path, remainders=(25, 99)):
    path = root / 'BJ/60/920701.DAT'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'\0' * 8 + b''.join(
        _record(datetime(2026, 9, 22, 9, 31 + index), 7 + index, remainder)
        for index, remainder in enumerate(remainders)
    ))
    return path


def test_qmt_minute_preserves_odd_shares_in_lots(tmp_path):
    _file(tmp_path)
    frame = QmtDatProvider(tmp_path).get_minute(
        ['920701.BJ'], datetime(2026, 9, 22), datetime(2026, 9, 22, 15),
    )
    assert frame['volume'].to_list() == pytest.approx([7.25, 8.99])
    assert frame['volume'].sum() * 100 == pytest.approx(1624)
    assert frame['amount'].sum() / (frame['volume'].sum() * 100) == pytest.approx(10)


def test_qmt_daily_import_sums_precise_shares(tmp_path):
    row = _read_qmt(_file(tmp_path), DAY)
    assert row['volume'] == pytest.approx(1624)
    assert row['amount'] == 16240


@pytest.mark.parametrize('reader', ['minute', 'daily'])
def test_unknown_qmt_remainder_layout_rejected(tmp_path, reader):
    path = _file(tmp_path, remainders=(25, 100))
    with pytest.raises(ValueError, match=r'QMT.*remainder'):
        if reader == 'minute':
            QmtDatProvider(tmp_path).get_minute(
                ['920701.BJ'], datetime(2026, 9, 22), datetime(2026, 9, 22, 15),
            )
        else:
            _read_qmt(path, DAY)


def test_whole_lot_records_remain_compatible(tmp_path):
    row = _read_qmt(_file(tmp_path, remainders=(0, 0)), DAY)
    assert row['volume'] == 1500
