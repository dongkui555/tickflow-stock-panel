import io
import json
from contextlib import redirect_stdout
from unittest import TestCase, main
from unittest.mock import Mock

from qmt_http_api import QmtApi


class Frame:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def iterrows(self):
        return iter(self.rows)


class QmtCapabilitiesTest(TestCase):
    def api(self, minute_rows, *, tick_error=True):
        def get_market_data(**params):
            if params["period"] == "1m":
                self.assertEqual(params["count"], -1)
                self.assertIn("start_time", params)
                self.assertIn("end_time", params)
                return Frame(minute_rows)
            if params["period"] == "tick" and tick_error:
                raise RuntimeError("live quote routing unavailable")
            return Frame([])

        def execute(code):
            output = io.StringIO()
            with redirect_stdout(output):
                exec(code, {"get_market_data": get_market_data})
            return json.loads(output.getvalue().split("QMT_BRIDGE_RESULT=")[-1])

        kernel = Mock()
        kernel.execute_json.side_effect = execute
        push = Mock()
        push.capabilities.return_value = {}
        return QmtApi(kernel, push)

    def test_history_probe_runs_when_live_tick_probe_raises(self):
        api = self.api([("20261009150000", {"close": 11.78})])
        caps = api.capabilities(refresh=True)
        self.assertTrue(caps["minute"])
        self.assertFalse(caps["realtime"])
        self.assertFalse(caps["depth5"])
        self.assertFalse(caps["symbol_routing"])

    def test_failed_live_routing_does_not_erase_valid_history(self):
        caps = self.api([("20261009150000", {"close": 11.78})], tick_error=False).capabilities(refresh=True)
        self.assertTrue(caps["minute"])
        self.assertFalse(caps["symbol_routing"])

    def test_empty_history_is_not_advertised(self):
        self.assertFalse(self.api([]).capabilities(refresh=True)["minute"])

    def test_invalid_minute_rows_are_not_advertised(self):
        for stamp, close in (("20261009120000", 11.78), ("invalid", 11.78),
                             ("20261009150000", 0), ("20261009150000", float("nan"))):
            with self.subTest(stamp=stamp, close=close):
                self.assertFalse(self.api([(stamp, {"close": close})]).capabilities(refresh=True)["minute"])

    def test_health_reuses_last_explicit_probe_without_reprobing(self):
        api = self.api([("20261009150000", {"close": 11.78})])
        api.capabilities(refresh=True)
        self.assertTrue(api.health()["capabilities"]["minute"])
        self.assertEqual(api.kernel.execute_json.call_count, 1)


if __name__ == "__main__":
    main()
