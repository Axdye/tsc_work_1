"""Check metric edge cases using the actual upstream metric statements."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / ".build/cpu-deps"), str(ROOT / ".build/python")]
from cpu_upstream_baselines import load_upstream, upstream_metrics


class UpstreamMetricTests(unittest.TestCase):
    def test_terminal_censoring_shadow_filter_and_network_aggregation(self):
        import numpy as np
        _, _, metric_code = load_upstream(np)
        env = SimpleNamespace(list_intersection=[
            SimpleNamespace(dic_vehicle_arrive_leave_time={
                "a": {"enter_time": 10, "leave_time": 20},
                "b": {"enter_time": 30, "leave_time": 50},
                "shadow_ignored": {"enter_time": 0, "leave_time": 99}}),
            SimpleNamespace(dic_vehicle_arrive_leave_time={
                "a": {"enter_time": 20, "leave_time": np.nan}})])
        result = upstream_metrics(metric_code, np, env, 100, [12, 8], [0, 6], 0)
        # a: 10 + (100-20) = 90; b: 20. Mean is 55, not finished-only 20.
        self.assertEqual(result, {"ATT": 55.0, "AWT": 3.0, "AQL": 10.0})


if __name__ == "__main__":
    unittest.main()
