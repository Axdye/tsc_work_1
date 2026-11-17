"""Opt-in short real-engine branch/metric integration test. No LLM training."""

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(
    os.environ.get("TSC_TEST_CITYFLOW") == "1",
    "Set TSC_TEST_CITYFLOW=1 for real-engine test",
)
class EngineTests(unittest.TestCase):
    def test_engine_and_python_timers_restore_together(self):
        from tsc_experiments.environment import TrafficEnvironment

        with tempfile.TemporaryDirectory() as path:
            env = TrafficEnvironment("jinan2", 0, Path(path) / "engine", horizon=300)
            for _ in range(6):
                env.step(env.rule_actions())
            snapshot = env.snapshot()
            before = env.measurement()
            arrivals = env.arrivals()
            actions = [1] * len(env.names)
            first = env.step(actions)
            end_arrivals = env.arrivals()
            env.restore(snapshot)
            self.assertEqual(env.measurement(), before)
            self.assertEqual(env.arrivals(), arrivals)
            repeated = env.step(actions)
            self.assertEqual(first, repeated)
            self.assertEqual(env.arrivals(), end_arrivals)
            self.assertTrue(
                all(i.eng is env.env.eng for i in env.env.list_intersection)
            )
            env.restore(snapshot)
            with contextlib.redirect_stdout(io.StringIO()):
                env.env.step(actions)
            expected = env.measurement()
            expected_arrivals = env.arrivals()
            env.restore(snapshot)
            env.step(actions)
            self.assertEqual(env.measurement(), expected)
            self.assertEqual(env.arrivals(), expected_arrivals)

    def test_metric_audit_and_upstream_equivalence(self):
        from tsc_experiments.common import configuration
        from tsc_experiments.runner import evaluate
        from tsc_experiments.visualization import audit_evaluation

        with tempfile.TemporaryDirectory() as path:
            c = configuration(
                ROOT / "configs/credit_smoke.json", ["episode_seconds=300"]
            )
            output = Path(path) / "evaluation"
            evaluate(c, output, "jinan2", 0, "maxpressure")
            self.assertTrue(audit_evaluation(output)["passed"])


if __name__ == "__main__":
    unittest.main()
