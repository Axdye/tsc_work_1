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
        from tsc_experiments.common import PHASES
        from tsc_experiments.environment import TrafficEnvironment, cluster_features, prompt

        with tempfile.TemporaryDirectory() as path:
            env = TrafficEnvironment("jinan2", 0, Path(path) / "engine", horizon=300)
            for _ in range(6):
                env.step(env.rule_actions())
            snapshot = env.snapshot()
            before = env.measurement()
            inputs = env.observations()
            self.assertTrue(any(
                sum(row["incoming_running"] for row in o["phase_observations"].values())
                for o in inputs
            ))
            for index, o in enumerate(inputs):
                self.assertEqual(len(cluster_features(o)), 16)
                self.assertEqual(tuple(o["phase_observations"]), PHASES)
                for phase, data in enumerate(o["phase_observations"].values()):
                    self.assertEqual(len(data["approaches"]), 2)
                    for a, (_, receiving) in zip(data["approaches"], env.phase_lanes[index][phase]):
                        self.assertEqual(len(a["approaching"]), 3)
                        self.assertLessEqual(a["wait_seconds"][0], a["wait_seconds"][1])
                        self.assertEqual(a["downstream"] is None, not receiving)
                        if receiving:
                            self.assertLessEqual(0, a["downstream"][1])
                            self.assertLessEqual(a["downstream"][1], a["downstream"][2])
                            self.assertLessEqual(a["downstream"][2], 1)
                messages = prompt(o)
                self.assertIn("近/中/远来车", messages[1]["content"])
                self.assertIn("ETWT｜东", messages[1]["content"])
                self.assertIn("<signal>ETWT</signal>", messages[0]["content"])
            arrivals = env.arrivals()
            actions = [1] * len(env.names)
            first = env.step(actions)
            end_arrivals = env.arrivals()
            env.restore(snapshot)
            self.assertEqual(env.measurement(), before)
            self.assertEqual(env.observations(), inputs)
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
