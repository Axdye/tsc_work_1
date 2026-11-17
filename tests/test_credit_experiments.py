"""Credit/budget/branch-restoration invariants, independent of GPU downloads."""

from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsc_experiments.common import configuration, standardize, returns_to_go
from tsc_experiments.policy import Response, ScriptedPolicy, parse_action
from tsc_experiments.credit import Collector, label_group


class FakeEnv:
    """Neighbor-sensitive known action value with snapshot-visible Python timers."""

    names = ["i0", "i1"]
    interval = 1
    horizon = 40

    def __init__(self):
        self.time, self.sim_seconds = 0, 0
        self.env = SimpleNamespace(
            list_intersection=[
                SimpleNamespace(current_phase_index=1) for _ in self.names
            ]
        )

    @property
    def remaining_steps(self):
        return self.horizon - self.time

    def prompt(self, i):
        return [{"role": "user", "content": f"{i}/{self.time}"}]

    def snapshot(self):
        return self.time, copy.deepcopy(self.env)

    def restore(self, state):
        self.time, self.env = copy.deepcopy(state)

    def rule_actions(self):
        return [0, 0]

    def step(self, actions):
        self.time += 1
        self.sim_seconds += 1
        for node, a in zip(self.env.list_intersection, actions):
            node.current_phase_index = a + 1
        return {
            "queue_costs": [10 + actions[0] + 8 * actions[1], 10 + actions[1]],
            "time": self.time,
        }


class CreditTests(unittest.TestCase):
    def config(self, **changes):
        return {
            **configuration(ROOT / "configs/credit_smoke.json"),
            "collect_steps": 2,
            "targets_per_state": 2,
            "horizon": 3,
            **changes,
        }

    def test_strict_signal_parser_and_group_zero_variance(self):
        self.assertEqual(parse_action("Reason. <signal>etwt</signal>"), 0)
        self.assertIsNone(parse_action("ETWT"))
        self.assertIsNone(parse_action("<signal>ETWT</signal><signal>NTST</signal>"))
        self.assertEqual(standardize([4, 4, 4]), [0, 0, 0])
        self.assertEqual(returns_to_go([1, 2, 3], 0.5), [2.75, 3.5, 3])

    def test_same_phase_texts_share_score_and_invalid_receives_penalty(self):
        rows = [
            Response([], [], [1], text, action)
            for text, action in [("a", 0), ("b", 0), ("bad", None), ("c", 1)]
        ]
        label_group(rows, [-10, -20, -4, -30], "group", 100)
        self.assertEqual([r.reward for r in rows], [-15, -15, -104, -30])
        self.assertAlmostEqual(sum(r.advantage for r in rows), 0)

    def test_all_collectors_budget_and_main_state(self):
        for method in ("paired", "unpaired", "joint", "trajectory", "rtg"):
            with self.subTest(method=method):
                env = FakeEnv()
                collector = Collector(
                    env, ScriptedPolicy(1), self.config(method=method), 1
                )
                expected = collector.window_cost()
                rows, diagnostics = collector.collect()
                self.assertEqual(env.sim_seconds, expected)
                self.assertEqual(
                    env.time, 2
                )  # Forks must not advance deployed trajectory.
                self.assertEqual(len(rows), 16)
                self.assertTrue(diagnostics)
                self.assertTrue(all(r.metadata["method"] == method for r in rows))

    def test_short_horizon_budget_never_passes_terminal(self):
        env = FakeEnv()
        env.time = env.horizon - 1
        collector = Collector(env, ScriptedPolicy(2), self.config(), 2)
        expected = collector.window_cost()
        collector.collect()
        self.assertEqual(env.time, env.horizon)
        self.assertEqual(env.sim_seconds, expected)

    def test_paired_removes_known_first_step_background_variance(self):
        from tsc_experiments.diagnostics import action_values

        env = FakeEnv()
        c = self.config(horizon=1)
        state = env.snapshot()
        # Under common background, changing a_i by one increases cost by exactly one.
        for seed in range(5):
            values = action_values(env, state, 0, ScriptedPolicy(seed), c, 1, True)
            self.assertEqual(values[0] - values[1], 1)
        self.assertEqual(env.time, 0)


class OptionalTorchTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "torch not installed; no GPU required for this unit test",
    )
    def test_advantage_direction_clipping_and_reference_kl(self):
        import torch
        from tsc_experiments.trainer import clipped_objective

        x = torch.tensor([-0.7, -0.9], requires_grad=True)
        old = x.detach().clone()
        loss, kl, _ = clipped_objective(x, old, old, 1.0, 0.2, 0.02)
        loss.backward()
        self.assertTrue(
            torch.all(x.grad < 0)
        )  # Positive A raises log probability under gradient descent.
        self.assertEqual(float(kl), 0)
        x = torch.tensor([0.0], requires_grad=True)
        loss, _, _ = clipped_objective(x, torch.tensor([-1.0]), x.detach(), 1.0, 0.2, 0)
        loss.backward()
        self.assertEqual(
            float(x.grad), 0
        )  # Positive advantage ratio above clip has no surrogate gradient.


if __name__ == "__main__":
    unittest.main()
