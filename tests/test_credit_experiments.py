"""Credit/budget/branch-restoration invariants, independent of GPU downloads."""

from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsc_experiments.common import (
    configuration,
    standardize,
    returns_to_go,
    validate_config,
)
from tsc_experiments.policy import Response, ScriptedPolicy, parse_action
from tsc_experiments.credit import Collector, label_group, group_diagnostic


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

    def test_background_generation_has_no_discarded_batch(self):
        from tsc_experiments.diagnostics import action_values

        # Two nodes, G=4, M=2, H=1: deployment + candidates + used backgrounds.
        for method, expected in (
            ("paired", 2 + 4 + 2 * 2),
            ("unpaired", 2 + 4 + 2 * 4 * 2),
        ):
            env, policy = FakeEnv(), ScriptedPolicy(0)
            c = self.config(
                method=method,
                collect_steps=1,
                targets_per_state=1,
                horizon=1,
                contexts=2,
            )
            Collector(env, policy, c, 0).collect()
            self.assertEqual(policy.calls, expected)
        for paired, expected in ((True, 2 * 2), (False, 2 * 4 * 2)):
            env, policy = FakeEnv(), ScriptedPolicy(0)
            action_values(
                env, env.snapshot(), 0, policy, self.config(horizon=1), 2, paired
            )
            self.assertEqual(policy.calls, expected)

    def test_format_signal_is_distinguished_from_traffic_contrast(self):
        rows = [Response([], [], [1], "", action) for action in (0, 0, None, 0)]
        label_group(rows, [-10, -10, -10, -10], "g", 100)
        stats = group_diagnostic(rows, "g", 0)
        self.assertFalse(stats["zero_advantage"])
        self.assertFalse(stats["traffic_signal"])
        self.assertEqual(stats["unique_phases"], 1)
        rows[-1].action = 1
        label_group(rows, [-10, -10, -10, -20], "g", 100)
        self.assertTrue(group_diagnostic(rows, "g", 0)["traffic_signal"])

    def test_unsupported_or_nonfinite_configuration_fails_before_collection(self):
        for change in (
            {"method": "joint", "reward_scope": "local"},
            {"method": "joint", "contexts": 2},
            {"method": "trajectory", "continuation": "maxpressure"},
            {"method": "rtg", "reward_scope": "local"},
            {"invalid_penalty": -1},
            {"invalid_penalty": float("nan")},
            {"learning_rate": float("inf")},
            {"warmup_steps": 1.5},
            {"group_size": True},
            {"local_files_only": "true"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(self.config(**change))
        offline = configuration(
            ROOT / "configs/credit_smoke.json",
            ["local_files_only=true", "model=/data/models/qwen"],
        )
        self.assertTrue(offline["local_files_only"])
        self.assertEqual(offline["model"], "/data/models/qwen")


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
