"""Focused regression checks for the Work 2 collector and advantage estimator."""

from collections import Counter
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsc_experiments.branches import score_slice
from tsc_experiments.common import PHASES, configuration, validate_config
from tsc_experiments.credit import Collector, loo_probabilities, set_counterfactual_advantages
from tsc_experiments.policy import Response, ScriptedPolicy, parse_action
from tsc_experiments.suite import make_plan


class FakeEnv:
    names = ["i0", "i1"]
    interval, horizon = 1, 4

    def __init__(self):
        self.time = self.sim_seconds = 0
        self.phases = [0, 0]

    @property
    def remaining_steps(self):
        return self.horizon - self.time

    def observations(self):
        return [{
            "intersection": name,
            "current_phase": PHASES[phase],
            "time_seconds": self.time,
            "phase_age_seconds": 0,
            "decision_interval_seconds": self.interval,
            "clearance_seconds": 0,
            "neighbors_ENWS": [None] * 4,
            "phase_observations": {
                label: {
                    "incoming_queued": 2,
                    "incoming_running": 0,
                    "receiving_occupancy": 0,
                    "approaches": [
                        {"queued": 1, "approaching": (0, 0, 0), "wait_seconds": (0, 0), "downstream": None}
                    ] * 2,
                }
                for label in PHASES
            }
        } for name, phase in zip(self.names, self.phases)]

    def snapshot(self):
        return self.time, self.phases[:]

    def restore(self, snapshot):
        self.time, self.phases = snapshot[0], snapshot[1][:]

    def rule_actions(self):
        return [0, 0]

    def step(self, actions):
        if not self.remaining_steps:
            raise ValueError("Crossed cutoff")
        self.time += 1
        self.sim_seconds += 1
        self.phases = list(actions)
        return {"queue_costs": [10 + actions[0] + 8 * actions[1], 10 + actions[1]]}


class OneCluster:
    def predict(self, features):
        return [0] * len(features)


class CreditTests(unittest.TestCase):
    def config(self, **changes):
        return {**configuration(ROOT / "configs/credit_smoke.json"), **changes}

    def test_fixed_horizon_cache_and_interference(self):
        env = FakeEnv()
        snapshot = env.snapshot()
        scores, first_reward = score_slice(env, [1, 2], 2, 0.95)
        self.assertEqual(first_reward, -39)
        self.assertEqual(scores[0], [-57, -58, -59, -60])
        self.assertEqual(scores[1], [-40, -49, -58, -67])
        self.assertEqual(env.sim_seconds, (1 + 3 * 2) * 2)
        self.assertEqual(env.snapshot(), snapshot)

    def test_full_window_budget_includes_last_full_horizon(self):
        for method in ("counterfactual", "joint", "trajectory", "rtg"):
            with self.subTest(method=method):
                env, policy = FakeEnv(), ScriptedPolicy(1)
                env.time = 1  # Last valid start given T=2, H=2.
                collector = Collector(env, policy, self.config(method=method), OneCluster())
                rows, info = collector.collect()
                self.assertEqual(env.time, 3)
                self.assertFalse(collector.ready)
                self.assertEqual(policy.calls, collector.response_cost)
                self.assertEqual(len(rows), 16)
                self.assertEqual(env.sim_seconds, info["main_trajectory_sim_seconds"] + info["branch_sim_seconds"])
                if method == "counterfactual":
                    self.assertEqual(info["branch_sim_seconds"], 4 * 2 * 7 * 2)
                    self.assertTrue(all(len(r.metadata["candidate_returns"]) == 4 for r in rows))
                with self.assertRaises(ValueError):
                    collector.collect()

    def test_loo_removes_only_self_and_excludes_invalid(self):
        self.assertEqual(loo_probabilities(Counter({0: 2, 1: 1}), 0, 1),
                         [2/6, 2/6, 1/6, 1/6])
        rows = [Response([], [], [], "", action, advantage=-1.0 if action is None else 2.0)
                for action in (0, 0, 1, None)]
        scores = [[4, 0, 0, 0]] * 4
        info = set_counterfactual_advantages(
            rows, [0, 0, 1, 1], scores, self.config(cluster_min=2)
        )
        self.assertEqual(info["cluster_sizes"], {0: 2, 1: 1})
        self.assertEqual(rows[0].metadata["loo_probabilities"], [2/5, 1/5, 1/5, 1/5])
        self.assertEqual(rows[2].metadata["loo_probabilities"], [3/6, 1/6, 1/6, 1/6])
        self.assertEqual(info["cluster_fallback_rate"], 1/3)
        self.assertAlmostEqual(info["cf_rms"], math.sqrt((2.4**2 * 2 + 4) / 3))
        self.assertEqual(rows[3].advantage, -1.0)
        self.assertNotIn("cluster", rows[3].metadata)

    def test_flat_rewards_and_symmetric_clipping(self):
        cfg = self.config(cf_baseline="uniform", cf_weight=1.0, cf_clip=0.3)
        rows = [Response([], [], [], "", action) for action in (0, 1, 2)]
        info = set_counterfactual_advantages(rows, [0]*3, [[100, 0, 50, 50]]*3, cfg)
        self.assertEqual([r.advantage for r in rows], [0.3, -0.3, 0])
        self.assertEqual(info["cf_clip_fraction"], 2/3)
        info = set_counterfactual_advantages(rows, [0]*3, [[-14]*4]*3, self.config())
        self.assertAlmostEqual(info["cf_rms"], 0.0)

    def test_budget_and_suite(self):
        c = configuration(ROOT / "configs/credit_main.json")
        self.assertEqual(c["rounds"] * c["group_size"] * c["collect_steps"] * 16, c["max_responses"])
        for changes in ({"horizon": 9}, {"method": "paired"}, {"train_seconds": 2701},
                        {"cf_weight": 1.1}, {"rms_floor": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**c, **changes})
        with tempfile.TemporaryDirectory() as path:
            plan = make_plan(ROOT / "configs/credit_suite.json", path)
            trains = {j["id"]: j for j in plan["jobs"] if j["kind"] == "train"}
            self.assertEqual(len(trains), 9)
            self.assertEqual(len(plan["jobs"]), 81)
            self.assertFalse(trains["train-colight-s0"]["gpu"])
            self.assertEqual(trains["train-trajectory-wall-s0"]["depends"], ["train-full-s0"])

    def test_phase_parser(self):
        self.assertEqual(parse_action("Reason. <signal>etwt</signal>"), 0)
        self.assertIsNone(parse_action("ETWT"))
        self.assertIsNone(parse_action("<signal>ETWT</signal><signal>NTST</signal>"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch not installed")
class OptionalTorchTests(unittest.TestCase):
    def test_advantage_direction_and_shared_colight_actions(self):
        import torch
        from tsc_experiments.colight import CoLight, LOCAL_FEATURES, NEIGHBOR_FEATURES, state
        from tsc_experiments.trainer import clipped_objective

        x = torch.tensor([-0.7, -0.9], requires_grad=True)
        old = x.detach().clone()
        loss, kl, _ = clipped_objective(x, old, old, 1.0, 0.2, 0.02)
        loss.backward()
        self.assertTrue(torch.all(x.grad < 0))
        self.assertEqual(float(kl), 0)
        model = CoLight()
        local, neighbors = state(FakeEnv())
        self.assertEqual(local.shape, (2, LOCAL_FEATURES))
        self.assertEqual(neighbors.shape, (2, 4, NEIGHBOR_FEATURES))
        values = model(local, neighbors)
        self.assertEqual(values.shape, (2, 4))
        self.assertTrue(torch.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
