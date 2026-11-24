"""Prevent inflated sample counts and incompatible-protocol result tables."""

from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsc_experiments.common import append_json, read_json, write_json


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
class ReportingTests(unittest.TestCase):
    def fixture(self, root, train_seed, seed, att, horizon=3600):
        folder = Path(root) / f"s{train_seed}-e{seed}"
        folder.mkdir()
        write_json(
            folder / "metrics.json",
            {
                "ATT": att,
                "AWT": 2.0,
                "AQL": 4.0,
                "dataset": "toy",
                "policy": "llm",
                "method": "full",
                "train_seed": train_seed,
                "seed": seed,
            },
        )
        write_json(
            folder / "manifest.json",
            {
                "protocol": "test",
                "interval": 30,
                "clearance_within_interval": 5,
                "roadnet_sha256": "road",
                "flow_sha256": "flow",
                "horizon": horizon,
                "laneChange": True,
            },
        )
        append_json(
            folder / "decisions.jsonl", {"time": 30, "AQL_sample": 4, "AWT_sample": 2}
        )
        return folder

    def test_average_environment_seeds_before_training_seeds(self):
        from tsc_experiments.visualization import report

        with tempfile.TemporaryDirectory() as path:
            root, out = Path(path) / "runs", Path(path) / "report"
            root.mkdir()
            for train_seed, seed, att in [
                (0, 0, 10),
                (0, 1, 20),
                (1, 0, 30),
                (1, 1, 40),
            ]:
                self.fixture(root, train_seed, seed, att)
            # Synthetic plotting fixture, never an experiment result.
            training = root / "training-fixture"
            training.mkdir()
            append_json(
                training / "training.jsonl",
                {
                    "round": 1,
                    "sim_seconds": 100,
                    "wall_seconds": 1,
                    "training_generated_tokens": 20,
                    "training_llm_calls": 4,
                    "cf_uniform_mean": 0.5,
                    "cf_loo_mean": 0.1,
                    "cluster_sharpness": 0.5,
                    "invalid_response_fraction": 0,
                    "truncated_fraction": 0,
                },
            )
            report(root, out)
            row = read_json(out / "summary.json")[0]
            self.assertEqual(row["independent_units"], 2)
            self.assertEqual(row["raw_runs"], 4)
            self.assertEqual(row["ATT_mean"], 25)
            self.assertAlmostEqual(row["ATT_sd"], 2**0.5 * 10)
            self.assertTrue((out / "toy-metrics.pdf").exists())
            self.assertTrue((out / "training-signal.pdf").exists())

    def test_mixed_timing_is_rejected(self):
        from tsc_experiments.visualization import report

        with tempfile.TemporaryDirectory() as path:
            self.fixture(path, 0, 0, 10, 3600)
            self.fixture(path, 1, 0, 20, 1800)
            with self.assertRaisesRegex(ValueError, "Mixed or missing"):
                report(path, Path(path) / "report")

    def test_duplicate_checkpoint_is_rejected(self):
        from tsc_experiments.visualization import report
        import shutil

        with tempfile.TemporaryDirectory() as path:
            folder = self.fixture(path, 0, 0, 10)
            shutil.copytree(folder, Path(path) / "another-checkpoint")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                report(path, Path(path) / "report")


if __name__ == "__main__":
    unittest.main()
