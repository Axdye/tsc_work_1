import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import reproduce
from cityflow_diagnostic import phase_movements, scores, clipped_flows


class ReproductionTests(unittest.TestCase):
    def test_phase_pressure_does_not_triple_count_one_incoming_lane(self):
        link = {'type': 'go_straight', 'startRoad': 'in', 'endRoad': 'out',
                'laneLinks': [{'startLaneIndex': 1, 'endLaneIndex': j} for j in range(3)]}
        inter = {'roadLinks': [link], 'trafficLight': {'lightphases':
                 [{'availableRoadLinks': []}] + [{'availableRoadLinks': [0]}] * 4}}
        moves = phase_movements(inter)
        self.assertEqual(scores(moves, {'in_1': 10, 'out_0': 3, 'out_1': 3, 'out_2': 3})[1], 7)

    def test_demand_clipping_uses_integer_inclusive_end_expected_by_engine(self):
        flows = [{'startTime': 0, 'endTime': -1, 'interval': 1},
                 {'startTime': 100, 'endTime': 200, 'interval': 1}]
        self.assertEqual(clipped_flows(flows, 100), [{'startTime': 0, 'endTime': 99, 'interval': 1}])
        self.assertEqual(flows[0]['endTime'], -1)

    def test_launcher_does_not_silently_reconstruct_missing_methods(self):
        args = SimpleNamespace(method='oracletsc')
        with self.assertRaisesRegex(ValueError, 'No verified official'):
            reproduce.build_native_plan(args, ROOT / 'outputs/test')

    def test_native_paths_and_forwarded_model_arguments(self):
        args = SimpleNamespace(method='collmlight', python='/python/env', dataset='jinan', flow=None,
                               extra=['--model_path', '/models/a path'])
        plan = reproduce.build_native_plan(args, ROOT / 'outputs/test')
        self.assertEqual(plan['command'][-2:], ['--model_path', '/models/a path'])
        self.assertIn('anon_3_4_jinan_real.json', plan['command'])

    def test_upstream_ppo_two_records_per_state_is_visible(self):
        path = ROOT / 'vendor/CoLLMLight/ppo_ft.py'
        if not path.exists():
            self.skipTest('Download CoLLMLight first')
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_ppo_data_from_offline')
        class Encoded:
            def squeeze(self): return self
            def to(self, device): return self
        namespace = {'torch': SimpleNamespace(tensor=lambda value, device: value)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        tokenizer = SimpleNamespace(encode=lambda *a, **kw: Encoded())
        sample = {'prompt_atr': '', 'response_atr': '', 'prompt_ra': '', 'response_ra': '',
                  'reward_atr': 1., 'reward_ra': 1.}
        q, r, rew = namespace[fn.name]([sample] * 32, tokenizer, 'cpu')
        self.assertEqual((len(q), len(r), len(rew)), (64, 64, 64))


if __name__ == '__main__':
    unittest.main()
