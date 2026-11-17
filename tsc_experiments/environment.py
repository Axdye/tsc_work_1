"""Unchanged upstream signal/measurement semantics with complete fork restoration.

Uses upstream _inner_step, not a new traffic model. Sampling/logging lives here;
the final three metrics execute the audited upstream metric statements.
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
from pathlib import Path
import shutil
import sys
import types

from .common import ROOT, PHASES, SYSTEM, read_json, write_json, sha

DATASETS = {
    "synthetic8000": ("CoLLMLight", "Synthetic", "4_4", "anon_4_4_synthetic_8000.json"),
    "jinan1": ("LLMTSCS", "Jinan", "3_4", "anon_3_4_jinan_real.json"),
    "jinan2": ("LLMTSCS", "Jinan", "3_4", "anon_3_4_jinan_real_2000.json"),
    "jinan3": ("LLMTSCS", "Jinan", "3_4", "anon_3_4_jinan_real_2500.json"),
    "hangzhou1": ("LLMTSCS", "Hangzhou", "4_4", "anon_4_4_hangzhou_real.json"),
    "hangzhou2": ("LLMTSCS", "Hangzhou", "4_4", "anon_4_4_hangzhou_real_5816.json"),
    "newyork_double": (
        "LLMTSCS",
        "NewYork",
        "28_7",
        "anon_28_7_newyork_real_double.json",
    ),
    "newyork_triple": (
        "LLMTSCS",
        "NewYork",
        "28_7",
        "anon_28_7_newyork_real_triple.json",
    ),
}


def dataset_paths(name):
    repo, city, grid, flow = DATASETS[name]
    folder = ROOT / "vendor" / repo / "data" / city / grid
    return folder / f"roadnet_{grid}.json", folder / flow, grid


def enable_local_dependencies():
    # Installed server packages win; the local extension is ABI-specific.
    import importlib.util

    for module, folder in (
        ("numpy", "cpu-deps"),
        ("pandas", "cpu-deps"),
        ("cityflow", "python"),
    ):
        if importlib.util.find_spec(module) is None:
            sys.path.append(str(ROOT / ".build" / folder))
    sys.path.insert(0, str(ROOT / "scripts"))


class TrafficEnvironment:
    def __init__(self, dataset, seed, workdir, interval=30, clearance=5, horizon=3600):
        enable_local_dependencies()
        import numpy as np
        import cityflow
        from cpu_upstream_baselines import load_upstream, literal_config
        from cityflow_diagnostic import phase_movements

        if interval <= clearance or clearance < 0 or horizon % interval:
            raise ValueError("Invalid interval/clearance/horizon")
        self.np = np
        self.dataset, self.seed = dataset, int(seed)
        self.interval, self.horizon = interval, horizon
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        road, flow, grid = dataset_paths(dataset)
        for path in (road, flow):
            if not path.is_file():
                raise FileNotFoundError(f"Missing vendored benchmark: {path}")
            shutil.copy2(path, self.workdir / path.name)
        self.roadnet = read_json(road)
        rows, cols = map(int, grid.split("_"))
        conf = literal_config("dic_traffic_env_conf")
        conf.update(
            NUM_ROW=rows,
            NUM_COL=cols,
            NUM_INTERSECTIONS=rows * cols,
            NUM_AGENTS=rows * cols,
            RUN_COUNTS=horizon,
            MODEL_NAME="CreditGRPO",
            MIN_ACTION_TIME=interval,
            YELLOW_TIME=clearance,
            ALL_RED_TIME=0,
            TRAFFIC_FILE=flow.name,
            ROADNET_FILE=road.name,
            LIST_STATE_FEATURE=[
                "cur_phase",
                "time_this_phase",
                "traffic_movement_pressure_queue",
            ],
        )
        paths = {
            "PATH_TO_DATA": os.path.relpath(self.workdir),
            "PATH_TO_WORK_DIRECTORY": str(self.workdir),
        }
        module, self.agent_classes, self.metric_code = load_upstream(np)

        def engine_factory(path, thread_num):
            c = read_json(path)
            c["saveReplay"] = False
            write_json(path, c)
            return cityflow.Engine(str(path), thread_num=thread_num)

        module.engine = types.SimpleNamespace(Engine=engine_factory)
        np.random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            self.env = module.CityFlowEnv(
                str(self.workdir), str(self.workdir), conf, paths
            )
            self.env.reset()
        self.names = [i.inter_name for i in self.env.list_intersection]
        self.movements = {
            i["id"]: phase_movements(i)
            for i in self.roadnet["intersections"]
            if not i["virtual"]
        }
        self._validate_phases()
        self.agents = {
            kind: [
                cls(
                    literal_config(
                        "DIC_FIXEDTIME_AGENT_CONF"
                        if kind == "fixedtime"
                        else "DIC_MAXPRESSURE_AGENT_CONF"
                    ),
                    conf,
                    paths,
                    0,
                    str(i),
                )
                for i in range(len(self.names))
            ]
            for kind, cls in self.agent_classes.items()
        }
        self.sim_seconds = 0  # Physical simulation work, including restored branches.
        self.provenance = {
            "dataset": dataset,
            "numpy_seed": seed,
            "engine_seed": read_json(self.workdir / "cityflow.config")["seed"],
            "roadnet_sha256": sha(road),
            "flow_sha256": sha(flow),
            "engine_file": cityflow.__file__,
            "engine_sha256": sha(cityflow.__file__),
            "upstream_env_sha256": sha(ROOT / "vendor/LLMTSCS/utils/cityflow_env.py"),
            "metric_source_sha256": sha(ROOT / "vendor/LLMTSCS/utils/oneline.py"),
            "protocol": "llmlight-public-code-3600s-v1"
            if (interval, clearance, horizon) == (30, 5, 3600)
            else "llmlight-code-custom-timing-v1",
            "interval": interval,
            "clearance_within_interval": clearance,
            "horizon": horizon,
            "laneChange": True,
            "no_drain": True,
        }
        write_json(self.workdir / "provenance.json", self.provenance)

    def _validate_phases(self):
        roads = {r["id"]: r for r in self.roadnet["roads"]}
        expected = [
            {("E", "go_straight"), ("W", "go_straight")},
            {("N", "go_straight"), ("S", "go_straight")},
            {("E", "turn_left"), ("W", "turn_left")},
            {("N", "turn_left"), ("S", "turn_left")},
        ]
        for node in self.roadnet["intersections"]:
            if node["virtual"]:
                continue
            for phase, want in enumerate(expected, 1):
                actual = set()
                for index in node["trafficLight"]["lightphases"][phase][
                    "availableRoadLinks"
                ]:
                    link = node["roadLinks"][index]
                    if link["type"] == "turn_right":
                        continue
                    p = roads[link["startRoad"]]["points"]
                    dx, dy = p[-1]["x"] - p[0]["x"], p[-1]["y"] - p[0]["y"]
                    origin = (
                        ("W" if dx > 0 else "E")
                        if abs(dx) > abs(dy)
                        else ("S" if dy > 0 else "N")
                    )
                    actual.add((origin, link["type"]))
                if actual != want:
                    raise ValueError(
                        f"Unsupported phase mapping: {node['id']}, {phase}: {actual}"
                    )

    @property
    def time(self):
        return int(self.env.eng.get_current_time())

    @property
    def remaining_steps(self):
        return max(0, (self.horizon - self.time) // self.interval)

    def snapshot(self):
        engine = self.env.eng
        return (
            engine.snapshot(),
            copy.deepcopy(self.env.__dict__, {id(engine): engine}),
            copy.deepcopy(self.np.random.get_state()),
            {k: [copy.deepcopy(a.__dict__) for a in v] for k, v in self.agents.items()},
        )

    def restore(self, snapshot):
        archive, python_state, numpy_state, agents = snapshot
        engine = self.env.eng
        engine.load(archive)
        state = copy.deepcopy(python_state, {id(engine): engine})
        self.env.__dict__.clear()
        self.env.__dict__.update(state)
        self.np.random.set_state(numpy_state)
        for kind, values in agents.items():
            for agent, value in zip(self.agents[kind], values):
                agent.__dict__ = copy.deepcopy(value)

    def queues(self):
        return [
            sum(i.dic_feature["lane_num_waiting_vehicle_in"])
            for i in self.env.list_intersection
        ]

    def measurement(self):
        waits = [v["time"] for v in self.env.waiting_vehicle_list.values()]
        return {
            "time": self.time,
            "AQL_sample": float(sum(self.queues())),
            "AWT_sample": float(self.np.mean(waits)) if waits else 0.0,
            "junction_queues": dict(zip(self.names, map(float, self.queues()))),
            "signal_phases": {
                i.inter_name: max(0, i.current_phase_index)
                for i in self.env.list_intersection
            },
        }

    def step(self, actions):
        if len(actions) != len(self.names) or any(a not in range(4) for a in actions):
            raise ValueError("Need one zero-based phase 0..3 per intersection")
        if self.remaining_steps == 0:
            raise RuntimeError("Episode horizon reached")
        integral = [0.0] * len(self.names)
        # Same signal/vehicle tracking and measurements as upstream env.step.
        # Skip only upstream per-second pickle feature logs and stdout.
        for _ in range(self.interval):
            self.env.current_time = self.time
            self.env._inner_step(actions)
            for i, q in enumerate(self.queues()):
                integral[i] += q
        self.sim_seconds += self.interval
        return {
            "queue_costs": [x / self.interval for x in integral],
            **self.measurement(),
        }

    def rule_actions(self, kind="maxpressure"):
        state, _ = self.env.get_state()
        return [
            int(agent.choose_action(self.time // self.interval, obs))
            for agent, obs in zip(self.agents[kind], state)
        ]

    def prompt(self, index):
        name = self.names[index]
        q = self.env.eng.get_lane_waiting_vehicle_count()
        counts = self.env.eng.get_lane_vehicle_count()
        features = {}
        for action, label in enumerate(PHASES):
            pairs = self.movements[name][action + 1]
            incoming = {a for a, _ in pairs}
            outgoing = {b for _, bs in pairs for b in bs}
            features[label] = {
                "incoming_queued": sum(q.get(x, 0) for x in incoming),
                "incoming_running": sum(
                    max(0, counts.get(x, 0) - q.get(x, 0)) for x in incoming
                ),
                "receiving_queued": sum(q.get(x, 0) for x in outgoing),
                "receiving_vehicles": sum(counts.get(x, 0) for x in outgoing),
            }
        node = self.env.list_intersection[index]
        neighbors = self.env.traffic_light_node_dict[name]["neighbor_ENWS"]
        if isinstance(neighbors, dict):
            neighbors = list(neighbors.values())
        demand = {
            n: float(
                sum(
                    self.env.list_intersection[self.env.id_to_index[n]].dic_feature[
                        "lane_num_waiting_vehicle_in"
                    ]
                )
            )
            for n in neighbors
            if n in self.env.id_to_index
        }
        phase = node.current_phase_index
        observation = {
            "intersection": name,
            "time_seconds": self.time,
            "current_phase": PHASES[phase - 1] if phase in range(1, 5) else "clearance",
            "decision_interval_seconds": self.interval,
            "phase_observations": features,
            "neighbor_total_queues": demand,
        }
        import json

        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(observation, ensure_ascii=False)},
        ]

    def metrics(self, measurements):
        from cpu_upstream_baselines import upstream_metrics

        result = upstream_metrics(
            self.metric_code,
            self.np,
            self.env,
            self.time,
            [m["AQL_sample"] for m in measurements],
            [m["AWT_sample"] for m in measurements],
            0.0,
        )
        result.update(
            remaining_vehicles_including_waiting=len(
                self.env.eng.get_vehicles(include_waiting=True)
            ),
            engine_ATT_diagnostic=float(self.env.eng.get_average_travel_time()),
        )
        return result

    def arrivals(self):
        return {
            i.inter_name: {
                v: {
                    "enter_time": float(r["enter_time"]),
                    "leave_time": None
                    if self.np.isnan(r["leave_time"])
                    else float(r["leave_time"]),
                }
                for v, r in i.dic_vehicle_arrive_leave_time.items()
            }
            for i in self.env.list_intersection
        }
