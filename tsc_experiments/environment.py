"""Unchanged upstream signal/measurement semantics with complete fork restoration.

Uses upstream _inner_step, not a new traffic model. Sampling/logging lives here;
the final three metrics execute the audited upstream metric statements.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
import os
from pathlib import Path
import shutil
import sys
import types

from .common import (
    DIRECTIONS, DIRECTION_NAMES, PHASES, PHASE_APPROACHES, ROOT, SYSTEM,
    read_json, sha, write_json,
)

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


def prompt(observation):
    phase = observation["current_phase"]
    lines = [
        f"路口 {observation['intersection']}，时刻 {observation['time_seconds']} 秒。",
        f"当前相位 {phase if phase in PHASES else '清空'}，已持续 {observation['phase_age_seconds']} 秒。",
        f"控制间隔 {observation['decision_interval_seconds']} 秒，切相时清空 {observation['clearance_seconds']} 秒。",
        "进口方位指车辆从哪一侧驶入。近/中/远是按距停止线将车道分为三等段后的行驶车辆数。",
        "下游按实际接收车道统计。路网出口表示无受控接收车道。停车时间为当前连续停车时长。",
        "相位｜进口｜排队｜近/中/远来车｜平均/最长停车秒数｜下游排队、平均/最高占用",
    ]
    for label, directions in zip(PHASES, PHASE_APPROACHES):
        approaches = observation["phase_observations"][label]["approaches"]
        for direction, approach in zip(directions, approaches):
            near, mid, far = approach["approaching"]
            mean_wait, max_wait = approach["wait_seconds"]
            downstream = approach["downstream"]
            receiving = "路网出口"
            if downstream is not None:
                queued, occupancy, peak = downstream
                receiving = f"{queued}辆、{occupancy:.0%}/{peak:.0%}"
            lines.append(
                f"{label}｜{DIRECTION_NAMES[direction]}｜{approach['queued']}｜"
                f"{near}/{mid}/{far}｜{mean_wait:.0f}/{max_wait:.0f}｜{receiving}"
            )
    neighbors = []
    for direction, neighbor in zip(DIRECTIONS, observation["neighbors_ENWS"]):
        name = DIRECTION_NAMES[direction]
        if neighbor is None:
            neighbors.append(f"{name}侧：路网边界")
        else:
            current = neighbor["current_phase"]
            neighbors.append(
                f"{name}侧：{neighbor['intersection']}，"
                f"相位 {current if current in PHASES else '清空'}，"
                f"总排队 {neighbor['queued']} 辆"
            )
    lines.append("一跳邻居当前观测（各路口随后各自选相位）")
    lines.extend(neighbors)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def inbound_direction(road):
    start, end = road["points"][0], road["points"][-1]
    dx, dy = end["x"] - start["x"], end["y"] - start["y"]
    if abs(dx) > abs(dy):
        return "W" if dx > 0 else "E"
    return "S" if dy > 0 else "N"


def cluster_features(observation):
    phases = observation["phase_observations"]
    return [
        phases[phase][field]
        for field in ("incoming_queued", "incoming_running", "receiving_occupancy")
        for phase in PHASES
    ] + [int(observation["current_phase"] == phase) for phase in PHASES]


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
        self.interval, self.clearance, self.horizon = interval, clearance, horizon
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        road_path, flow_path, grid = dataset_paths(dataset)
        for path in (road_path, flow_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing vendored benchmark: {path}")
            shutil.copy2(path, self.workdir / path.name)
        self.roadnet = read_json(road_path)
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
            TRAFFIC_FILE=flow_path.name,
            ROADNET_FILE=road_path.name,
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
        movements = {
            i["id"]: phase_movements(i)
            for i in self.roadnet["intersections"]
            if not i["virtual"]
        }
        self._validate_phases()
        self.neighbors = [
            tuple(self.env.id_to_index.get(n) for n in
                  self.env.traffic_light_node_dict[name]["neighbor_ENWS"])
            for name in self.names
        ]
        controlled = set(self.names)
        road_lengths, road_directions = {}, {}
        controlled_receivers = set()
        for road in self.roadnet["roads"]:
            points = road["points"]
            road_lengths[road["id"]] = sum(
                math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                for a, b in zip(points, points[1:])
            )
            road_directions[road["id"]] = inbound_direction(road)
            if road["endIntersection"] in controlled:
                controlled_receivers.add(road["id"])
        self.phase_lanes = []
        self.inlet_lengths = {}
        self.receiving_capacity = {}
        for name in self.names:
            phases = []
            for phase, directions in enumerate(PHASE_APPROACHES, 1):
                approaches = []
                for direction in directions:
                    incoming, outgoing = set(), set()
                    for inlet, receivers in movements[name][phase]:
                        road_id = inlet.rsplit("_", 1)[0]
                        if road_directions[road_id] == direction:
                            incoming.add(inlet)
                            outgoing.update(receivers)
                            self.inlet_lengths[inlet] = road_lengths[road_id]
                    receiving = []
                    for lane in sorted(outgoing):
                        road_id = lane.rsplit("_", 1)[0]
                        if road_id in controlled_receivers:
                            receiving.append(lane)
                            self.receiving_capacity[lane] = max(1, int(road_lengths[road_id] / 7.5))
                    approaches.append((sorted(incoming), receiving))
                phases.append(approaches)
            self.phase_lanes.append(phases)
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
            "roadnet_sha256": sha(road_path),
            "flow_sha256": sha(flow_path),
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
                    origin = inbound_direction(roads[link["startRoad"]])
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

    def observations(self):
        current = self.env.system_states
        lane_vehicles = current["get_lane_vehicles"]
        speeds = current["get_vehicle_speed"]
        distances = current["get_vehicle_distance"]
        waiting = self.env.waiting_vehicle_list
        # The running-vehicle map excludes CityFlow's lane-change shadows.
        vehicles = {
            lane: [v for v in lane_vehicles[lane] if v in speeds]
            for lane in self.inlet_lengths.keys() | self.receiving_capacity.keys()
        }
        stopped = {
            lane: [v for v in ids if speeds[v] < 0.1] for lane, ids in vehicles.items()
        }
        arrivals = {}
        for lane, length in self.inlet_lengths.items():
            segments = [0, 0, 0]
            for vehicle in vehicles[lane]:
                if speeds[vehicle] >= 0.1:
                    remaining = max(0.0, length - distances[vehicle])
                    segment = min(2, int(3 * remaining / max(1.0, length)))
                    segments[segment] += 1
            arrivals[lane] = segments

        def downstream(lanes):
            if not lanes:
                return None
            capacity = [self.receiving_capacity[lane] for lane in lanes]
            counts = [len(vehicles[lane]) for lane in lanes]
            return (
                sum(len(stopped[lane]) for lane in lanes),
                min(1.0, sum(counts) / sum(capacity)),
                min(1.0, max(n / c for n, c in zip(counts, capacity))),
            )

        def approach(incoming, outgoing):
            q = sum(len(stopped[lane]) for lane in incoming)
            waits = [
                waiting[v]["time"] if v in waiting and waiting[v]["link"] == lane else 0
                for lane in incoming for v in stopped[lane]
            ]
            return {
                "queued": q,
                "approaching": tuple(sum(arrivals[lane][j] for lane in incoming) for j in range(3)),
                "wait_seconds": (sum(waits) / q, max(waits)) if q else (0, 0),
                "downstream": downstream(outgoing),
            }

        nodes = self.env.list_intersection
        neighborhood = [
            {
                "intersection": name,
                "current_phase": PHASES[node.current_phase_index - 1]
                if node.current_phase_index in range(1, 5) else "clearance",
                "queued": queue,
            }
            for name, node, queue in zip(self.names, nodes, self.queues())
        ]
        observations = []
        for index, node in enumerate(nodes):
            features = {}
            for label, lanes in zip(PHASES, self.phase_lanes[index]):
                approaches = [approach(incoming, outgoing) for incoming, outgoing in lanes]
                receivers = sorted({lane for _, outgoing in lanes for lane in outgoing})
                occupancy = downstream(receivers)
                features[label] = {
                    "approaches": approaches,
                    "incoming_queued": sum(a["queued"] for a in approaches),
                    "incoming_running": sum(sum(a["approaching"]) for a in approaches),
                    "receiving_occupancy": occupancy[1] if occupancy is not None else 0.0,
                }
            observations.append({
                "intersection": self.names[index],
                "time_seconds": self.time,
                "current_phase": neighborhood[index]["current_phase"],
                "phase_age_seconds": max(0, node.current_phase_duration),
                "decision_interval_seconds": self.interval,
                "clearance_seconds": self.clearance,
                "phase_observations": features,
                "neighbors_ENWS": [
                    neighborhood[n] if n is not None else None for n in self.neighbors[index]
                ],
            })
        return observations

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
