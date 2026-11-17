#!/usr/bin/env python3
"""CPU adapter: LLMLight's actual environment, agents and metric statements.

Avoid unrelated TensorFlow/torch/wandb imports without replacing traffic code.
Only replay output is disabled. Simulator seed is the upstream NumPy RNG draw.
"""
import argparse
import ast
import contextlib
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/LLMTSCS"
DATASETS = {
    "jinan1": ("Jinan", "3_4", "anon_3_4_jinan_real.json"),
    "jinan2": ("Jinan", "3_4", "anon_3_4_jinan_real_2000.json"),
    "jinan3": ("Jinan", "3_4", "anon_3_4_jinan_real_2500.json"),
    "hangzhou1": ("Hangzhou", "4_4", "anon_4_4_hangzhou_real.json"),
    "hangzhou2": ("Hangzhou", "4_4", "anon_4_4_hangzhou_real_5816.json"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def literal_config(name):
    tree = ast.parse((UPSTREAM / "utils/config.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)


def load_upstream(np):
    # Namespace packages let relative imports work without the broad model registry.
    prefix = "_cpu_llmlight"
    for suffix, path in [("", UPSTREAM), (".utils", UPSTREAM / "utils"),
                         (".models", UPSTREAM / "models")]:
        module = types.ModuleType(prefix + suffix)
        module.__path__ = [str(path)]
        sys.modules[module.__name__] = module
    helper_path = UPSTREAM / "utils/my_utils.py"
    helper = types.ModuleType(prefix + ".utils.my_utils")
    helper.__dict__.update(np=np, json=json)
    tree = ast.parse(helper_path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"load_json", "calculate_road_length"}]
    if len(functions) != 2:
        raise RuntimeError("Upstream helper interface changed")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(helper_path), "exec"), helper.__dict__)
    sys.modules[helper.__name__] = helper
    environment = importlib.import_module(prefix + ".utils.cityflow_env")
    agents = {"fixedtime": importlib.import_module(prefix + ".models.fixedtime_agent").FixedtimeAgent,
              "maxpressure": importlib.import_module(prefix + ".models.maxpressure_agent").MaxPressureAgent}
    # Reuse the exact final metric block from OneLine.train, including its censoring.
    metric_path = UPSTREAM / "utils/oneline.py"
    tree = ast.parse(metric_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OneLine")
    train = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, n in enumerate(train.body) if assigns(n, "vehicle_travel_times"))
    end = next(i for i, n in enumerate(train.body) if assigns(n, "results"))
    metric_code = compile(ast.Module(body=train.body[start:end + 1], type_ignores=[]), str(metric_path), "exec")
    return environment, agents, metric_code


def upstream_metrics(metric_code, np, env, horizon, queues, waits, total_reward):
    namespace = {"np": np, "self": types.SimpleNamespace(env=env, dic_traffic_env_conf={"RUN_COUNTS": horizon}),
                 "queue_length_episode": queues, "waiting_time_episode": waits, "total_reward": total_reward}
    exec(metric_code, namespace)
    result = namespace["results"]
    metrics = {"ATT": float(result["test_avg_travel_time_over"]),
               "AWT": float(result["test_avg_waiting_time_over"]),
               "AQL": float(result["test_avg_queue_len_over"])}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RuntimeError(f"Nonfinite metrics: {metrics}")
    return metrics


def run_trial(args, dataset, policy, seed, modules, np, cityflow):
    environment, agent_classes, metric_code = modules
    city, grid, flow_name = DATASETS[dataset]
    data = UPSTREAM / "data" / city / grid
    road_name = f"roadnet_{grid}.json"
    trial = Path(args.output).resolve() / f"{dataset}-{policy}-s{seed}"
    trial.mkdir()
    for name in [flow_name, road_name]:
        shutil.copy2(data / name, trial / name)
    rows, cols = map(int, grid.split("_"))
    config = literal_config("dic_traffic_env_conf")
    config.update(NUM_ROW=rows, NUM_COL=cols, NUM_INTERSECTIONS=rows * cols,
                  NUM_AGENTS=rows * cols, RUN_COUNTS=3600,
                  MODEL_NAME="Fixedtime" if policy == "fixedtime" else "MaxPressure",
                  TRAFFIC_FILE=flow_name, ROADNET_FILE=road_name,
                  LIST_STATE_FEATURE=["cur_phase", "time_this_phase", "traffic_movement_pressure_queue"])
    # Upstream prefixes PATH_TO_DATA with './', so it must be relative to cwd.
    paths = {"PATH_TO_DATA": os.path.relpath(data), "PATH_TO_WORK_DIRECTORY": str(trial)}
    agent_config = literal_config("DIC_FIXEDTIME_AGENT_CONF" if policy == "fixedtime" else "DIC_MAXPRESSURE_AGENT_CONF")
    # Upstream reset draws randint(0,100); keep this mapping and record actual seed.
    np.random.seed(seed)
    def engine_factory(path, thread_num):
        engine_config = json.loads(Path(path).read_text())
        engine_config["saveReplay"] = False
        save(path, engine_config)
        return cityflow.Engine(str(path), thread_num=thread_num)
    environment.engine = types.SimpleNamespace(Engine=engine_factory)
    queues, waits, reward_sum = [], [], 0.0
    started = time.perf_counter()
    with (trial / "upstream.log").open("w") as log, contextlib.redirect_stdout(log):
        env = environment.CityFlowEnv(str(trial), str(trial), config, paths)
        state = env.reset()
        agents = [agent_classes[policy](agent_config, config, paths, 0, str(i)) for i in range(rows * cols)]
        with (trial / "decisions.jsonl").open("w") as stream:
            for step in range(120):
                timestamp = env.get_current_time()
                actions = [int(agent.choose_action(step, obs)) for agent, obs in zip(agents, state)]
                state, rewards, done, _ = env.step(actions)
                if done or env.get_current_time() != (step + 1) * 30:
                    raise RuntimeError("Unexpected upstream episode timing")
                reward_sum += sum(rewards)
                queues.append(sum(sum(i.dic_feature["lane_num_waiting_vehicle_in"]) for i in env.list_intersection))
                values = [v["time"] for v in env.waiting_vehicle_list.values()]
                waits.append(float(np.mean(values)) if values else 0.0)
                stream.write(json.dumps({"time": timestamp, "actions_zero_based": actions,
                                         "AQL_sample": queues[-1], "AWT_sample": waits[-1]}) + "\n")
        metrics = upstream_metrics(metric_code, np, env, 3600, queues, waits, reward_sum)
        # Retain raw arrival/leave records for independent ATT recomputation.
        save(trial / "arrivals.json", {i.inter_name: {
            vehicle: {"enter_time": float(record["enter_time"]),
                      "leave_time": None if np.isnan(record["leave_time"]) else float(record["leave_time"])}
            for vehicle, record in i.dic_vehicle_arrive_leave_time.items()}
            for i in env.list_intersection})
        metrics.update(dataset=dataset, policy=policy, numpy_seed=seed,
                       engine_seed=json.loads((trial / "cityflow.config").read_text())["seed"],
                       end_time=env.get_current_time(), decision_samples=len(queues),
                       remaining_vehicles_including_waiting=len(env.eng.get_vehicles(include_waiting=True)),
                       engine_ATT_at_horizon=env.eng.get_average_travel_time(),
                       wall_seconds=time.perf_counter() - started,
                       roadnet_sha256=sha(data / road_name), flow_sha256=sha(data / flow_name))
        save(trial / "environment.json", config)
        save(trial / "metrics.json", metrics)
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--policies", nargs="+", choices=["fixedtime", "maxpressure"], default=["fixedtime", "maxpressure"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / ".build/cpu-deps"), str(ROOT / ".build/python")]
    import numpy as np
    import pandas as pd
    import cityflow
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    modules = load_upstream(np)
    files = ["utils/config.py", "utils/cityflow_env.py", "utils/my_utils.py", "utils/oneline.py",
             "models/agent.py", "models/fixedtime_agent.py", "models/maxpressure_agent.py"]
    save(output / "manifest.json", {"protocol": "llmlight-public-code-3600s-v1",
        "args": vars(args), "versions": {"numpy": np.__version__, "pandas": pd.__version__, "python": sys.version},
        "engine_path": cityflow.__file__, "engine_sha256": sha(cityflow.__file__),
        "adapter_sha256": sha(__file__), "upstream_sha256": {f: sha(UPSTREAM / f) for f in files},
        "changes": ["Load only two required helper functions and config literals via AST; no fake dependency modules",
                    "Use unchanged upstream environment and agents; disable replay output only",
                    "Reuse exact upstream final metric statements; no wandb or neural models",
                    "Seed NumPy before the upstream simulator-seed draw"],
        "timing": "3600s, 1s engine step, 30s decision interval including 5s phase-0 on changes; no drain; laneChange=True",
        "metrics": {"ATT": "upstream per-intersection arrival/leave durations summed per vehicle; missing leave censored at 3600",
                    "AWT": "time mean of upstream current waiting_vehicle_list time mean at decision ends; NOT completed-trip mean wait",
                    "AQL": "time mean of network total queue over all entering lanes at decision ends; NOT divided by intersections"}})
    results = []
    with (output / "metrics.csv").open("w") as stream:
        writer = None
        for dataset in args.datasets:
            for policy in args.policies:
                for seed in args.seeds:
                    result = run_trial(args, dataset, policy, seed, modules, np, cityflow)
                    results.append(result)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(result))
                        writer.writeheader()
                    writer.writerow(result)
                    stream.flush()
    summary = []
    for dataset in args.datasets:
        for policy in args.policies:
            group = [r for r in results if r["dataset"] == dataset and r["policy"] == policy]
            row = {"dataset": dataset, "policy": policy, "runs": len(group)}
            for metric in ["ATT", "AWT", "AQL"]:
                row[metric + "_mean"] = statistics.mean(r[metric] for r in group)
                row[metric + "_sample_sd"] = statistics.stdev(r[metric] for r in group) if len(group) > 1 else None
            summary.append(row)
    save(output / "summary.json", summary)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
