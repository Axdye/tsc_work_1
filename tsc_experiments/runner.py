from __future__ import annotations

from dataclasses import asdict
import importlib.metadata
import random
from contextlib import nullcontext
import json
import time

from .common import ROOT, empty_dir, write_json, append_json, sha
from .environment import TrafficEnvironment, cluster_features
from .branches import branch_pool
from .credit import Collector, StateClusters, joint_sample
from .policy import LLMPolicy, ScriptedPolicy


def manifest(config, kind):
    versions = {}
    for package in (
        "torch",
        "transformers",
        "peft",
        "numpy",
        "pandas",
        "matplotlib",
        "Pillow",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    files = sorted((ROOT / "tsc_experiments").glob("*.py")) + [
        ROOT / "scripts" / name
        for name in (
            "tsc_experiment.py",
            "cpu_upstream_baselines.py",
            "cityflow_diagnostic.py",
        )
    ]
    files += [
        ROOT / "vendor/LLMTSCS" / name
        for name in (
            "utils/config.py",
            "utils/cityflow_env.py",
            "utils/my_utils.py",
            "utils/oneline.py",
            "models/agent.py",
            "models/fixedtime_agent.py",
            "models/maxpressure_agent.py",
        )
    ]
    return {
        "kind": kind,
        "config": config,
        "packages": versions,
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "claims": "new project implementation, not an exact reproduction of Traffic-R1/SocialLight/DGLight",
    }


def evaluate(
    config,
    output,
    dataset=None,
    seed=None,
    policy_name="llm",
    adapter=None,
    policy=None,
    run_metadata=None,
    start_time=0,
):
    out = empty_dir(output)
    dataset = dataset or config["train_dataset"]
    seed = config["seed"] if seed is None else seed
    env = TrafficEnvironment(
        dataset,
        seed,
        out / "engine",
        config["action_interval"],
        config["clearance"],
        config["episode_seconds"],
    )
    if policy_name not in ("llm", "fixedtime", "maxpressure", "scripted", "colight"):
        raise ValueError(f"Unknown policy: {policy_name}")
    if policy_name == "llm" and policy is None:
        policy = LLMPolicy(config, adapter=adapter)
    elif policy_name == "scripted" and policy is None:
        policy = ScriptedPolicy(seed)
    elif policy_name == "colight" and policy is None:
        from .colight import CoLight

        policy = CoLight.load(adapter)
    if start_time:
        if start_time % env.interval or not 0 < start_time < env.horizon:
            raise ValueError("Validation start must be aligned and precede the episode end")
        while env.time < start_time:
            env.step(env.rule_actions())
        for junction in env.env.list_intersection:
            junction.dic_vehicle_arrive_leave_time = {
                vehicle: {"enter_time": start_time, "leave_time": env.np.nan}
                for vehicle, arrival in junction.dic_vehicle_arrive_leave_time.items()
                if env.np.isnan(arrival["leave_time"])
            }
    started = time.perf_counter()
    measurements, invalid, decisions = [], 0, 0
    calls0 = getattr(policy, "calls", 0)
    tokens0 = getattr(policy, "generated_tokens", 0)
    while env.remaining_steps:
        now = env.time
        if policy_name in ("fixedtime", "maxpressure"):
            actions, responses = env.rule_actions(policy_name), []
        elif policy_name == "colight":
            actions, responses = policy.actions(env), []
        else:
            actions, responses = joint_sample(env, policy, greedy=True)
            invalid += sum(r.action is None for r in responses)
        decisions += len(env.names)
        step = env.step(actions)
        measurements.append(step)
        append_json(
            out / "decisions.jsonl",
            {
                "decision_time": now,
                **step,
                "actions_zero_based": dict(zip(env.names, actions)),
                "responses": {n: r.text for n, r in zip(env.names, responses)},
            },
        )
    result = {
        **env.metrics(measurements),
        "dataset": dataset,
        "seed": seed,
        "policy": policy_name,
        "adapter": str(adapter) if adapter else None,
        "invalid_response_rate": invalid / decisions,
        "decision_count": decisions,
        "start_time": start_time,
        "end_time": env.time,
        "wall_seconds": time.perf_counter() - started,
        "llm_calls": getattr(policy, "calls", 0) - calls0,
        "generated_tokens": getattr(policy, "generated_tokens", 0) - tokens0,
        **(run_metadata or {}),
    }
    write_json(out / "metrics.json", result)
    write_json(out / "arrivals.json", env.arrivals())
    write_json(
        out / "manifest.json",
        {
            **manifest(config, "closed-loop evaluation"),
            **env.provenance,
            "metric_note": "ATT: upstream intersection arrival/leave sum per vehicle, horizon censored; "
            "AWT: sampled active-stop timer mean; AQL: sampled NETWORK total queue; "
            "training fork outcomes are never included in these metrics",
        },
    )
    return result



def training_environment(config, output, seed, name):
    return TrafficEnvironment(
        config["train_dataset"], seed, output / name, config["action_interval"],
        config["clearance"], config["train_seconds"],
    )


def validation_run(config, output, policy, round_number, sim_seconds, elapsed):
    if config["train_seconds"] == config["episode_seconds"]:
        return {}
    numpy_state = __import__("numpy").random.get_state()
    try:
        is_colight = config["method"] == "colight"
        context = nullcontext() if is_colight else policy.torch.random.fork_rng(devices=[0])
        with context:
            metrics = evaluate(
                config, output / "validation" / f"round-{round_number:04d}",
                dataset=config["train_dataset"], seed=config["validation_seed"],
                policy=policy, policy_name="colight" if is_colight else "llm",
                start_time=config["train_seconds"],
                run_metadata={"split": "validation", "round": round_number,
                              "sim_seconds": sim_seconds, "training_wall_seconds": elapsed},
            )
    finally:
        __import__("numpy").random.set_state(numpy_state)
    return {"validation_" + key: metrics[key] for key in ("ATT", "AWT", "AQL")}


def train(config, output, scripted=False):
    out = empty_dir(output)
    write_json(out / "config.json", config)
    write_json(out / "manifest.json", manifest(
        config, "scripted plumbing check" if scripted else "work 2 training"
    ))
    random.seed(config["seed"])
    if config["method"] == "colight":
        if scripted:
            raise ValueError("The CoLight baseline requires CPU PyTorch")
        from .colight import train_colight

        return train_colight(
            config, out,
            lambda ep: training_environment(config, out, config["seed"] + ep, f"episodes/{ep:04d}"),
            lambda *args: validation_run(config, out, *args),
        )

    started = time.perf_counter()
    if scripted:
        policy, updater = ScriptedPolicy(config["seed"]), None
    else:
        import torch
        from .trainer import GRPOUpdater

        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        policy = LLMPolicy(config, trainable=True)
        updater = GRPOUpdater(policy, config)
        write_json(out / "model.json", {
            "class": type(policy.model).__name__,
            "base_commit": getattr(policy.model.config, "_commit_hash", None),
            "trainable_parameters": sum(p.numel() for p in updater.parameters),
        })

    active_started = time.perf_counter()
    clusters, total_sim = None, 0
    use_cf = config["method"] == "counterfactual" and config["cf_weight"] > 0
    if use_cf:
        env = training_environment(config, out, config["seed"], "calibration")
        features = []
        for _ in range(config["calibration_steps"]):
            features.extend(cluster_features(o) for o in env.observations())
            env.step(env.rule_actions())
        clusters = StateClusters(env.np, features, config["cluster_count"], config["seed"])
        write_json(out / "clusters.json", clusters.parameters())
        total_sim = env.sim_seconds
        del env

    pool_context = (
        branch_pool(config["branch_workers"], config["group_size"])
        if use_cf and config["branch_workers"] > 1 else nullcontext()
    )
    episode, collector, completed, attempts = -1, None, 0, 0
    elapsed, selected_elapsed, previous_window = time.perf_counter() - active_started, 0.0, 0.0
    limit, deadline = config["max_responses"], config["max_wall_seconds"]
    stopped = "round_limit"
    validated = 0
    with pool_context as pool:
        for round_number in range(1, config["rounds"] + 1):
            window_started = time.perf_counter()
            if deadline and elapsed + previous_window >= deadline:
                stopped = "wall_budget_before_next_window"
                break
            if collector is None or not collector.ready:
                episode += 1
                env = training_environment(
                    config, out, config["seed"] + episode, f"episodes/{episode:04d}"
                )
                collector = Collector(env, policy, config, clusters, pool)
                collector.warmup()
                total_sim += env.sim_seconds
            if limit and attempts + collector.response_cost > limit:
                stopped = "response_budget_before_next_window"
                break

            calls0, tokens0, sim0 = policy.calls, policy.generated_tokens, env.sim_seconds
            sampling_started = time.perf_counter()
            rows, details = collector.collect()
            sampling_wall = time.perf_counter() - sampling_started
            expected_sim = details["main_trajectory_sim_seconds"]
            if pool is None:
                expected_sim += details["branch_sim_seconds"]
            if env.sim_seconds - sim0 != expected_sim:
                raise RuntimeError("Counterfactual simulation accounting mismatch")
            total_sim += sum(details[key] for key in (
                "main_trajectory_sim_seconds", "branch_sim_seconds", "replay_sim_seconds"
            ))
            generated = policy.calls - calls0
            if generated != collector.response_cost or len(rows) != generated:
                raise RuntimeError("LLM response budget accounting mismatch")
            attempts += generated
            for sample_id, row in enumerate(rows):
                row.metadata.update(round=round_number, episode=episode, sample_id=sample_id)
            with (out / "rollouts.jsonl").open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(asdict(row), ensure_ascii=False, allow_nan=False) + "\n")

            backup = [v.detach().cpu().clone() for v in updater.parameters] if updater and deadline else []
            update_started = time.perf_counter()
            learning = updater.update(rows) if updater else {"optimizer_steps": 0}
            if updater:
                policy.torch.cuda.synchronize()
            update_wall = time.perf_counter() - update_started

            previous_window = time.perf_counter() - window_started
            elapsed += previous_window
            selected = not deadline or elapsed <= deadline
            if not selected:
                if updater:
                    with policy.torch.no_grad():
                        for parameter, old in zip(updater.parameters, backup):
                            parameter.copy_(old)
                stopped = "wall_budget_after_window_rollback"
            else:
                completed, selected_elapsed = round_number, elapsed
            metrics = {
                "round": round_number, "episode": episode, "episode_time": env.time,
                "sim_seconds": total_sim, "training_llm_calls": generated,
                "training_generated_tokens": policy.generated_tokens - tokens0,
                "total_training_responses": attempts,
                "wall_seconds": elapsed, "window_wall_seconds": previous_window,
                "update_wall_seconds": update_wall,
                "simulation_and_scoring_wall_seconds": sampling_wall - details["generation_wall_seconds"],
                "invalid_response_fraction": sum(row.action is None for row in rows) / len(rows),
                "nonzero_traffic_advantage_fraction": sum(row.action is not None and abs(row.advantage) > 1e-10 for row in rows) / len(rows),
                "selected_within_budget": selected,
                **details, **learning,
            }
            last = round_number == config["rounds"] or (
                limit and attempts + collector.response_cost > limit
            )
            if selected and updater and (last or round_number % config["eval_every"] == 0):
                metrics.update(validation_run(config, out, policy, round_number, total_sim, elapsed))
                validated = round_number
            append_json(out / "training.jsonl", metrics)
            print(json.dumps(metrics), flush=True)
            if not selected:
                break
            if limit and attempts == limit:
                stopped = "response_budget"
                break

    if completed == 0:
        raise RuntimeError("No full training update fits the configured budget")
    final = None
    if updater:
        if validated != completed:
            validation_run(config, out, policy, completed, total_sim, selected_elapsed)
        final = out / "checkpoints" / "final" / "adapter"
        policy.save(final)
    summary = {
        "stop_reason": stopped, "rounds": completed, "sim_seconds": total_sim,
        "training_responses": attempts, "scripted": scripted,
        "final_adapter": str(final) if final else None,
        "wall_seconds": elapsed, "selected_wall_seconds": selected_elapsed,
        "total_wall_seconds": time.perf_counter() - started,
        "target_wall_seconds": deadline,
    }
    write_json(out / "summary.json", summary)
    return summary
