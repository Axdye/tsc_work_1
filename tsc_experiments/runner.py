from __future__ import annotations

from dataclasses import asdict
import importlib.metadata
import random
import time

from .common import ROOT, empty_dir, write_json, append_json, sha
from .environment import TrafficEnvironment
from .credit import Collector, joint_sample
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
            "minimal_cityflow_grpo.py",
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
    if policy_name not in ("llm", "fixedtime", "maxpressure", "scripted"):
        raise ValueError(f"Unknown policy: {policy_name}")
    if policy_name == "llm" and policy is None:
        policy = LLMPolicy(config, adapter=adapter)
    elif policy_name == "scripted" and policy is None:
        policy = ScriptedPolicy(seed)
    started = time.perf_counter()
    measurements, invalid, decisions = [], 0, 0
    calls0 = policy.calls if policy else 0
    tokens0 = policy.generated_tokens if policy else 0
    while env.remaining_steps:
        now = env.time
        if policy_name in ("fixedtime", "maxpressure"):
            actions, responses = env.rule_actions(policy_name), []
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
        "end_time": env.time,
        "wall_seconds": time.perf_counter() - started,
        "llm_calls": policy.calls - calls0 if policy else 0,
        "generated_tokens": policy.generated_tokens - tokens0 if policy else 0,
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


def save_checkpoint(policy, updater, config, folder, round_number, sim_seconds):
    folder.mkdir(parents=True)
    policy.save(folder / "adapter")
    import torch

    torch.save(
        {
            "optimizer": updater.optimizer.state_dict(),
            "round": round_number,
            "sim_seconds": sim_seconds,
            "optimizer_steps": updater.optimizer_steps,
        },
        folder / "optimizer.pt",
    )
    # This is an inference checkpoint + optimizer archive, not mid-episode resume.
    write_json(
        folder / "training_state.json",
        {
            "round": round_number,
            "sim_seconds": sim_seconds,
            "config": config,
            "resume_supported": False,
        },
    )


def train(config, output, scripted=False):
    out = empty_dir(output)
    write_json(out / "config.json", config)
    write_json(
        out / "manifest.json",
        manifest(config, "scripted plumbing check" if scripted else "online LoRA GRPO"),
    )
    random.seed(config["seed"])
    if scripted:
        policy, updater = ScriptedPolicy(config["seed"]), None
    else:
        import torch
        from .trainer import GRPOUpdater

        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        policy = LLMPolicy(config, trainable=True)
        updater = GRPOUpdater(policy, config)
        write_json(
            out / "model.json",
            {
                "class": type(policy.model).__name__,
                "base_commit": getattr(policy.model.config, "_commit_hash", None),
                "trainable_parameters": sum(
                    p.numel() for p in policy.model.parameters() if p.requires_grad
                ),
            },
        )
    total_sim, episode, collector = 0, -1, None
    stopped = "round_limit"
    started = time.perf_counter()
    best, best_path = float("inf"), None
    for round_number in range(1, config["rounds"] + 1):
        if collector is None or collector.env.remaining_steps == 0:
            episode += 1
            env = TrafficEnvironment(
                config["train_dataset"],
                config["seed"] + episode,
                out / "episodes" / f"episode-{episode:04d}",
                config["action_interval"],
                config["clearance"],
                config["episode_seconds"],
            )
            collector = Collector(env, policy, config, config["seed"] + episode)
            # Counterfactual groups and visited states are regenerated each round.
            warmup = min(config["warmup_steps"], env.remaining_steps)
            if (
                config["max_sim_seconds"]
                and total_sim + warmup * env.interval > config["max_sim_seconds"]
            ):
                stopped = "budget_before_warmup"
                break
            for _ in range(warmup):
                env.step(env.rule_actions())
            total_sim += warmup * env.interval
        env = collector.env
        expected = collector.window_cost()
        if (
            config["max_sim_seconds"]
            and total_sim + expected > config["max_sim_seconds"]
        ):
            stopped = "budget_before_next_window"
            break
        before = env.sim_seconds
        calls0, tokens0 = policy.calls, policy.generated_tokens
        rows, diagnostics = collector.collect()
        total_sim += env.sim_seconds - before
        if env.sim_seconds - before != expected:
            raise RuntimeError(
                f"Budget accounting mismatch: expected {expected}, got {env.sim_seconds - before}"
            )
        # Write the actual sampled tokens and scores before any optimizer update.
        for sample_number, row in enumerate(rows):
            row.metadata.update(
                round=round_number,
                episode=episode,
                sample_id=f"r{round_number}-s{sample_number}",
                group=f"r{round_number}-{row.metadata['group']}",
            )
            append_json(out / "rollouts.jsonl", asdict(row))
        for row in diagnostics:
            row["group"] = f"r{round_number}-{row['group']}"
            append_json(
                out / "groups.jsonl", {"round": round_number, "episode": episode, **row}
            )
        learning = (
            updater.update(rows)
            if updater
            else {"responses": len(rows), "optimizer_steps": 0}
        )
        metrics = {
            "round": round_number,
            "episode": episode,
            "episode_time": env.time,
            "sim_seconds": total_sim,
            "window_sim_seconds": env.sim_seconds - before,
            "training_llm_calls": policy.calls - calls0,
            "training_generated_tokens": policy.generated_tokens - tokens0,
            "zero_adv_group_fraction": sum(g["zero_advantage"] for g in diagnostics)
            / len(diagnostics),
            "wall_seconds": time.perf_counter() - started,
            **learning,
        }
        if not scripted and round_number % config["eval_every"] == 0:
            # Evaluation changes no policy weights and must not consume behavior RNG.
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            python_rng, numpy_rng = random.getstate(), env.np.random.get_state()
            try:
                validation = evaluate(
                    config,
                    out / "validation" / f"round-{round_number:04d}",
                    dataset=config["validation_dataset"],
                    seed=config["validation_seed"],
                    policy=policy,
                    run_metadata={
                        "method": config["method"],
                        "train_seed": config["seed"],
                        "round": round_number,
                        "sim_seconds": total_sim,
                        "split": "validation",
                    },
                )
            finally:
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(gpu_rng)
                random.setstate(python_rng)
                env.np.random.set_state(numpy_rng)
            for key in ("ATT", "AWT", "AQL"):
                metrics["validation_" + key] = validation[key]
            if validation["ATT"] < best:
                best = validation["ATT"]
                # Versioned checkpoints, never silently overwrite earlier best.
                best_path = out / "checkpoints" / f"best-round-{round_number:04d}"
                save_checkpoint(
                    policy, updater, config, best_path, round_number, total_sim
                )
        if not scripted and round_number % config["save_every"] == 0:
            save_checkpoint(
                policy,
                updater,
                config,
                out / "checkpoints" / f"round-{round_number:04d}",
                round_number,
                total_sim,
            )
        metrics["wall_seconds"] = time.perf_counter() - started
        append_json(out / "training.jsonl", metrics)
        print(__import__("json").dumps(metrics), flush=True)
    if not scripted:
        save_checkpoint(
            policy,
            updater,
            config,
            out / "checkpoints" / "final",  # inference-only final is always available
            metrics["round"] if (out / "training.jsonl").exists() else 0,
            total_sim,
        )
    summary = {
        "stop_reason": stopped,
        "sim_seconds": total_sim,
        "episodes_created": episode + 1,
        "scripted": scripted,
        "best_adapter": str(best_path / "adapter") if best_path else None,
        "final_adapter": None if scripted else str(out / "checkpoints/final/adapter"),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(out / "summary.json", summary)
    return summary
