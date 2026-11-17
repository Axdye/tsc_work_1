"""Held-out high-budget action-ranking reference, not causal ground truth."""

from __future__ import annotations

import itertools
import statistics

from .common import empty_dir, write_json, append_json
from .credit import joint_sample, fork_return
from .environment import TrafficEnvironment
from .policy import LLMPolicy, ScriptedPolicy
from .runner import manifest


def action_values(env, snapshot, target, policy, config, contexts, paired):
    values = [[] for _ in range(4)]
    try:
        for _ in range(contexts):
            env.restore(snapshot)
            if paired:
                background, _ = joint_sample(env, policy)
            for a in range(4):
                env.restore(snapshot)
                if not paired:
                    background, _ = joint_sample(env, policy)
                actions = list(background)
                actions[target] = a
                values[a].append(
                    fork_return(env, snapshot, actions, policy, config, target)
                )
    finally:
        env.restore(snapshot)
    return [statistics.mean(x) for x in values]


def diagnose(
    config,
    output,
    states=4,
    repeats=8,
    reference_contexts=32,
    scripted=False,
    adapter=None,
):
    if min(states, repeats, reference_contexts) < 1:
        raise ValueError("states/repeats/reference_contexts must be positive")
    out = empty_dir(output)
    policy = (
        ScriptedPolicy(config["seed"])
        if scripted
        else LLMPolicy(config, adapter=adapter)
    )
    env = TrafficEnvironment(
        config["train_dataset"],
        config["seed"],
        out / "engine",
        config["action_interval"],
        config["clearance"],
        config["episode_seconds"],
    )
    if config["warmup_steps"] + states > env.remaining_steps:
        raise ValueError("Too many diagnostic states for the episode")
    for _ in range(config["warmup_steps"]):
        env.step(env.rule_actions())
    all_errors = {"paired": [], "unpaired": []}
    agreements = {"paired": [], "unpaired": []}
    variances = {"paired": [], "unpaired": []}
    for s in range(states):
        target, snapshot = s % len(env.names), env.snapshot()
        # Independent reference continuations, never reused as method samples.
        reference = action_values(
            env, snapshot, target, policy, config, reference_contexts, True
        )
        for method in ("paired", "unpaired"):
            estimates = [
                action_values(
                    env,
                    snapshot,
                    target,
                    policy,
                    config,
                    config["contexts"],
                    method == "paired",
                )
                for _ in range(repeats)
            ]
            for a, b in itertools.combinations(range(4), 2):
                delta = reference[a] - reference[b]
                differences = [v[a] - v[b] for v in estimates]
                variances[method].append(statistics.pvariance(differences))
                all_errors[method].extend((d - delta) ** 2 for d in differences)
                if abs(delta) > 1e-6:
                    agreements[method].extend(float(d * delta > 0) for d in differences)
            append_json(
                out / "estimates.jsonl",
                {
                    "state": s,
                    "time": env.time,
                    "target": env.names[target],
                    "method": method,
                    "reference": reference,
                    "estimates": estimates,
                },
            )
        env.restore(snapshot)
        env.step(joint_sample(env, policy)[0])
    summary = {
        method: {
            "pair_difference_mse": statistics.mean(all_errors[method]),
            "pair_difference_variance": statistics.mean(variances[method]),
            "sign_agreement": statistics.mean(agreements[method])
            if agreements[method]
            else None,
        }
        for method in all_errors
    }
    write_json(
        out / "credit_quality.json",
        {
            "summary": summary,
            "states": states,
            "repeats": repeats,
            "reference_contexts": reference_contexts,
            "sim_seconds": env.sim_seconds,
            "scripted": scripted,
            "adapter": str(adapter) if adapter else None,
            "note": "Finite-sample high-budget reference, not a true causal oracle. "
            "Noise includes current neighbor contexts and later policy/environment transitions.",
        },
    )
    write_json(
        out / "manifest.json",
        {**manifest(config, "credit estimator diagnostic"), **env.provenance},
    )
    return summary
