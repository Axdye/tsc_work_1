"""Small matched MaxPressure versus greedy LLM continuation diagnostic."""

from collections import Counter
from pathlib import Path
import random

from .branches import branch_return
from .common import empty_dir, read_json, write_json
from .credit import executed_actions, joint_sample, loo_probabilities
from .environment import TrafficEnvironment, cluster_features, prompt
from .policy import LLMPolicy, ScriptedPolicy


def diagnose(config, output, states=12, scripted=False, training_run=None):
    if states <= 0 or not scripted and not training_run:
        raise ValueError("Provide positive --states and --training-run for a learned diagnostic")
    path = Path(training_run) if training_run else None
    centers = read_json(path / "clusters.json") if path else None
    adapter = read_json(path / "summary.json")["final_adapter"] if path else None
    if path and not adapter and not scripted:
        raise ValueError("Reference training run has no final LLM checkpoint")
    out = empty_dir(output)
    policy = ScriptedPolicy(config["seed"]) if scripted else LLMPolicy(config, adapter=adapter)
    env = TrafficEnvironment(
        config["train_dataset"], config["validation_seed"], out / "engine",
        config["action_interval"], config["clearance"], config["episode_seconds"],
    )
    while env.time < config["train_seconds"]:
        env.step(env.rule_actions())
    global_counts, group_counts = Counter(), {}
    available = []
    horizon, gamma = config["horizon"], config["gamma"]
    rng = random.Random(config["seed"])
    while env.remaining_steps:
        observations = env.observations()
        responses = policy.sample([prompt(o) for o in observations], greedy=True)
        actions = executed_actions(responses, observations)
        groups = [0] * len(responses)
        if centers:
            np = env.np
            x = (np.asarray([cluster_features(o) for o in observations]) - centers["mean"]) / centers["scale"]
            groups = ((x[:, None] - np.asarray(centers["centers"])) ** 2).sum(2).argmin(1).tolist()
        for row, group in zip(responses, groups):
            if row.action is not None:
                global_counts[row.action] += 1
                group_counts.setdefault(group, Counter())[row.action] += 1
        targets = [i for i, o in enumerate(observations)
                   if responses[i].action is not None and sum(cluster_features(o)[:8]) > 0]
        if targets and env.remaining_steps >= horizon:
            i = rng.choice(targets)
            available.append((env.time, env.snapshot(), actions, i, groups[i]))
        env.step(actions)
    if not available:
        raise RuntimeError("No nonempty valid validation states fit the full horizon")
    collection_calls, reference_calls = policy.calls, 0
    results = []
    for now, snapshot, actions, i, group in rng.sample(available, min(states, len(available))):
        start_calls = policy.calls
        values = []
        actual = actions[i]
        counts = group_counts[group]
        if sum(counts.values()) < config["cluster_min"]:
            counts = global_counts
        weights = loo_probabilities(counts, actual, config["cluster_alpha"])
        for continuation in (None, lambda: joint_sample(env, policy, greedy=True)[0]):
            candidates = []
            for a in range(4):
                env.restore(snapshot)
                intervention = list(actions)
                intervention[i] = a
                score, _ = branch_return(env, intervention, horizon, gamma, continuation)
                candidates.append(score)
            credit = sum(w * (candidates[actual] - q) for w, q in zip(weights, candidates))
            values.append((candidates, credit))
        reference_calls += policy.calls - start_calls
        (mp, mp_credit), (llm, llm_credit) = values
        sign = lambda x: 0 if abs(x) <= 1e-8 else (1 if x > 0 else -1)
        top = lambda x: [i for i, v in enumerate(x) if abs(v - max(x)) <= 1e-8]
        results.append({
            "time": now, "intersection": env.names[i], "actual_phase": actual,
            "maxpressure_returns": mp, "llm_returns": llm,
            "maxpressure_credit": mp_credit, "llm_credit": llm_credit,
            "sign_agreement": sign(mp_credit) == sign(llm_credit),
            "top1_agreement": top(mp)[0] == top(llm)[0],
            "top1_tied": len(top(mp)) > 1 or len(top(llm)) > 1,
        })
    summary = {
        "scripted": scripted, "samples": len(results),
        "sign_agreement": sum(r["sign_agreement"] for r in results) / len(results),
        "top1_agreement": sum(r["top1_agreement"] for r in results) / len(results),
        "top1_ties": sum(r["top1_tied"] for r in results),
        "validation_collection_responses": collection_calls,
        "reference_continuation_responses": reference_calls,
        "cases": results,
    }
    write_json(out / "credit_quality.json", summary)
    return summary
