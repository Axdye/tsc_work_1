"""Fixed-horizon branching, with native snapshot reuse inside each engine."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import time

from .environment import TrafficEnvironment


def step_reward(step):
    return -float(sum(step["queue_costs"]))


def branch_return(env, first_actions, horizon, gamma, continuation=None):
    if env.remaining_steps < horizon:
        raise ValueError("A complete counterfactual horizon must fit before the cutoff")
    choose = continuation or env.rule_actions
    first_reward = step_reward(env.step(first_actions))
    total = first_reward
    for step in range(1, horizon):
        total += gamma ** step * step_reward(env.step(choose()))
    return total, first_reward


def score_slice(env, actual, horizon, gamma):
    snapshot = env.snapshot()
    try:
        common, first_reward = branch_return(env, actual, horizon, gamma)
        scores = []
        for i, phase in enumerate(actual):
            actions = list(actual)
            values = [common] * 4
            for candidate in range(4):
                if candidate != phase:
                    env.restore(snapshot)
                    actions[i] = candidate
                    values[candidate], _ = branch_return(env, actions, horizon, gamma)
            scores.append(values)
        return scores, first_reward
    finally:
        env.restore(snapshot)


_worker = None


def score_trajectory(env_args, workdir, prefix, trajectory, horizon, gamma):
    """Materialize a trajectory in a worker whose archives belong to its own engine."""
    global _worker
    started = time.process_time()
    identity = (env_args, workdir)
    if _worker is None or _worker[0] != identity:
        dataset, seed, interval, clearance, cutoff = env_args
        env = TrafficEnvironment(
            dataset, seed, Path(workdir) / f"worker-{os.getpid()}" / f"seed-{seed}",
            interval, clearance, cutoff,
        )
        _worker = identity, env, env.snapshot(), 0
    _, env, anchor, position = _worker
    env.restore(anchor)
    for actions in prefix[position:]:
        env.step(actions)
    if len(prefix) != position:
        _worker = identity, env, env.snapshot(), len(prefix)
    values = []
    for actions, expected in trajectory:
        scores, first_reward = score_slice(env, actions, horizon, gamma)
        actual = env.step(actions)
        if actual["queue_costs"] != expected or first_reward != step_reward(actual):
            raise RuntimeError("Counterfactual worker failed deterministic snapshot replay")
        values.append(scores)
    return values, (len(prefix) - position + len(trajectory)) * env.interval, time.process_time() - started


def branch_pool(workers, group_size):
    return ProcessPoolExecutor(
        max_workers=min(workers, group_size),
        mp_context=multiprocessing.get_context("spawn"),
    )
