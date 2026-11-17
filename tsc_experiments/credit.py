"""Rollout credit variants. Every variant uses the same prompts and token actor."""

from __future__ import annotations

from dataclasses import asdict
import statistics

from .common import PHASES, standardize, returns_to_go, target_indices


def executed_action(response, env, i):
    if response.action is not None:
        return response.action
    phase = env.env.list_intersection[i].current_phase_index
    return phase - 1 if phase in range(1, 5) else 0


def joint_sample(env, policy, greedy=False):
    responses = policy.sample(
        [env.prompt(i) for i in range(len(env.names))], greedy=greedy
    )
    return [executed_action(r, env, i) for i, r in enumerate(responses)], responses


def step_reward(step, scope="global", target=None):
    costs = step["queue_costs"]
    return -float(sum(costs) if scope == "global" else costs[target])


def continuation_actions(env, policy, c):
    return (
        env.rule_actions()
        if c["continuation"] == "maxpressure"
        else joint_sample(env, policy)[0]
    )


def fork_return(env, snapshot, first_actions, policy, c, target=None):
    env.restore(snapshot)
    value, steps = 0.0, min(c["horizon"], env.remaining_steps)
    for h in range(steps):
        actions = first_actions if h == 0 else continuation_actions(env, policy, c)
        value += c["gamma"] ** h * step_reward(
            env.step(actions), c["reward_scope"], target
        )
    return value


def label_group(responses, values, group_id, invalid_penalty):
    # Same physical action: same averaged traffic score even if wording differs.
    by_action = {}
    for r, v in zip(responses, values):
        if r.action is not None:
            by_action.setdefault(r.action, []).append(v)
    table = {a: statistics.mean(v) for a, v in by_action.items()}
    rewards = [
        table[r.action] if r.action is not None else v - invalid_penalty
        for r, v in zip(responses, values)
    ]
    advantages = standardize(rewards)
    for r, reward, adv in zip(responses, rewards, advantages):
        r.reward, r.advantage = reward, adv
        r.metadata.update(
            group=group_id, action_scores={PHASES[a]: v for a, v in table.items()}
        )
    return responses


class Collector:
    def __init__(self, env, policy, config, seed):
        import random

        self.env, self.policy, self.c = env, policy, config
        self.random = random.Random(seed)
        self.offset = 0
        self.group_counter = 0

    def _id(self):
        self.group_counter += 1
        return f"g{self.group_counter}"

    def window_cost(self):
        e, c = self.env, self.c
        steps = min(c["collect_steps"], e.remaining_steps)
        g, h, m = c["group_size"], c["horizon"], c["contexts"]
        if c["method"] in ("trajectory", "rtg"):
            return steps * e.interval * g
        targets = min(len(e.names), c["targets_per_state"])
        cost = sum(min(h, e.remaining_steps - s) for s in range(steps))
        branches = targets * m * g if c["method"] in ("paired", "unpaired") else g
        return (cost * branches + steps) * e.interval

    def collect(self):
        if self.c["method"] in ("trajectory", "rtg"):
            return self._trajectories()
        return self._branch_windows()

    def _branch_windows(self):
        e, p, c = self.env, self.policy, self.c
        rows, diagnostics = [], []
        for _ in range(min(c["collect_steps"], e.remaining_steps)):
            # Current LLM policy visits the next state; never deploy a rollout argmax.
            actual_actions, _ = joint_sample(e, p)
            snapshot = e.snapshot()
            time = e.time
            targets = target_indices(len(e.names), c["targets_per_state"], self.offset)
            self.offset += len(targets)
            try:
                if c["method"] == "joint":
                    # JS-style projection: aligned candidates form G joint actions.
                    # Full sampled responses retain their original behavior density.
                    candidates = {
                        i: p.sample([e.prompt(i)] * c["group_size"])
                        for i in range(len(e.names))
                    }
                    joint_scores = []
                    for k in range(c["group_size"]):
                        e.restore(snapshot)
                        actions = [
                            executed_action(candidates[i][k], e, i)
                            for i in range(len(e.names))
                        ]
                        if c["reward_scope"] == "local":
                            raise ValueError(
                                "joint variant uses global reward; local is paired-only ablation"
                            )
                        joint_scores.append(fork_return(e, snapshot, actions, p, c))
                    # Limit trained target groups identically to paired method.
                    for i in targets:
                        group = label_group(
                            candidates[i],
                            joint_scores,
                            self._id(),
                            c["invalid_penalty"],
                        )
                        for r in group:
                            r.metadata.update(
                                intersection=e.names[i], time=time, method="joint"
                            )
                        rows.extend(group)
                else:
                    for i in targets:
                        e.restore(snapshot)
                        candidates = p.sample([e.prompt(i)] * c["group_size"])
                        values = [[] for _ in candidates]
                        for m in range(c["contexts"]):
                            e.restore(snapshot)
                            background, _ = joint_sample(e, p)
                            for k, response in enumerate(candidates):
                                e.restore(snapshot)
                                if c["method"] == "unpaired":
                                    background, _ = joint_sample(e, p)
                                actions = list(background)
                                actions[i] = executed_action(response, e, i)
                                # Every candidate gets the same # continuations: no hidden phase-cache savings.
                                values[k].append(
                                    fork_return(e, snapshot, actions, p, c, i)
                                )
                        scores = [statistics.mean(v) for v in values]
                        group = label_group(
                            candidates, scores, self._id(), c["invalid_penalty"]
                        )
                        for k, r in enumerate(group):
                            r.metadata.update(
                                intersection=e.names[i],
                                time=time,
                                method=c["method"],
                                context_returns=values[k],
                            )
                        rows.extend(group)
                new_groups = {
                    r.metadata["group"] for r in rows if r.metadata["time"] == time
                }
                for gid in new_groups:
                    subset = [r for r in rows if r.metadata["group"] == gid]
                    diagnostics.append(
                        {
                            "group": gid,
                            "time": time,
                            "unique_phases": len(
                                {r.action for r in subset if r.action is not None}
                            ),
                            "reward_std": statistics.pstdev(r.reward for r in subset),
                            "zero_advantage": all(
                                abs(r.advantage) < 1e-10 for r in subset
                            ),
                            "invalid_fraction": sum(r.action is None for r in subset)
                            / len(subset),
                        }
                    )
            finally:
                e.restore(snapshot)
            e.step(actual_actions)
        return rows, diagnostics

    def _trajectories(self):
        e, p, c = self.env, self.policy, self.c
        snapshot, initial = e.snapshot(), e.time
        count = min(c["collect_steps"], e.remaining_steps)
        targets = target_indices(len(e.names), c["targets_per_state"], self.offset)
        self.offset += len(targets)
        trajectories, rewards, first_end = [], [], None
        try:
            for k in range(c["group_size"]):
                e.restore(snapshot)
                trajectory, rs = [], []
                for t in range(count):
                    actions, response = joint_sample(e, p)
                    trajectory.append(response)
                    rs.append(step_reward(e.step(actions)))
                trajectories.append(trajectory)
                rewards.append(rs)
                if k == 0:
                    first_end = e.snapshot()
        finally:
            e.restore(first_end if first_end is not None else snapshot)
        rows, diagnostics = [], []
        if c["method"] == "trajectory":
            # Uniform trajectory scalar: a controlled baseline, NOT exact Traffic-R1 STPO.
            totals = [sum(rs) / count for rs in rewards]
        else:
            rtgs = [returns_to_go(rs, c["gamma"]) for rs in rewards]
        for t in range(count):
            values = totals if c["method"] == "trajectory" else [rtg[t] for rtg in rtgs]
            for i in targets:
                gid = self._id()
                group_values = [
                    values[j]
                    - c["invalid_penalty"] * (trajectories[j][t][i].action is None)
                    for j in range(c["group_size"])
                ]
                group_advs = standardize(group_values)
                for k in range(c["group_size"]):
                    r = trajectories[k][t][i]
                    r.reward, r.advantage = group_values[k], group_advs[k]
                    r.metadata.update(
                        group=gid,
                        intersection=e.names[i],
                        time=initial + t * e.interval,
                        method=c["method"],
                        trajectory=k,
                    )
                    rows.append(r)
                diagnostics.append(
                    {
                        "group": gid,
                        "time": initial + t * e.interval,
                        "reward_std": statistics.pstdev(group_values),
                        "zero_advantage": all(abs(a) < 1e-10 for a in group_advs),
                        "unique_phases": len(
                            {
                                trajectories[k][t][i].action
                                for k in range(c["group_size"])
                                if trajectories[k][t][i].action is not None
                            }
                        ),
                        "invalid_fraction": sum(
                            trajectories[k][t][i].action is None
                            for k in range(c["group_size"])
                        )
                        / c["group_size"],
                    }
                )
        return rows, diagnostics


def serialize_rows(rows):
    return [asdict(row) for row in rows]
