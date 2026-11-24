"""One collector for fixed-horizon counterfactual GRPO and its controls."""

from collections import Counter
from dataclasses import dataclass
import math
import statistics
import time

from .branches import branch_return, score_slice, score_trajectory, step_reward
from .common import PHASES, returns_to_go, standardize
from .environment import cluster_features, prompt


class StateClusters:
    """Frozen Z-score and deterministic K-means over action-independent features."""

    def __init__(self, np, features, count, seed):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != 16 or len(x) < count:
            raise ValueError("Clustering requires at least G traffic states of 16 features")
        self.np = np
        self.mean, scale = x.mean(axis=0), x.std(axis=0)
        self.scale = np.where(scale > 1e-8, scale, 1)
        x = (x - self.mean) / self.scale
        rng = np.random.default_rng(seed)
        chosen = [x[rng.integers(len(x))]]
        distance = ((x - chosen[0]) ** 2).sum(axis=1)
        while len(chosen) < count:
            mass = distance.sum()
            index = rng.choice(len(x), p=distance / mass) if mass else rng.integers(len(x))
            chosen.append(x[index])
            distance = np.minimum(distance, ((x - x[index]) ** 2).sum(axis=1))
        centers = np.asarray(chosen)
        for _ in range(100):
            groups = self.assign(x, centers)
            updated = np.array([
                x[groups == j].mean(axis=0) if (groups == j).any() else centers[j]
                for j in range(count)
            ])
            if np.allclose(updated, centers, rtol=0, atol=1e-8):
                break
            centers = updated
        self.centers = updated

    def assign(self, x, centers):
        return ((x[:, None, :] - centers) ** 2).sum(axis=2).argmin(axis=1)

    def predict(self, features):
        x = (self.np.asarray(features) - self.mean) / self.scale
        return self.assign(x, self.centers).tolist()

    def parameters(self):
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "centers": self.centers.tolist(),
        }


def executed_actions(responses, observations):
    return [
        r.action if r.action is not None else
        PHASES.index(o["current_phase"]) if o["current_phase"] in PHASES else 0
        for r, o in zip(responses, observations)
    ]


def joint_sample(env, policy, greedy=False):
    observations = env.observations()
    responses = policy.sample([prompt(o) for o in observations], greedy=greedy)
    return executed_actions(responses, observations), responses


def loo_probabilities(counts, actual, alpha):
    n = sum(counts.values()) - 1 + 4 * alpha
    return [(counts[b] - (actual == b) + alpha) / n for b in range(4)]


def set_counterfactual_advantages(rows, groups, score_vectors, config):
    """Use only legal responses in the LOO frequencies and the global RMS."""
    valid = [j for j, row in enumerate(rows) if row.action is not None]
    counts = Counter(rows[j].action for j in valid)
    grouped = {}
    for j in valid:
        grouped.setdefault(groups[j], Counter())[rows[j].action] += 1
    raw, uniform, loo, fallback = [], [], [], []
    for j in valid:
        action, g, scores = rows[j].action, groups[j], score_vectors[j]
        local = grouped[g]
        use_global = sum(local.values()) < config["cluster_min"]
        weights = loo_probabilities(counts if use_global else local, action, config["cluster_alpha"])
        loo_credit = sum(w * (scores[action] - q) for w, q in zip(weights, scores))
        uniform_credit = scores[action] - statistics.mean(scores)
        credit = loo_credit if config["cf_baseline"] == "loo" else uniform_credit
        raw.append(credit)
        uniform.append(uniform_credit)
        loo.append(loo_credit)
        fallback.append(use_global)
        rows[j].metadata.update(
            cluster=g,
            cluster_fallback=use_global,
            loo_probabilities=weights,
            candidate_returns=scores,
            raw_cf_credit=credit,
        )
    rms = math.sqrt(statistics.mean(q * q for q in raw)) if raw else 0.0
    scale = max(rms, config["rms_floor"])
    clip, mix = config["cf_clip"], config["cf_weight"]
    for j, q in zip(valid, raw):
        local = max(-clip, min(clip, q / scale))
        rows[j].reward = q
        rows[j].metadata["cf_advantage"] = local
        rows[j].advantage = (1 - mix) * rows[j].advantage + mix * local
    n = len(valid)
    return {
        "cf_raw_mean": statistics.mean(raw) if n else 0.0,
        "cf_uniform_mean": statistics.mean(uniform) if n else 0.0,
        "cf_loo_mean": statistics.mean(loo) if n else 0.0,
        "cf_rms": rms,
        "cf_clip_fraction": sum(abs(q / scale) > clip for q in raw) / n if n else 0.0,
        "cluster_fallback_rate": sum(fallback) / n if n else 0.0,
        "cluster_sharpness": statistics.mean(
            (max(g.values()) + config["cluster_alpha"]) /
            (sum(g.values()) + 4 * config["cluster_alpha"]) for g in grouped.values()
        ) if grouped else 0.0,
        "cluster_sizes": {g: sum(c.values()) for g, c in sorted(grouped.items())},
        "cf_valid_count": n,
    }


@dataclass
class Decision:
    time: int
    actions: list
    queue_costs: list
    responses: list
    features: list
    scores: list


class Collector:
    def __init__(self, env, policy, config, clusters=None, pool=None):
        self.env, self.policy, self.c = env, policy, config
        self.clusters, self.pool = clusters, pool
        self.prefix = []
        self.generation_seconds = 0.0

    @property
    def use_cf(self):
        return self.c["method"] == "counterfactual" and self.c["cf_weight"] > 0

    @property
    def ready(self):
        return self.env.remaining_steps >= self.c["collect_steps"] + self.c["guard_steps"]

    @property
    def response_cost(self):
        return self.c["group_size"] * self.c["collect_steps"] * len(self.env.names)

    def warmup(self):
        for _ in range(self.c["warmup_steps"]):
            actions = self.env.rule_actions()
            self.env.step(actions)
            self.prefix.append(actions)

    def sample(self, observations, count=1):
        started = time.perf_counter()
        rows = self.policy.sample([prompt(o) for o in observations] * count)
        self.generation_seconds += time.perf_counter() - started
        return rows

    def collect(self):
        if not self.ready:
            raise ValueError("Full collection window and guard must fit before training cutoff")
        self.generation_seconds = 0.0
        if self.c["method"] == "joint":
            return self.collect_joint()
        return self.collect_trajectories()

    def collect_trajectories(self):
        env, c = self.env, self.c
        start = env.snapshot()
        trajectories = []
        for k in range(c["group_size"]):
            env.restore(start)
            sequence = []
            for _ in range(c["collect_steps"]):
                observations = env.observations()
                responses = self.sample(observations)
                actions = executed_actions(responses, observations)
                now = env.time
                scores = []
                if self.use_cf and self.pool is None:
                    scores, check = score_slice(env, actions, c["horizon"], c["gamma"])
                costs = env.step(actions)["queue_costs"]
                if scores and check != -sum(costs):
                    raise RuntimeError("Snapshot replay changed the actual first-step reward")
                sequence.append(Decision(
                    now, actions, costs, responses,
                    [cluster_features(o) for o in observations] if self.use_cf else [],
                    scores,
                ))
            if k == 0:
                next_state = env.snapshot()
            trajectories.append(sequence)
        env.restore(next_state)

        replay_seconds, worker_cpu = 0, 0.0
        if self.use_cf and self.pool is not None:
            arguments = (env.dataset, env.seed, env.interval, c["clearance"], env.horizon)
            jobs = [
                self.pool.submit(
                    score_trajectory, arguments, str(env.workdir / "branches"),
                    self.prefix, [(d.actions, d.queue_costs) for d in trajectory],
                    c["horizon"], c["gamma"],
                )
                for trajectory in trajectories
            ]
            for trajectory, job in zip(trajectories, jobs):
                scores, replay, cpu_seconds = job.result()
                replay_seconds += replay
                worker_cpu += cpu_seconds
                for d, values in zip(trajectory, scores):
                    d.scores = values
        self.prefix.extend(d.actions for d in trajectories[0])
        rewards = [[-sum(d.queue_costs) for d in trajectory] for trajectory in trajectories]
        length = c["collect_steps"]
        if c["method"] == "rtg":
            signals = [returns_to_go(p, 1.0) for p in rewards]
            advantages = [standardize(v) for v in zip(*signals)]
        else:
            values = [sum(p) / length for p in rewards]
            signals = [[v] * length for v in values]
            advantages = [standardize(values)] * length
        rows, vectors, features = [], [], []
        for k, trajectory in enumerate(trajectories):
            for t, d in enumerate(trajectory):
                for i, row in enumerate(d.responses):
                    row.advantage = advantages[t][k] if row.action is not None else -c["invalid_penalty"]
                    row.reward = signals[k][t] if row.action is not None else -c["invalid_penalty"]
                    row.metadata.update(
                        trajectory=k, step=t, time=d.time, intersection=env.names[i],
                        executed_phase=d.actions[i], trajectory_return=sum(rewards[k]),
                        team_advantage=advantages[t][k],
                    )
                rows.extend(d.responses)
                vectors.extend(d.scores)
                features.extend(d.features)
        diagnostics = {}
        if self.use_cf:
            diagnostics = set_counterfactual_advantages(
                rows, self.clusters.predict(features), vectors, c
            )
            diagnostics["snapshot_replay_checked"] = c["group_size"] * length
        branch_seconds = (
            c["group_size"] * length * (1 + 3 * len(env.names)) * c["horizon"] * env.interval
            if self.use_cf else 0
        )
        return rows, {
            **diagnostics,
            "main_trajectory_sim_seconds": c["group_size"] * length * env.interval,
            "branch_sim_seconds": branch_seconds,
            "replay_sim_seconds": replay_seconds,
            "branch_worker_cpu_seconds": worker_cpu,
            "generation_wall_seconds": self.generation_seconds,
        }

    def collect_joint(self):
        env, c = self.env, self.c
        rows = []
        n = len(env.names)
        for t in range(c["collect_steps"]):
            observations = env.observations()
            sampled = self.sample(observations, c["group_size"])
            candidates = [sampled[i:i+n] for i in range(0, len(sampled), n)]
            actions = [executed_actions(group, observations) for group in candidates]
            snapshot, now = env.snapshot(), env.time
            values = []
            try:
                for joint in actions:
                    env.restore(snapshot)
                    score, _ = branch_return(env, joint, c["horizon"], c["gamma"])
                    values.append(score)
            finally:
                env.restore(snapshot)
            # A fixed candidate (index zero) advances the visited trajectory.
            env.step(actions[0])
            for i in range(n):
                buckets = {}
                for k, group in enumerate(candidates):
                    action = group[i].action
                    if action is not None:
                        buckets.setdefault(action, []).append(values[k])
                action_scores = {a: statistics.mean(v) for a, v in buckets.items()}
                valid = [group[i] for group in candidates if group[i].action is not None]
                group_values = [action_scores[r.action] for r in valid]
                for row, adv in zip(valid, standardize(group_values) if len(valid) > 1 else [0.0]):
                    row.reward, row.advantage = action_scores[row.action], adv
                for k, group in enumerate(candidates):
                    row = group[i]
                    if row.action is None:
                        row.reward = row.advantage = -c["invalid_penalty"]
                    row.metadata.update(
                        candidate=k, step=t, time=now, intersection=env.names[i],
                        executed_phase=actions[0][i], candidate_returns=action_scores,
                    )
            rows.extend(sampled)
        return rows, {
            "main_trajectory_sim_seconds": c["collect_steps"] * env.interval,
            "branch_sim_seconds": c["collect_steps"] * c["group_size"] * c["horizon"] * env.interval,
            "replay_sim_seconds": 0,
            "branch_worker_cpu_seconds": 0.0,
            "generation_wall_seconds": self.generation_seconds,
        }
