"""Shared CoLight-style DQN under the same observation and signal protocol."""

from collections import deque
import math
from pathlib import Path
import random
import time

import torch
from torch import nn

from .branches import step_reward
from .common import PHASES, append_json, write_json
from .environment import cluster_features


def state(env):
    local, neighbors = [], []
    for o in env.observations():
        features = cluster_features(o)
        features[:8] = [math.log1p(v) for v in features[:8]]
        features += [
            math.log1p(o["phase_observations"][phase][key])
            for key in ("receiving_queued", "receiving_vehicles")
            for phase in PHASES
        ]
        local.append(features + [o["time_seconds"] / 3600])
        neighbors.append([-1 if n is None else math.log1p(n)
                          for n in o["neighbor_total_queues_ENWS"]])
    return torch.tensor(local, dtype=torch.float32), torch.tensor(neighbors, dtype=torch.float32)


class CoLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(25, 64), nn.ReLU())
        self.direction = nn.Parameter(torch.zeros(5, 64))
        self.messages = nn.Linear(1, 64)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.q = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, local, neighbors):
        local = self.local(local)
        # Inputs include four direction-specific queues and a self token.
        messages = torch.cat((local.unsqueeze(1), self.messages(neighbors.clamp_min(0).unsqueeze(-1))), 1)
        mask = torch.cat((torch.zeros_like(neighbors[:, :1], dtype=torch.bool), neighbors < 0), 1)
        context, _ = self.attention(local.unsqueeze(1), messages + self.direction,
                                    messages + self.direction, key_padding_mask=mask,
                                    need_weights=False)
        return self.q(torch.cat((local, context.squeeze(1)), 1))

    def actions(self, env):
        with torch.no_grad():
            return self(*state(env)).argmax(1).tolist()

    @classmethod
    def load(cls, checkpoint):
        policy = cls()
        policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        return policy.eval()


def train_colight(config, out, make_env, validate):
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    policy, target = CoLight(), CoLight()
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)
    replay = deque(maxlen=4096)
    steps_per_round = config["collect_steps"] * config["group_size"]
    total_steps = config["rounds"] * steps_per_round
    env, episode = None, -1
    elapsed, sim_seconds, updates = 0.0, 0, 0
    for round_number in range(1, config["rounds"] + 1):
        started = time.perf_counter()
        for step_number in range(steps_per_round):
            if env is None or not env.remaining_steps:
                episode += 1
                env = make_env(episode)
                for _ in range(config["warmup_steps"]):
                    env.step(env.rule_actions())
                    sim_seconds += env.interval
                observation = state(env)
            progress = ((round_number - 1) * steps_per_round + step_number) / total_steps
            epsilon = 1.0 - 0.95 * min(1.0, 2 * progress)
            with torch.no_grad():
                greedy = policy(*observation).argmax(1).tolist()
            actions = [rng.randrange(4) if rng.random() < epsilon else a for a in greedy]
            reward = step_reward(env.step(actions)) / len(env.names)
            sim_seconds += env.interval
            next_observation = state(env)
            replay.append((observation, actions, reward, next_observation, bool(env.remaining_steps)))
            observation = next_observation
            if len(replay) < 32:
                continue
            samples = rng.sample(list(replay), 32)
            local, neighbors = (torch.cat([s[0][i] for s in samples]) for i in range(2))
            next_local, next_neighbors = (torch.cat([s[3][i] for s in samples]) for i in range(2))
            chosen = torch.tensor([a for s in samples for a in s[1]]).unsqueeze(1)
            y = torch.tensor([s[2] for s in samples]).repeat_interleave(len(env.names))
            done_mask = torch.tensor([s[4] for s in samples]).repeat_interleave(len(env.names))
            with torch.no_grad():
                next_action = policy(next_local, next_neighbors).argmax(1, keepdim=True)
                y += config["gamma"] * done_mask * target(next_local, next_neighbors).gather(1, next_action).squeeze(1)
            q = policy(local, neighbors).gather(1, chosen).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5)
            optimizer.step()
            updates += 1
            if updates % 64 == 0:
                target.load_state_dict(policy.state_dict())
        elapsed += time.perf_counter() - started
        metrics = {
            "round": round_number, "wall_seconds": elapsed,
            "sim_seconds": sim_seconds, "optimizer_steps": updates,
            "training_llm_calls": 0, "training_generated_tokens": 0,
        }
        if round_number % config["eval_every"] == 0 or round_number == config["rounds"]:
            metrics.update(validate(policy, round_number, sim_seconds, elapsed))
        append_json(out / "training.jsonl", metrics)
    checkpoint = Path(out) / "checkpoints" / "final" / "colight.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(policy.state_dict(), checkpoint)
    summary = {"stop_reason": "round_limit", "final_adapter": str(checkpoint),
               "sim_seconds": sim_seconds, "wall_seconds": elapsed}
    write_json(out / "summary.json", summary)
    return summary
