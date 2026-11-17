from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("ETWT", "NTST", "ELWL", "NLSL")
METHODS = ("paired", "unpaired", "joint", "trajectory", "rtg")
SYSTEM = (
    "You control one traffic intersection. Choose one phase: ETWT (east/west "
    "through), NTST (north/south through), ELWL (east/west left), NLSL "
    "(north/south left). Right turns follow the road network. Consider incoming "
    "demand, downstream receiving traffic and neighboring demand. Give at most "
    "one short sentence, then exactly one <signal>PHASE</signal> tag."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def empty_dir(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Output must be empty: {path}")
    return path


def standardize(values, eps=1e-6):
    if len(values) < 2 or not all(math.isfinite(x) for x in values):
        raise ValueError("A group needs >=2 finite scores")
    mean = statistics.mean(values)
    # Population std, explicitly fixed for all variants.
    sd = statistics.pstdev(values)
    return [(x - mean) / (sd + eps) for x in values]


def returns_to_go(rewards, gamma):
    out, total = [], 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
        out.append(total)
    return list(reversed(out))


def target_indices(count, size, offset):
    return [(offset + j) % count for j in range(min(count, size))]


def validate_config(c):
    positive = (
        "rounds",
        "collect_steps",
        "episode_seconds",
        "action_interval",
        "horizon",
        "group_size",
        "targets_per_state",
        "contexts",
        "generation_batch_size",
        "max_prompt_tokens",
        "max_new_tokens",
        "update_epochs",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "save_every",
        "eval_every",
        "lora_rank",
    )
    for key in positive:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "seed",
        "validation_seed",
        "warmup_steps",
        "max_sim_seconds",
        "clearance",
    ):
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in (
        "gamma",
        "clip_epsilon",
        "learning_rate",
        "kl_beta",
        "max_grad_norm",
        "invalid_penalty",
        "temperature",
    ):
        value = c[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be finite")
    if c["group_size"] < 2 or c["method"] not in METHODS:
        raise ValueError("Unknown method or group_size < 2")
    if not 0 <= c["clearance"] < c["action_interval"]:
        raise ValueError("Require 0 <= clearance < action_interval")
    if c["episode_seconds"] % c["action_interval"]:
        raise ValueError("episode_seconds must be divisible by action_interval")
    if not 0 <= c["gamma"] <= 1 or not 0 < c["clip_epsilon"] < 1:
        raise ValueError("Invalid gamma/clip_epsilon")
    if c["reward_scope"] not in ("global", "local") or c["continuation"] not in (
        "llm",
        "maxpressure",
    ):
        raise ValueError("Invalid scope/continuation")
    if c["method"] in ("joint", "trajectory", "rtg"):
        if c["reward_scope"] != "global" or c["contexts"] != 1:
            raise ValueError(
                "local reward/multiple contexts are supported only by paired/unpaired"
            )
    if c["method"] in ("trajectory", "rtg") and c["continuation"] != "llm":
        raise ValueError(
            "trajectory/rtg always collect LLM trajectories; rule continuation is unsupported"
        )
    if c["invalid_penalty"] < 0:
        raise ValueError("invalid_penalty must be nonnegative")
    if type(c.get("local_files_only", False)) is not bool:
        raise ValueError("local_files_only must be a boolean")
    if c["max_sim_seconds"] < 0 or c["warmup_steps"] < 0:
        raise ValueError("Invalid budget/warmup")
    if c["warmup_steps"] * c["action_interval"] >= c["episode_seconds"]:
        raise ValueError("Warmup must end before episode horizon")
    if c["learning_rate"] <= 0 or c["kl_beta"] < 0 or c["max_grad_norm"] <= 0:
        raise ValueError("Invalid optimizer parameters")
    # Old log-probabilities are measured under unwarped model probabilities.
    # Supporting temperature/top-p would require matching the behavior density.
    if c.get("temperature", 1.0) != 1.0:
        raise ValueError("Training currently requires temperature=1, top_p=1")
    return c


def configuration(path, overrides=()):
    c = read_json(path)
    c.setdefault("local_files_only", False)
    for item in overrides:
        key, raw = item.split("=", 1)
        if key not in c:
            raise ValueError(f"Unknown configuration key: {key}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        c[key] = value
    return validate_config(c)


def rng(seed):
    return random.Random(int(seed))
