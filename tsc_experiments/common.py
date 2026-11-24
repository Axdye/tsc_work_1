from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("ETWT", "NTST", "ELWL", "NLSL")
METHODS = ("counterfactual", "joint", "trajectory", "rtg", "colight")
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
    return [(x - mean) / (max(sd, eps)) for x in values]


def returns_to_go(rewards, gamma):
    out, total = [], 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
        out.append(total)
    return list(reversed(out))


def validate_config(c):
    positive = (
        "rounds", "collect_steps", "episode_seconds", "action_interval",
        "train_seconds", "horizon", "group_size", "cluster_count", "cluster_min",
        "calibration_steps", "branch_workers", "generation_batch_size",
        "max_prompt_tokens", "max_new_tokens", "update_epochs", "micro_batch_size",
        "gradient_accumulation_steps", "eval_every", "lora_rank",
    )
    nonnegative = ("seed", "validation_seed", "warmup_steps", "max_responses",
                   "guard_steps", "clearance")
    for key in positive + nonnegative:
        if type(c[key]) is not int or c[key] < int(key in positive):
            raise ValueError(f"Invalid integer configuration: {key}")
    for key in ("gamma", "clip_epsilon", "learning_rate", "kl_beta", "max_grad_norm",
                "invalid_penalty", "cf_weight", "cluster_alpha", "rms_floor",
                "cf_clip", "max_wall_seconds"):
        value = c[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if c["method"] not in METHODS or c["group_size"] < 2:
        raise ValueError("Unknown method or group_size < 2")
    interval = c["action_interval"]
    if not 0 <= c["clearance"] < interval:
        raise ValueError("Require 0 <= clearance < action_interval")
    if c["episode_seconds"] % interval or c["train_seconds"] % interval:
        raise ValueError("Episode and training cutoff must align to control intervals")
    if c["train_seconds"] > c["episode_seconds"]:
        raise ValueError("Training cutoff exceeds episode")
    if c["guard_steps"] < c["horizon"] - 1:
        raise ValueError("guard_steps must cover horizon minus the intervention step")
    if (c["warmup_steps"] + c["collect_steps"] + c["guard_steps"]) * interval > c["train_seconds"]:
        raise ValueError("A full training window must fit before the cutoff")
    if c["calibration_steps"] * interval > c["train_seconds"]:
        raise ValueError("Clustering calibration must end before the training cutoff")
    if not 0 <= c["gamma"] <= 1 or not 0 < c["clip_epsilon"] < 1:
        raise ValueError("Invalid gamma or clip_epsilon")
    if not 0 <= c["cf_weight"] <= 1 or c["cf_baseline"] not in ("loo", "uniform"):
        raise ValueError("Invalid counterfactual weight or baseline")
    if min(c["cluster_alpha"], c["rms_floor"], c["cf_clip"]) <= 0:
        raise ValueError("Cluster smoothing, RMS floor and credit clip must be positive")
    if min(c["max_wall_seconds"], c["invalid_penalty"], c["kl_beta"]) < 0:
        raise ValueError("Time limit, format penalty and KL coefficient must be nonnegative")
    if min(c["learning_rate"], c["max_grad_norm"]) <= 0:
        raise ValueError("Optimizer learning rate and gradient norm must be positive")
    if type(c["local_files_only"]) is not bool:
        raise ValueError("local_files_only must be a boolean")
    if c["temperature"] != 1.0:
        raise ValueError("Training requires temperature=1, top_p=1")
    return c


def configuration(path, overrides=()):
    c = read_json(path)
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
