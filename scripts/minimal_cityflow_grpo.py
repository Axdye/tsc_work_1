#!/usr/bin/env python3
"""Small CityFlow -> continuous phase returns -> critic-free GRPO baseline.

No installation or job launch happens on import. This is a fixed-state training
prototype, not a reproduction of DGLight or a multi-step online RL algorithm.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PHASE_IDS = {"ETWT": 1, "NTST": 2, "ELWL": 3, "NLSL": 4}
SYSTEM = (
    "You control one traffic intersection. Choose exactly one legal phase: "
    "ETWT (east/west through), NTST (north/south through), ELWL (east/west left), "
    "NLSL (north/south left). Right turns follow the simulator road network. "
    "Consider incoming demand and receiving-lane capacity. Give at most one short "
    "sentence, then exactly one <signal>PHASE</signal> tag. Do not output scores."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(str(message.get("content", "")) for message in completion)
    raise TypeError(f"Unsupported completion type: {type(completion).__name__}")


def parse_phase(text):
    matches = re.findall(r"<signal>(.*?)</signal>", text, flags=re.I | re.S)
    if len(matches) != 1:
        return None
    phase = matches[0].strip().upper()
    return phase if phase in PHASE_IDS else None


def validate_scores(values):
    if set(values) != set(PHASE_IDS):
        raise ValueError("Each row must contain all four phase_scores")
    if not all(math.isfinite(float(v)) for v in values.values()):
        raise ValueError("Nonfinite phase score")


def traffic_reward(completions, phase_scores, **kwargs):
    """TRL reward callback; metadata contains simulator scores, never prompt labels."""
    if len(completions) != len(phase_scores):
        raise ValueError("Completion/score lengths differ")
    rewards = []
    for completion, values in zip(completions, phase_scores):
        validate_scores(values)
        phase = parse_phase(completion_text(completion))
        # Valid returns are negative queue costs. Zero would reward invalid text.
        rewards.append(float(values[phase]) if phase else min(values.values()) - 1.0)
    return rewards


class TrafficEnv:
    """Direct engine wrapper for the included straight-road, four-phase benchmarks.

    Every API decision is at a control boundary. The only Python control state
    needed for a fork is `current`; clearance is completed inside each interval.
    Phase 0 is the benchmark clearance phase, not a newly modeled yellow light.
    """

    def __init__(self, args, workdir):
        if args.engine_module_dir:
            sys.path.insert(0, str(Path(args.engine_module_dir).resolve()))
        import cityflow
        from cityflow_diagnostic import phase_movements, scores

        self.pressure_scores = scores
        self.roadnet = Path(args.roadnet).resolve()
        self.flow = Path(args.flow).resolve()
        self.interval = args.action_interval
        self.clearance = args.clearance
        if self.interval <= self.clearance or self.clearance < 0:
            raise ValueError("Require 0 <= clearance < action_interval")
        roadnet = read_json(self.roadnet)
        self.nodes = {i["id"]: i for i in roadnet["intersections"] if not i["virtual"]}
        roads = {r["id"]: r for r in roadnet["roads"]}
        expected = {1: {("E", "go_straight"), ("W", "go_straight")},
                    2: {("N", "go_straight"), ("S", "go_straight")},
                    3: {("E", "turn_left"), ("W", "turn_left")},
                    4: {("N", "turn_left"), ("S", "turn_left")}}
        for inter in self.nodes.values():
            for phase, movements in expected.items():
                actual = set()
                for index in inter["trafficLight"]["lightphases"][phase]["availableRoadLinks"]:
                    link = inter["roadLinks"][index]
                    if link["type"] == "turn_right":
                        continue
                    points = roads[link["startRoad"]]["points"]
                    dx = points[-1]["x"] - points[0]["x"]
                    dy = points[-1]["y"] - points[0]["y"]
                    origin = ("W" if dx > 0 else "E") if abs(dx) > abs(dy) else ("S" if dy > 0 else "N")
                    actual.add((origin, link["type"]))
                if actual != movements:
                    raise ValueError(f"Unsupported phase mapping at {inter['id']}, phase {phase}: {actual}")
        self.movements = {name: phase_movements(node) for name, node in self.nodes.items()}
        self.incoming = sorted({lane for m in self.movements.values()
                                for pairs in m.values() for lane, _ in pairs})
        config = {"interval": 1, "seed": args.seed, "dir": "",
                  "roadnetFile": str(self.roadnet), "flowFile": str(self.flow),
                  "rlTrafficLight": True, "laneChange": False, "saveReplay": False}
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        config_path = workdir / "engine.json"
        write_json(config_path, config)
        self.engine = cityflow.Engine(str(config_path), thread_num=1)
        self.current = {name: 1 for name in self.nodes}
        for name in self.nodes:
            self.engine.set_tl_phase(name, 1)

    def queues(self):
        return self.engine.get_lane_waiting_vehicle_count()

    def queue_total(self):
        queues = self.queues()
        return sum(queues.get(lane, 0) for lane in self.incoming)

    def baseline_actions(self):
        queues = self.queues()
        actions = {}
        for name, movements in self.movements.items():
            values = self.pressure_scores(movements, queues)
            actions[name] = max(values, key=values.get)
        return actions

    def advance(self, actions, seconds=None):
        seconds = self.interval if seconds is None else seconds
        changed = {name for name in self.nodes if actions[name] != self.current[name]}
        integrated_queue = 0.0
        for second in range(seconds):
            for name, phase in actions.items():
                self.engine.set_tl_phase(name, 0 if name in changed and second < self.clearance else phase)
            self.engine.next_step()
            integrated_queue += self.queue_total()
        self.current = dict(actions)
        return integrated_queue

    def snapshot(self):
        return self.engine.snapshot(), dict(self.current)

    def restore(self, state):
        archive, current = state
        self.engine.load(archive)
        self.current = dict(current)

    def phase_scores(self, target, horizon_steps):
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        state = self.snapshot()
        first_actions = self.baseline_actions()
        results = {}
        try:
            for label, phase in PHASE_IDS.items():
                self.restore(state)
                actions = dict(first_actions)
                actions[target] = phase
                cost = 0.0
                for step in range(horizon_steps):
                    cost += self.advance(actions)
                    if step + 1 < horizon_steps:
                        actions = self.baseline_actions()
                results[label] = -cost / (horizon_steps * self.interval)
        finally:
            self.restore(state)
        validate_scores(results)
        return results

    def prompt(self, target):
        queues = self.queues()
        counts = self.engine.get_lane_vehicle_count()
        phase_state = {}
        for label, phase in PHASE_IDS.items():
            pairs = self.movements[target][phase]
            inputs = {lane for lane, _ in pairs}
            outputs = {lane for _, lanes in pairs for lane in lanes}
            phase_state[label] = {
                "incoming_queued": sum(queues.get(lane, 0) for lane in inputs),
                "incoming_running": sum(max(0, counts.get(lane, 0) - queues.get(lane, 0)) for lane in inputs),
                "receiving_queued": sum(queues.get(lane, 0) for lane in outputs),
                "receiving_vehicles": sum(counts.get(lane, 0) for lane in outputs),
            }
        observation = {"intersection": target,
                       "time_seconds": self.engine.get_current_time(),
                       "current_phase": next(k for k, v in PHASE_IDS.items() if v == self.current[target]),
                       "decision_interval_seconds": self.interval,
                       "clearance_seconds_on_change": self.clearance,
                       "phase_observations": phase_state}
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(observation, ensure_ascii=False)}]


def collect(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "train.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite {output / 'train.jsonl'}")
    if args.samples < 1 or args.horizon_steps < 1 or args.warmup_steps < 0:
        raise ValueError("Invalid collection size or horizon")
    env = TrafficEnv(args, output / "engine")
    rng = random.Random(args.seed)
    for _ in range(args.warmup_steps):
        env.advance(env.baseline_actions())
    targets = list(env.nodes)
    rows = []
    for index in range(args.samples):
        target = targets[index % len(targets)]
        # Build the prompt before branching; no future scores are actor inputs.
        prompt = env.prompt(target)
        values = env.phase_scores(target, args.horizon_steps)
        if index == 0:
            repeated = env.phase_scores(target, args.horizon_steps)
            if any(abs(values[k] - repeated[k]) > 1e-6 for k in PHASE_IDS):
                raise RuntimeError("Repeated snapshot forks disagree; inspect engine restoration before training")
        rows.append({"sample_id": f"s{args.seed}-{index}", "prompt": prompt,
                     "phase_scores": values, "intersection": target,
                     "time_seconds": env.engine.get_current_time()})
        actions = env.baseline_actions()
        # Exploratory collection independent of the phase-score argmax.
        actions[target] = rng.choice(list(PHASE_IDS.values()))
        env.advance(actions)
        print(json.dumps({"sample": index + 1, "target": target, "scores": values}), flush=True)
    with (output / "train.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = {"kind": "fixed-state simulator-scored GRPO prototype",
                "args": vars(args), "roadnet_sha256": digest(env.roadnet),
                "flow_sha256": digest(env.flow), "phase_ids": PHASE_IDS,
                "reward": "negative global incoming queue integral / horizon seconds",
                "continuation": "fixed queue-pressure rule, not the trained LLM",
                "stochasticity": "one simulator seed; paired engine snapshots, not an exact individual causal effect",
                "clearance": "benchmark phase 0 within the fixed decision interval on a phase change",
                "score_spread_nonzero_rows": sum(max(r["phase_scores"].values()) - min(r["phase_scores"].values()) > 1e-6 for r in rows)}
    write_json(output / "manifest.json", manifest)
    print(f"Saved {len(rows)} samples to {output / 'train.jsonl'}", flush=True)


def load_policy(model_path, adapter_path=None, device=None, local_files_only=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", local_files_only=local_files_only)
    if not tokenizer.chat_template:
        raise ValueError("Choose an instruction model with a chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
    load_kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa",
                   "output_loading_info": True, "local_files_only": local_files_only}
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        from transformers import Qwen3_5ForCausalLM
        # The official checkpoint has model.language_model.* and model.visual.*.
        # Instantiate only the text backbone; rename its weights explicitly.
        if config.model_type == "qwen3_5":
            load_kwargs["key_mapping"] = {r"^model\.language_model\.": "model."}
            config = config.text_config
        model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
            model_path, config=config, **load_kwargs)
    else:
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, **load_kwargs)
    # Fail on missing language weights instead of training random initialization.
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
        raise RuntimeError(f"Incomplete policy checkpoint loading: {loading_info}")
    unexpected = [key for key in loading_info.get("unexpected_keys", [])
                  if not key.startswith(("model.visual.", "mtp.", "model.mtp."))]
    if unexpected:
        raise RuntimeError(f"Unexpected non-vision checkpoint weights: {unexpected}")
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=False, local_files_only=local_files_only)
    if device is not None:
        model.to(device)
    return model, tokenizer


def render_prompt(tokenizer, messages):
    # Keep the same non-thinking prefix for length validation and deployment.
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def train(args):
    import torch
    from accelerate import PartialState
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import GRPOConfig, GRPOTrainer

    config = read_json(args.config)
    model_path = args.model or config["model"]
    output = Path(args.output).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("This baseline targets a BF16-capable CUDA GPU, e.g. A100")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16")
    state = PartialState()
    # Only rank 0 checks before Trainer starts writing, then shares the verdict.
    errors = [None]
    if state.is_main_process and output.exists() and any(output.iterdir()):
        errors[0] = f"Choose an empty training output directory: {output}"
    if world_size > 1:
        torch.distributed.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise FileExistsError(errors[0])
    generation_batch = config["per_device_train_batch_size"] * world_size * config["gradient_accumulation_steps"]
    if config["num_generations"] < 2 or generation_batch % config["num_generations"]:
        raise ValueError("Effective generation batch must be divisible by num_generations >= 2")
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < generation_batch // config["num_generations"]:
        raise ValueError("Not enough distinct prompts for one generation batch")
    for row in rows:
        validate_scores(row["phase_scores"])
    if not any(max(r["phase_scores"].values()) - min(r["phase_scores"].values()) > 1e-6 for r in rows):
        raise ValueError("All phase returns are equal; collect informative traffic states first")
    model, tokenizer = load_policy(model_path)
    # Fail rather than silently truncating traffic state or the output instructions.
    for row in rows:
        length = len(tokenizer(render_prompt(tokenizer, row["prompt"]), add_special_tokens=False)["input_ids"])
        if length > config["max_prompt_length"]:
            raise ValueError(f"Prompt {row['sample_id']} has {length} tokens; increase max_prompt_length")
    model = get_peft_model(model, LoraConfig(
        r=config["lora_rank"], lora_alpha=2 * config["lora_rank"], lora_dropout=0.0,
        target_modules=config["lora_target_modules"],
        task_type="CAUSAL_LM", bias="none"))
    model.config.use_cache = False
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
    max_steps = int(config["max_steps"] if args.max_steps is None else args.max_steps)
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    training_args = GRPOConfig(
        output_dir=str(output), max_steps=max_steps,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_generations=config["num_generations"],
        # TRL 0.29 removed max_prompt_length; enforce our limit above.
        chat_template_kwargs={"enable_thinking": False},
        max_completion_length=config["max_completion_length"],
        temperature=1.0, top_p=1.0, top_k=0,
        beta=config["kl_beta"], num_iterations=1, loss_type="grpo",
        scale_rewards="group", use_vllm=False,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, remove_unused_columns=False,
        logging_steps=1, save_strategy="no", report_to="none",
        seed=config["seed"], data_seed=config["seed"],
    )
    trainer = GRPOTrainer(model=model, processing_class=tokenizer,
                          args=training_args, reward_funcs=traffic_reward,
                          train_dataset=Dataset.from_list(rows))
    result = trainer.train()
    largest_delta = max((value.detach().cpu() - initial[name]).abs().max().item()
                        for name, value in model.named_parameters() if name in initial)
    if not math.isfinite(largest_delta) or largest_delta <= 0:
        raise RuntimeError("No finite nonzero LoRA update: inspect invalid outputs and zero-variance reward groups")
    trainer.save_model(str(output / "adapter"))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / "adapter")
        packages = {name: importlib.metadata.version(name)
                    for name in ["torch", "transformers", "trl", "peft", "accelerate", "datasets"]}
        write_json(output / "run_manifest.json", {
            "kind": "LoRA + GRPO fixed-state baseline; not online traffic RL",
            "base_model": model_path, "data_sha256": digest(args.data),
            "policy_class": type(model.get_base_model()).__name__, "enable_thinking": False,
            "config": config, "max_steps": max_steps, "world_size": world_size,
            "packages": packages, "metrics": result.metrics,
            "max_abs_lora_update": largest_delta,
            "trainable_parameters": sum(v.numel() for v in model.parameters() if v.requires_grad),
            "evaluation": "not performed by the training command; use evaluate separately"})


def evaluate(args):
    if args.run_seconds < 1:
        raise ValueError("run_seconds must be positive")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty evaluation output directory: {output}")
    env = TrafficEnv(args, output / "engine")
    model, tokenizer = None, None
    if args.policy == "llm":
        import torch
        if not args.model or not torch.cuda.is_available():
            raise ValueError("LLM evaluation requires --model and CUDA")
        model, tokenizer = load_policy(args.model, args.adapter, "cuda:0")
        model.eval()
    total_queue, invalid, decisions, initial_time = 0.0, 0, 0, env.engine.get_current_time()
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        while env.engine.get_current_time() - initial_time < args.run_seconds:
            time_now = env.engine.get_current_time()
            if args.policy == "fixedtime":
                actions = {name: 1 + int(time_now // env.interval) % 4 for name in env.nodes}
                texts = []
            elif args.policy == "pressure":
                actions, texts = env.baseline_actions(), []
            else:
                names = list(env.nodes)
                prompts = [render_prompt(tokenizer, env.prompt(name)) for name in names]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                                    add_special_tokens=False).to("cuda:0")
                with torch.inference_mode():
                    generated = model.generate(**encoded, do_sample=False,
                                               max_new_tokens=args.max_new_tokens,
                                               pad_token_id=tokenizer.pad_token_id)
                texts = tokenizer.batch_decode(generated[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
                actions = {}
                for name, text in zip(names, texts):
                    phase = parse_phase(text)
                    invalid += int(phase is None)
                    # Fixed deployment fallback; log it rather than hiding invalidity.
                    actions[name] = PHASE_IDS[phase] if phase else env.current[name]
            decisions += len(env.nodes)
            seconds = min(env.interval, int(args.run_seconds - (time_now - initial_time)))
            total_queue += env.advance(actions, seconds)
            stream.write(json.dumps({"time": time_now, "actions": actions,
                                     "responses": texts}, ensure_ascii=False) + "\n")
    write_json(output / "metrics.json", {
        "policy": args.policy, "model": args.model, "adapter": args.adapter,
        "enable_thinking": False if args.policy == "llm" else None,
        "seed": args.seed, "run_seconds": args.run_seconds,
        "network_mean_queued_vehicles": total_queue / args.run_seconds,
        "engine_att_at_horizon_seconds": env.engine.get_average_travel_time(),
        "remaining_vehicles_including_waiting": len(env.engine.get_vehicles(include_waiting=True)),
        "invalid_response_rate": invalid / decisions if decisions else 0,
        "roadnet_sha256": digest(env.roadnet), "flow_sha256": digest(env.flow),
        "metric_note": "Finite-horizon native Engine ATT; not completed-trip-only ATT or a paper reproduction. Pressure is the local diagnostic variant, not an audited MaxPressure baseline."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["collect", "evaluate"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--roadnet", default=str(ROOT / "vendor/CoLLMLight/data/Synthetic/4_4/roadnet_4_4.json"))
        sub.add_argument("--flow", default=str(ROOT / "vendor/CoLLMLight/data/Synthetic/4_4/anon_4_4_synthetic_8000.json"))
        sub.add_argument("--engine-module-dir", help="Optional directory containing an ABI-compatible CityFlow extension")
        sub.add_argument("--action-interval", type=int, default=30)
        sub.add_argument("--clearance", type=int, default=5)
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--output", required=True)
        if name == "collect":
            sub.add_argument("--samples", type=int, default=32)
            sub.add_argument("--warmup-steps", type=int, default=6)
            sub.add_argument("--horizon-steps", type=int, default=3)
        else:
            sub.add_argument("--policy", choices=["llm", "fixedtime", "pressure"], default="llm")
            sub.add_argument("--model")
            sub.add_argument("--adapter")
            sub.add_argument("--run-seconds", type=int, default=3600)
            sub.add_argument("--max-new-tokens", type=int, default=192)
    sub = subparsers.add_parser("train")
    sub.add_argument("--config", default=str(ROOT / "configs/grpo_smoke.json"))
    sub.add_argument("--model", help="Local model directory or Hugging Face ID; overrides config")
    sub.add_argument("--data", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    {"collect": collect, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
