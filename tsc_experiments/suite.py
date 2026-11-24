"""Reviewable experiment plan + single-GPU-per-job scheduler (8 GPUs = 8 jobs)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, read_json, write_json, validate_config


def make_plan(spec_path, output, model=None):
    spec_path, out = Path(spec_path).resolve(), Path(output).resolve()
    spec = read_json(spec_path)
    base = read_json(spec_path.parent / spec["base_config"])
    if model:
        base["model"] = model
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    variants = dict(spec["variants"])
    if spec.get("wall_match"):
        if "full" not in variants or "trajectory" not in variants:
            raise ValueError("Wall-clock matching needs full and trajectory variants")
        variants["trajectory-wall"] = variants["trajectory"]
    for variant, changes in variants.items():
        for seed in spec["train_seeds"]:
            config = validate_config({**base, **changes, "seed": seed})
            config_path = out / "configs" / f"{variant}-s{seed}.json"
            if config_path.exists() and read_json(config_path) != config:
                raise ValueError(
                    f"Existing plan differs: {config_path}; use a new output directory"
                )
            write_json(config_path, config)
            train_id = f"train-{variant}-s{seed}"
            train_out = out / "train" / f"{variant}-s{seed}"
            wall_reference = (
                str(out / "train" / f"full-s{seed}" / "summary.json")
                if variant == "trajectory-wall" else None
            )
            gpu = config["method"] != "colight"
            jobs.append(
                {
                    "id": train_id,
                    "kind": "train",
                    "gpu": gpu,
                    "depends": [f"train-full-s{seed}"] if wall_reference else [],
                    "match_wall_to": wall_reference,
                    "config": str(config_path),
                    "output": str(train_out),
                }
            )
            for dataset in spec["test_datasets"]:
                for eval_seed in spec["eval_seeds"]:
                    jobs.append(
                        {
                            "id": f"eval-{variant}-s{seed}-{dataset}-e{eval_seed}",
                            "kind": "evaluate",
                            "gpu": gpu,
                            "depends": [train_id],
                            "config": str(config_path),
                            "output": str(
                                out
                                / "evaluation"
                                / f"{variant}-s{seed}-{dataset}-e{eval_seed}"
                            ),
                            "dataset": dataset,
                            "seed": eval_seed,
                            "policy": "llm" if gpu else "colight",
                            "method_label": variant,
                            "train_seed": seed,
                            "train_output": str(train_out),
                        }
                    )
    for dataset in spec["test_datasets"]:
        for seed in spec["eval_seeds"]:
            for policy in spec["baselines"]:
                config_path = out / "configs" / "baseline.json"
                # Baseline evaluation shares EXACTLY the base timing and road/flow data.
                write_json(config_path, validate_config(base))
                jobs.append(
                    {
                        "id": f"eval-{policy}-{dataset}-e{seed}",
                        "kind": "evaluate",
                        "depends": [],
                        "gpu": policy == "llm",
                        "config": str(config_path),
                        "dataset": dataset,
                        "seed": seed,
                        "policy": policy,
                        "method_label": "llm-base" if policy == "llm" else policy,
                        "output": str(
                            out / "evaluation" / f"{policy}-{dataset}-e{seed}"
                        ),
                    }
                )
    plan = {
        "spec": spec,
        "jobs": jobs,
        "note": "One GPU per LLM job; counterfactual workers only use CPU. Final checkpoints only.",
    }
    if (out / "plan.json").exists() and read_json(out / "plan.json") != plan:
        raise ValueError("Existing plan differs; use a new output directory")
    write_json(out / "plan.json", plan)
    return plan


def command(job):
    cmd = [
        sys.executable,
        str(ROOT / "scripts/tsc_experiment.py"),
        job["kind"],
        "--config",
        job["config"],
        "--output",
        job["output"],
    ]
    if job.get("match_wall_to"):
        cmd += ["--match-wall-to", job["match_wall_to"]]
    if job["kind"] == "evaluate":
        cmd += [
            "--dataset",
            job["dataset"],
            "--seed",
            str(job["seed"]),
            "--policy",
            job["policy"],
            "--method-label",
            job["method_label"],
        ]
        if "train_output" in job:
            summary = read_json(Path(job["train_output"]) / "summary.json")
            adapter = summary["final_adapter"]
            if not adapter:
                raise RuntimeError(f"No final checkpoint in {job['train_output']}")
            cmd += ["--adapter", adapter, "--train-seed", str(job["train_seed"])]
    return cmd


def execute(plan, output, gpus):
    out = Path(output)
    status_path = out / "status.json"
    status = read_json(status_path) if status_path.exists() else {}
    jobs = {j["id"]: j for j in plan["jobs"]}
    # A stale process status after interruption needs explicit inspection, not a silent overwrite.
    stale = [name for name, value in status.items() if value.get("state") == "running"]
    if stale:
        raise RuntimeError(
            f"Inspect interrupted jobs {stale}; preserve/rename outputs and remove stale status entries manually"
        )
    available, active, cpu_busy = list(gpus), {}, False
    if not available and any(j["gpu"] for j in jobs.values()):
        raise ValueError("GPU jobs need --gpus, e.g. --gpus 0 1 2 3 4 5 6 7")
    try:
        while True:
            for name, job in jobs.items():
                if name in status:
                    continue
                deps = [status.get(d, {}).get("state") for d in job["depends"]]
                if any(s in ("failed", "blocked") for s in deps):
                    status[name] = {"state": "blocked", "reason": "dependency failed"}
                    continue
                if any(s != "complete" for s in deps):
                    continue
                if job["gpu"] and not available or not job["gpu"] and cpu_busy:
                    continue
                gpu = available.pop(0) if job["gpu"] else None
                if gpu is None:
                    cpu_busy = True
                folder = out / "logs"
                folder.mkdir(exist_ok=True)
                log = (folder / f"{name}.log").open("a")
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu) if gpu is not None else ""
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                try:
                    cmd = command(job)
                    process = subprocess.Popen(
                        cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT
                    )
                except Exception as exc:
                    log.close()
                    status[name] = {"state": "failed", "error": str(exc)}
                    if gpu is None:
                        cpu_busy = False
                    else:
                        available.append(gpu)
                    continue
                active[name] = (process, gpu, log)
                status[name] = {
                    "state": "running",
                    "pid": process.pid,
                    "gpu": gpu,
                    "command": cmd,
                }
                print(
                    f"Started {name} on {gpu if gpu is not None else 'CPU'}", flush=True
                )
            for name, (process, gpu, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                status[name].update(
                    state="complete" if code == 0 else "failed", exit_code=code
                )
                del active[name]
                if gpu is None:
                    cpu_busy = False
                else:
                    available.append(gpu)
                print(f"Finished {name}: {code}", flush=True)
            write_json(status_path, status)
            if not active and len(status) == len(jobs):
                break
            time.sleep(0.25)
    finally:
        for process, _, log in active.values():
            process.terminate()
            log.close()
        write_json(status_path, status)
    failed = [name for name, value in status.items() if value["state"] != "complete"]
    if failed:
        raise RuntimeError(f"Failed/blocked jobs: {failed}; inspect {out / 'logs'}")
    return {"complete": len(status), "output": str(out)}
