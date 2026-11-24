#!/usr/bin/env python3
"""Online CityFlow LoRA/GRPO experiment suite. No jobs start on import."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsc_experiments.common import configuration, read_json, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--config", default=str(ROOT / "configs/credit_smoke.json"))
        s.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
        s.add_argument("--output", required=True)
        if name == "train":
            s.add_argument("--match-wall-to", help="Summary.json for the full run with the same seed")
            s.add_argument(
                "--scripted",
                action="store_true",
                help="CPU plumbing check; never claims LLM training",
            )
        else:
            s.add_argument("--dataset")
            s.add_argument("--seed", type=int)
            s.add_argument(
                "--policy",
                choices=["llm", "fixedtime", "maxpressure", "scripted", "colight"],
                default="llm",
            )
            s.add_argument("--adapter")
            s.add_argument("--method-label")
            s.add_argument("--train-seed", type=int)
    s = sub.add_parser("suite")
    s.add_argument("--spec", default=str(ROOT / "configs/credit_suite.json"))
    s.add_argument("--output", required=True)
    s.add_argument("--model")
    s.add_argument("--gpus", nargs="*", default=[])
    s.add_argument(
        "--execute",
        action="store_true",
        help="Default only writes reviewable plan/configs",
    )
    s = sub.add_parser("report")
    s.add_argument("--root", required=True)
    s.add_argument("--output", required=True)
    s = sub.add_parser("animate")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--labels", nargs="+")
    s.add_argument("--output", required=True)
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--fps", type=int, default=8)
    s.add_argument("--queue-max", type=float)
    s = sub.add_parser("audit")
    s.add_argument("--run", required=True)
    s = sub.add_parser("diagnose")
    s.add_argument("--config", default=str(ROOT / "configs/credit_smoke.json"))
    s.add_argument("--set", action="append", default=[])
    s.add_argument("--output", required=True)
    s.add_argument("--states", type=int, default=12)
    s.add_argument("--scripted", action="store_true")
    s.add_argument("--training-run", help="Full-method training directory with final adapter and clusters")
    args = parser.parse_args()
    if args.command == "train":
        from tsc_experiments.runner import train

        config = configuration(args.config, args.set)
        if args.match_wall_to:
            if config["method"] != "trajectory" or args.scripted:
                parser.error("--match-wall-to is for learned Trajectory-GRPO")
            reference = Path(args.match_wall_to)
            summary = read_json(reference)
            reference_config = read_json(reference.parent / "config.json")
            if (summary.get("scripted") or reference_config["method"] != "counterfactual"
                    or config["seed"] != reference_config["seed"]):
                parser.error("Wall-clock reference must be the trained counterfactual run with the same seed")
            config.update(max_wall_seconds=summary["wall_seconds"], max_responses=0,
                          rounds=2**31, wall_reference=str(reference.resolve()))
            validate_config(config)
        result = train(config, args.output, args.scripted)
    elif args.command == "evaluate":
        from tsc_experiments.runner import evaluate

        metadata = {
            "method": args.method_label or args.policy,
            "train_seed": args.train_seed,
            "split": "test",
        }
        result = evaluate(
            configuration(args.config, args.set),
            args.output,
            args.dataset,
            args.seed,
            args.policy,
            args.adapter,
            run_metadata=metadata,
        )
    elif args.command == "suite":
        from tsc_experiments.suite import make_plan, execute

        plan = make_plan(args.spec, args.output, args.model)
        result = (
            execute(plan, args.output, args.gpus)
            if args.execute
            else {
                "plan": str(Path(args.output).resolve() / "plan.json"),
                "jobs": len(plan["jobs"]),
                "executed": False,
            }
        )
    elif args.command == "report":
        from tsc_experiments.visualization import report

        result = report(args.root, args.output)
    elif args.command == "animate":
        from tsc_experiments.visualization import animate

        result = animate(
            args.runs, args.output, args.labels, args.stride, args.fps, args.queue_max
        )
    elif args.command == "audit":
        from tsc_experiments.visualization import audit_evaluation

        result = audit_evaluation(args.run)
    else:
        from tsc_experiments.diagnostics import diagnose

        result = diagnose(
            configuration(args.config, args.set),
            args.output,
            args.states,
            args.scripted,
            args.training_run,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
