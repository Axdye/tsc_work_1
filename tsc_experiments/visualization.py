"""Headless figures, raw CSV summaries and road-network queue animations."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from .common import read_json, write_json


def jsonlines(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def audit_evaluation(folder):
    folder = Path(folder)
    metrics = read_json(folder / "metrics.json")
    rows, arrivals = (
        jsonlines(folder / "decisions.jsonl"),
        read_json(folder / "arrivals.json"),
    )
    totals = defaultdict(float)
    horizon = metrics["end_time"]
    for intersection in arrivals.values():
        for vehicle, r in intersection.items():
            if "shadow" in vehicle:
                continue
            totals[vehicle] += (
                r["leave_time"] if r["leave_time"] is not None else horizon
            ) - r["enter_time"]
    actual = {
        "ATT": statistics.mean(totals.values()) if totals else 0.0,
        "AWT": statistics.mean(r["AWT_sample"] for r in rows),
        "AQL": statistics.mean(r["AQL_sample"] for r in rows),
    }
    errors = {key: actual[key] - metrics[key] for key in actual}
    if any(abs(e) > 1e-8 for e in errors.values()):
        raise AssertionError(f"Metric audit failed: {errors}")
    return {"passed": True, "independent_metrics": actual, "errors": errors}


def plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    return plt


def report(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(root.rglob("metrics.json")):
        row = read_json(path)
        if (
            not all(k in row for k in ("ATT", "AWT", "AQL"))
            or row.get("split") == "validation"
        ):
            continue
        if row.get("policy") == "scripted":
            continue  # Infrastructure demo is never included in scientific summary tables.
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        provenance = read_json(manifest_path)
        row.update(
            protocol=provenance.get("protocol"),
            interval=provenance.get("interval"),
            clearance=provenance.get("clearance_within_interval"),
            roadnet_sha256=provenance.get("roadnet_sha256"),
            flow_sha256=provenance.get("flow_sha256"),
            horizon=provenance.get("horizon"),
            laneChange=provenance.get("laneChange"),
            source=str(path),
        )
        row["method"] = row.get("method", row["policy"])
        results.append(row)
    if not results:
        raise ValueError(
            "No evaluation results found (scripted and validation runs are excluded)"
        )
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["dataset"], row["method"])].append(row)
    # Refuse incomparable traffic protocols even if someone manually mixed folders.
    for dataset in {r["dataset"] for r in results}:
        keys = (
            "protocol",
            "interval",
            "clearance",
            "roadnet_sha256",
            "flow_sha256",
            "horizon",
            "laneChange",
        )
        protocols = {
            tuple(r.get(k) for k in keys) for r in results if r["dataset"] == dataset
        }
        if len(protocols) != 1 or any(None in p for p in protocols):
            raise ValueError(
                f"Mixed or missing traffic protocol for {dataset}; report separately"
            )
    summaries = []
    for (dataset, method), group in sorted(grouped.items()):
        # Aggregate environment seeds within each independently trained policy first.
        units = defaultdict(list)
        seen = set()
        for r in group:
            identity = (r.get("train_seed"), r["seed"])
            if identity in seen:
                raise ValueError(
                    f"Duplicate run for {dataset}/{method}/{identity}; select one checkpoint"
                )
            seen.add(identity)
            unit = r.get("train_seed") if r.get("train_seed") is not None else r["seed"]
            units[unit].append(r)
        summary = {
            "dataset": dataset,
            "method": method,
            "raw_runs": len(group),
            "independent_units": len(units),
            "unit": "training_seed"
            if any(r.get("train_seed") is not None for r in group)
            else "environment_seed",
        }
        for metric in (
            "ATT",
            "AWT",
            "AQL",
            "invalid_response_rate",
            "generated_tokens",
            "wall_seconds",
        ):
            values = [
                statistics.mean(r[metric] for r in rows)
                for rows in units.values()
                if all(metric in r for r in rows)
            ]
            summary[metric + "_mean"] = statistics.mean(values) if values else None
            summary[metric + "_sd"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        summaries.append(summary)
    write_json(output / "summary.json", summaries)
    for name, rows in (("runs.csv", results), ("summary.csv", summaries)):
        fields = sorted(set().union(*(r.keys() for r in rows)))
        with (output / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    plt = plotting()
    credit_paths = [
        p for p in root.rglob("credit_quality.json") if not read_json(p).get("scripted")
    ]
    if credit_paths:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for path in credit_paths:
            record = read_json(path)
            if record.get("scripted"):
                continue
            methods = list(record["summary"])
            for ax, key in zip(
                axes,
                ("pair_difference_mse", "pair_difference_variance", "sign_agreement"),
            ):
                values = [record["summary"][m][key] for m in methods]
                ax.plot(
                    methods,
                    [float("nan") if v is None else v for v in values],
                    marker="o",
                    label=path.parent.name,
                )
                ax.set_ylabel(key)
        fig.tight_layout()
        fig.savefig(output / "credit-quality.png", dpi=180)
        fig.savefig(output / "credit-quality.pdf")
        plt.close(fig)
    for dataset in sorted({r["dataset"] for r in summaries}):
        data = [r for r in summaries if r["dataset"] == dataset]
        fig, axes = plt.subplots(1, 3, figsize=(max(12, len(data) * 1.6), 4.8))
        for metric, ax in zip(("ATT", "AWT", "AQL"), axes):
            ax.bar(
                range(len(data)),
                [r[metric + "_mean"] for r in data],
                yerr=(
                    [r[metric + "_sd"] or 0 for r in data]
                    if any(r[metric + "_sd"] is not None for r in data)
                    else None
                ),
                capsize=3,
                color="#287d8e",
            )
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels([r["method"] for r in data], rotation=45, ha="right")
            ax.set_ylabel(
                metric + (" (vehicles, network)" if metric == "AQL" else " (s)")
            )
        fig.suptitle(f"{dataset} | means; SD shown only when n >= 2")
        fig.tight_layout()
        fig.savefig(output / f"{dataset}-metrics.png", dpi=180)
        fig.savefig(output / f"{dataset}-metrics.pdf")
        plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for (ds, method), group in sorted(grouped.items()):
            if ds != dataset:
                continue
            series = [
                jsonlines(Path(r["source"]).parent / "decisions.jsonl") for r in group
            ]
            times = [r["time"] for r in series[0]]
            if any([r["time"] for r in rows] != times for rows in series):
                raise ValueError(f"Mismatched time axes: {dataset}/{method}")
            for ax, field in zip(axes, ("AQL_sample", "AWT_sample")):
                # Descriptive time curves only; no inferential CI across dependent observations.
                values = [
                    statistics.mean(rows[t][field] for rows in series)
                    for t in range(len(times))
                ]
                ax.plot(times, values, label=method)
                ax.set_ylabel(field)
        axes[-1].set_xlabel("Simulation time (s)")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{dataset}-queue-wait-curves.png", dpi=180)
        fig.savefig(output / f"{dataset}-queue-wait-curves.pdf")
        plt.close(fig)
    # Per-policy training curves, no interpolation across unmatched training seeds.
    logs = [
        (p, jsonlines(p))
        for p in sorted(root.rglob("training.jsonl"))
        if not (p.parent / "manifest.json").exists()
        or read_json(p.parent / "manifest.json").get("kind")
        != "scripted plumbing check"
    ]
    if logs:
        fields = [
            "validation_ATT",
            "validation_AWT",
            "validation_AQL",
            "zero_adv_group_fraction",
            "kl",
            "loss",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, field in zip(axes.flat, fields):
            for path, rows in logs:
                rows = [r for r in rows if field in r]
                if rows:
                    ax.plot(
                        [r["sim_seconds"] for r in rows],
                        [r[field] for r in rows],
                        label=path.parent.name,
                    )
            ax.set(xlabel="Training simulation seconds (including forks)", ylabel=field)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if not handles:
            handles, labels = axes.flat[3].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(4, len(labels)),
                fontsize=7,
            )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(output / "learning-curves.png", dpi=180)
        fig.savefig(output / "learning-curves.pdf")
        plt.close(fig)
        signal_fields = [
            "traffic_signal_group_fraction",
            "mean_unique_phases",
            "invalid_response_fraction",
            "truncated_fraction",
        ]
        if any(
            any(field in row for field in signal_fields)
            for _, rows in logs
            for row in rows
        ):
            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            for ax, field in zip(axes.flat, signal_fields):
                for path, rows in logs:
                    series = [r for r in rows if field in r]
                    if series:
                        ax.plot(
                            [r["sim_seconds"] for r in series],
                            [r[field] for r in series],
                            label=path.parent.name,
                        )
                ax.set(
                    xlabel="Training simulation seconds (including forks)", ylabel=field
                )
            handles, labels = axes.flat[0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    ncol=min(4, len(labels)),
                    fontsize=7,
                )
            fig.tight_layout(rect=(0, 0.08, 1, 1))
            fig.savefig(output / "training-signal.png", dpi=180)
            fig.savefig(output / "training-signal.pdf")
            plt.close(fig)
        budget_rows = []
        for path, rows in logs:
            if not rows:
                continue
            config_path = path.parent / "config.json"
            c = read_json(config_path) if config_path.exists() else {}
            budget_rows.append(
                {
                    "run": str(path.parent),
                    "method": c.get("method"),
                    "seed": c.get("seed"),
                    "sim_seconds": rows[-1]["sim_seconds"],
                    "rounds": len(rows),
                    "optimizer_steps": rows[-1].get("optimizer_steps", 0),
                    "training_generated_tokens": sum(
                        r["training_generated_tokens"] for r in rows
                    ),
                    "training_llm_calls": sum(r["training_llm_calls"] for r in rows),
                    "wall_seconds_including_validation": rows[-1]["wall_seconds"],
                }
            )
        write_json(output / "training-costs.json", budget_rows)
    markdown = [
        "# Experimental results",
        "",
        "Error bars are sample SD, not confidence intervals.",
        "Trained models: average evaluation seeds within each training seed first.",
        "",
        "| Dataset | Method | Units | ATT | AWT | AQL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summaries:
        cells = [
            f"{r[m + '_mean']:.2f}"
            + (f" ± {r[m + '_sd']:.2f}" if r[m + "_sd"] is not None else " (n=1)")
            for m in ("ATT", "AWT", "AQL")
        ]
        markdown.append(
            f"| {r['dataset']} | {r['method']} | {r['independent_units']} | "
            + " | ".join(cells)
            + " |"
        )
    (output / "RESULTS.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"runs": len(results), "groups": len(summaries), "output": str(output)}


def animate(run_dirs, output, labels=None, stride=1, fps=8, queue_max=None):
    """Queue heat circles + actual green/clearance movements; no invented car trajectories."""
    from PIL import Image
    import numpy as np

    plt = plotting()
    from matplotlib.collections import LineCollection

    runs = [Path(p) for p in run_dirs]
    if fps <= 0 or stride < 1:
        raise ValueError("fps and stride must be positive")
    logs = []
    for path in runs:
        full = jsonlines(path / "decisions.jsonl")
        selected = full[::stride]
        if selected and selected[-1]["time"] != full[-1]["time"]:
            selected.append(full[-1])
        logs.append(selected)
    manifests = [read_json(p / "manifest.json") for p in runs]
    signatures = [
        (
            m["roadnet_sha256"],
            m["flow_sha256"],
            m["horizon"],
            m["interval"],
            m["clearance_within_interval"],
            m["numpy_seed"],
        )
        for m in manifests
    ]
    if len(set(signatures)) != 1:
        raise ValueError(
            "Animation comparison requires identical roads, flow, timing and environment seed"
        )
    if not logs[0] or any(
        [r["time"] for r in log] != [r["time"] for r in logs[0]] for log in logs
    ):
        raise ValueError("Animation logs must have identical time axes")
    if labels and len(labels) != len(runs):
        raise ValueError("Need one label per run")
    if queue_max is not None and queue_max <= 0:
        raise ValueError("queue_max must be positive")
    roadnet = read_json(next((runs[0] / "engine").glob("roadnet_*.json")))
    nodes = {n["id"]: n for n in roadnet["intersections"]}
    roads = {r["id"]: r for r in roadnet["roads"]}
    names = list(logs[0][0]["junction_queues"])
    xy = np.array([[nodes[n]["point"]["x"], nodes[n]["point"]["y"]] for n in names])
    vmax = queue_max or max(
        1, max(v for log in logs for r in log for v in r["junction_queues"].values())
    )
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6), squeeze=False)
    markers, titles, greens = [], [], []
    for i, ax in enumerate(axes[0]):
        ax.add_collection(
            LineCollection(
                [[(p["x"], p["y"]) for p in r["points"]] for r in roads.values()],
                colors="#b7bdc8",
                linewidths=1.8,
                zorder=1,
            )
        )
        marker = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=np.zeros(len(names)),
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            s=180,
            edgecolors="#24374b",
            linewidths=0.5,
            zorder=3,
        )
        green = LineCollection([], colors="#12924b", linewidths=3, zorder=2)
        ax.add_collection(green)
        markers.append(marker)
        greens.append(green)
        titles.append(ax.set_title(""))
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.subplots_adjust(top=0.85, bottom=0.06, left=0.02, right=0.88, wspace=0.18)
    color_axis = fig.add_axes([0.91, 0.20, 0.013, 0.53])
    fig.colorbar(
        markers[0],
        cax=color_axis,
        label="Queued vehicles per intersection (incoming lanes)",
    )
    frames = []
    for t in range(len(logs[0])):
        for i, log in enumerate(logs):
            row = log[t]
            markers[i].set_array(np.array([row["junction_queues"][n] for n in names]))
            segments = []
            for name in names:
                phase = row.get("signal_phases", {}).get(name)
                if phase is None:
                    # New evaluation logs include actual boundary phase, including phase 0 clearance.
                    continue
                links = nodes[name]["trafficLight"]["lightphases"][phase][
                    "availableRoadLinks"
                ]
                for k in links:
                    link = nodes[name]["roadLinks"][k]
                    if link["type"] == "turn_right":
                        continue
                    for road_id in (link["startRoad"], link["endRoad"]):
                        points = roads[road_id]["points"]
                        a, b = (
                            points[-2:] if road_id == link["startRoad"] else points[:2]
                        )
                        x, y = nodes[name]["point"]["x"], nodes[name]["point"]["y"]
                        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
                        length = math.hypot(dx, dy) or 1
                        sign = -1 if road_id == link["startRoad"] else 1
                        segments.append(
                            [
                                (x, y),
                                (
                                    x + sign * dx / length * 85,
                                    y + sign * dy / length * 85,
                                ),
                            ]
                        )
            greens[i].set_segments(segments)
            label = labels[i] if labels else runs[i].name
            titles[i].set_text(
                f"{label}\nt={row['time']}s | total queue={row['AQL_sample']:.0f}"
            )
        fig.canvas.draw()
        frames.append(
            Image.fromarray(
                np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            ).convert("RGB")
        )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        target,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    frames[-1].save(target.with_suffix(".png"))
    plt.close(fig)
    return {"frames": len(frames), "queue_max": vmax, "output": str(target)}
