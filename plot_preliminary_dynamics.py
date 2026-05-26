"""Plot preliminary learning/forgetting curves from run_learning_curve.py logs.

Expected inputs are JSONL files named dynamics_<label>.jsonl, each containing
records from the learning and forgetting phases.

Usage:
    python plot_preliminary_dynamics.py \
        --log_dir output/learning_curve/llama \
        --out_dir paper/figures
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import fire
import matplotlib.pyplot as plt


LABEL_ORDER = ["fixed_refusal", "random", "natural_per_prompt", "fixed_natural"]
LABEL_NAMES = {
    "fixed_refusal": "Fixed refusal",
    "random": "Fixed unrelated",
    "fixed_natural": "Fixed natural",
    "natural_per_prompt": "Per-prompt natural",
}
COLORS = {
    "fixed_refusal": "#1f77b4",
    "random": "#7f7f7f",
    "fixed_natural": "#2ca02c",
    "natural_per_prompt": "#d62728",
}
LINESTYLES = {
    "fixed_refusal": "-",
    "random": "--",
    "fixed_natural": "-.",
    "natural_per_prompt": ":",
}

FIGSIZE = (5.4, 3.6)
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 13
LEGEND_SIZE = 16
LINE_WIDTH = 2.6
GRID_WIDTH = 0.7


def style_axes():
    plt.xlabel("Fine-tuning step", fontsize=AXIS_LABEL_SIZE)
    plt.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    plt.grid(alpha=0.25, linewidth=GRID_WIDTH)
    plt.legend(frameon=False, fontsize=LEGEND_SIZE)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_logs(log_dir):
    log_dir = Path(log_dir)
    grouped = defaultdict(list)
    for path in sorted(log_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            label = row.get("label_type") or path.stem.replace("dynamics_", "")
            grouped[label].append(row)
    return grouped


def aggregate(rows, phase, metric):
    by_step = defaultdict(list)
    for row in rows:
        if row.get("phase") != phase:
            continue
        value = row.get(metric)
        if value is None:
            continue
        by_step[int(row["step"])].append(float(value))
    out = []
    for step in sorted(by_step):
        values = by_step[step]
        out.append({
            "step": step,
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        })
    return out


def plot_metric(grouped, phase, metric, ylabel, out_path):
    plt.figure(figsize=FIGSIZE)
    for label in LABEL_ORDER:
        rows = grouped.get(label, [])
        series = aggregate(rows, phase, metric)
        if not series:
            continue
        xs = [r["step"] for r in series]
        ys = [r["mean"] for r in series]
        stds = [r["std"] for r in series]
        plt.plot(
            xs,
            ys,
            label=LABEL_NAMES.get(label, label),
            color=COLORS.get(label),
            linestyle=LINESTYLES.get(label, "-"),
            linewidth=LINE_WIDTH,
        )
        if any(s > 0 for s in stds):
            lo = [y - s for y, s in zip(ys, stds)]
            hi = [y + s for y, s in zip(ys, stds)]
            plt.fill_between(xs, lo, hi, color=COLORS.get(label), alpha=0.12, linewidth=0)

    plt.ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    style_axes()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def aggregate_field(rows, metric):
    by_step = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        by_step[int(row["step"])].append(float(value))
    out = []
    for step in sorted(by_step):
        values = by_step[step]
        out.append({
            "step": step,
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        })
    return out


def plot_train_val_metric(grouped, metric, ylabel, out_path):
    plt.figure(figsize=FIGSIZE)
    for label in LABEL_ORDER:
        rows = [r for r in grouped.get(label, []) if r.get("phase", "learn") == "learn"]
        series = aggregate_field(rows, metric)
        if not series:
            continue
        xs = [r["step"] for r in series]
        ys = [r["mean"] for r in series]
        stds = [r["std"] for r in series]
        plt.plot(
            xs,
            ys,
            label=LABEL_NAMES.get(label, label),
            color=COLORS.get(label),
            linestyle=LINESTYLES.get(label, "-"),
            linewidth=LINE_WIDTH,
        )
        if any(s > 0 for s in stds):
            lo = [y - s for y, s in zip(ys, stds)]
            hi = [y + s for y, s in zip(ys, stds)]
            plt.fill_between(xs, lo, hi, color=COLORS.get(label), alpha=0.12, linewidth=0)

    plt.ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    style_axes()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_train_val_phase_metric(grouped, phase, metric, ylabel, out_path, x_key="step"):
    plt.figure(figsize=FIGSIZE)
    for label in LABEL_ORDER:
        rows = [r for r in grouped.get(label, []) if r.get("phase") == phase]
        series = aggregate_field(rows, metric)
        if not series:
            continue
        xs = [r["step"] for r in series]
        ys = [r["mean"] for r in series]
        stds = [r["std"] for r in series]
        plt.plot(
            xs,
            ys,
            label=LABEL_NAMES.get(label, label),
            color=COLORS.get(label),
            linestyle=LINESTYLES.get(label, "-"),
            linewidth=LINE_WIDTH,
        )
        if any(s > 0 for s in stds):
            lo = [y - s for y, s in zip(ys, stds)]
            hi = [y + s for y, s in zip(ys, stds)]
            plt.fill_between(xs, lo, hi, color=COLORS.get(label), alpha=0.12, linewidth=0)

    plt.ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    style_axes()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main(log_dir="output/learning_curve/llama", out_dir="paper/figures"):
    grouped = load_logs(log_dir)
    if not grouped:
        raise ValueError(f"No dynamics_*.jsonl logs found in {log_dir}")

    plots = [
        ("learn", "harmful_accuracy", "Exact-match accuracy", "prelim_learning_accuracy.pdf"),
        ("learn", "harmful_rougel", "ROUGE-L", "prelim_learning_rougel.pdf"),
        ("forget", "harmful_accuracy", "Exact-match accuracy", "prelim_forgetting_accuracy.pdf"),
        ("forget", "harmful_rougel", "ROUGE-L", "prelim_forgetting_rougel.pdf"),
    ]
    for phase, metric, ylabel, filename in plots:
        if any(any(row.get("phase") == phase and metric in row for row in rows) for rows in grouped.values()):
            plot_metric(grouped, phase, metric, ylabel, Path(out_dir) / filename)
            print(f"Wrote {Path(out_dir) / filename}")

    train_val_plots = [
        ("train_rougel", "Train ROUGE-L", "prelim_train_rougel.pdf"),
        ("val_rougel", "Held-out ROUGE-L", "prelim_val_rougel.pdf"),
        ("train_accuracy", "Train exact-match accuracy", "prelim_train_accuracy.pdf"),
        ("val_accuracy", "Held-out exact-match accuracy", "prelim_val_accuracy.pdf"),
    ]
    if any(any(metric in row for row in rows) for metric, _, _ in train_val_plots for rows in grouped.values()):
        for metric, ylabel, filename in train_val_plots:
            plot_train_val_metric(grouped, metric, ylabel, Path(out_dir) / filename)
            print(f"Wrote {Path(out_dir) / filename}")

    phase_train_val_plots = [
        ("learn", "train_rougel", "Train ROUGE-L", "prelim_learning_train_rougel.pdf"),
        ("learn", "val_rougel", "Held-out ROUGE-L", "prelim_learning_val_rougel.pdf"),
        ("forget", "train_rougel", "Train ROUGE-L", "prelim_forgetting_train_rougel.pdf"),
        ("forget", "val_rougel", "Held-out ROUGE-L", "prelim_forgetting_val_rougel.pdf"),
    ]
    for phase, metric, ylabel, filename in phase_train_val_plots:
        if any(any(row.get("phase") == phase and metric in row for row in rows) for rows in grouped.values()):
            plot_train_val_phase_metric(grouped, phase, metric, ylabel, Path(out_dir) / filename)
            print(f"Wrote {Path(out_dir) / filename}")


if __name__ == "__main__":
    fire.Fire(main)
