#!/usr/bin/env python3
"""
plot_results.py — Generate all required analysis plots from experiment results.

Produces four matplotlib figures:
    1. LR vs Best Validation Loss (log-scale LR)
    2. LR vs MMLU Accuracy (log-scale LR)
    3. Model Size vs Best Validation Loss
    4. Model Size vs Best Learning Rate (log-scale LR)

Usage:
    python scripts/plot_results.py
"""

import json
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from config import (
    MODELS,
    MODEL_SIZE_LABELS,
    MODEL_PARAM_COUNTS,
    RESULTS_DIR,
    PLOTS_DIR,
)
from src.utils import ensure_dirs


# ── Plot styling ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

# One color per model size
COLORS = {
    "135M": "#2196F3",
    "360M": "#FF9800",
    "1.7B": "#4CAF50",
}

MARKERS = {
    "135M": "o",
    "360M": "s",
    "1.7B": "D",
}


def load_results() -> tuple[dict, dict]:
    """
    Load sweep and post-eval results from JSON files.
    Returns (sweep_results, post_eval_results).
    """
    sweep_path = os.path.join(RESULTS_DIR, "sweep_results.json")
    post_eval_path = os.path.join(RESULTS_DIR, "post_eval_results.json")

    sweep_results = {}
    post_eval_results = {}

    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            sweep_results = json.load(f)
    else:
        print(f"[plot] Warning: {sweep_path} not found")

    if os.path.exists(post_eval_path):
        with open(post_eval_path) as f:
            post_eval_results = json.load(f)
    else:
        print(f"[plot] Warning: {post_eval_path} not found")

    return sweep_results, post_eval_results


def plot_lr_vs_val_loss(sweep_results: dict) -> None:
    """
    Plot 1: Learning Rate vs Best Validation Loss.
    Log-scale LR on x-axis, one line per model.
    """
    fig, ax = plt.subplots()

    for model_name in MODELS:
        if model_name not in sweep_results:
            continue

        data = sweep_results[model_name]
        size_label = data["model_size"]
        runs = data["runs"]

        lrs = []
        val_losses = []
        for lr_str, run in sorted(runs.items(), key=lambda x: x[1]["learning_rate"]):
            lrs.append(run["learning_rate"])
            val_losses.append(run["best_val_loss"])

        ax.plot(
            lrs, val_losses,
            marker=MARKERS[size_label],
            color=COLORS[size_label],
            label=size_label,
            linewidth=2,
            markersize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Learning Rate vs Best Validation Loss")
    ax.legend(title="Model Size")
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(PLOTS_DIR, "lr_vs_val_loss.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


def plot_lr_vs_mmlu_accuracy(post_eval_results: dict) -> None:
    """
    Plot 2: Learning Rate vs MMLU Accuracy.
    Log-scale LR on x-axis, one line per model.
    """
    fig, ax = plt.subplots()

    for model_name in MODELS:
        if model_name not in post_eval_results:
            continue

        data = post_eval_results[model_name]
        size_label = data["model_size"]
        runs = data["runs"]

        lrs = []
        accuracies = []
        for lr_str, run in sorted(runs.items(), key=lambda x: x[1]["learning_rate"]):
            lrs.append(run["learning_rate"])
            accuracies.append(run["mmlu_accuracy"] * 100)  # Convert to percentage

        ax.plot(
            lrs, accuracies,
            marker=MARKERS[size_label],
            color=COLORS[size_label],
            label=size_label,
            linewidth=2,
            markersize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("MMLU-PRO-Ita Accuracy (%)")
    ax.set_title("Learning Rate vs MMLU-PRO-Ita Accuracy")
    ax.legend(title="Model Size")
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(PLOTS_DIR, "lr_vs_mmlu_accuracy.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


def plot_model_size_vs_val_loss(sweep_results: dict) -> None:
    """
    Plot 3: Model Size vs Best Validation Loss.
    Shows the best val loss achieved across all LRs for each model.
    """
    fig, ax = plt.subplots()

    sizes = []
    labels = []
    best_losses = []
    colors = []

    for model_name in MODELS:
        if model_name not in sweep_results:
            continue

        data = sweep_results[model_name]
        size_label = data["model_size"]
        runs = data["runs"]

        # Find best val loss across all LRs
        best_loss = min(r["best_val_loss"] for r in runs.values())

        sizes.append(MODEL_PARAM_COUNTS[model_name])
        labels.append(size_label)
        best_losses.append(best_loss)
        colors.append(COLORS[size_label])

    ax.bar(
        range(len(sizes)),
        best_losses,
        color=colors,
        width=0.6,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model Size")
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Model Size vs Best Validation Loss")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (loss, label) in enumerate(zip(best_losses, labels)):
        ax.text(i, loss + 0.01, f"{loss:.4f}", ha="center", va="bottom", fontsize=10)

    save_path = os.path.join(PLOTS_DIR, "model_size_vs_val_loss.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


def plot_model_size_vs_best_lr(sweep_results: dict) -> None:
    """
    Plot 4: Model Size vs Best Learning Rate (log-scale y-axis).
    Shows how the optimal LR changes with model scale.
    """
    fig, ax = plt.subplots()

    sizes = []
    labels = []
    best_lrs = []
    colors = []

    for model_name in MODELS:
        if model_name not in sweep_results:
            continue

        data = sweep_results[model_name]
        size_label = data["model_size"]
        runs = data["runs"]

        # Find LR with lowest val loss
        best_run = min(runs.values(), key=lambda r: r["best_val_loss"])
        best_lr = best_run["learning_rate"]

        sizes.append(MODEL_PARAM_COUNTS[model_name])
        labels.append(size_label)
        best_lrs.append(best_lr)
        colors.append(COLORS[size_label])

    ax.scatter(
        range(len(sizes)),
        best_lrs,
        c=colors,
        s=200,
        zorder=5,
        edgecolors="black",
        linewidth=1,
    )

    # Connect with a line
    ax.plot(range(len(sizes)), best_lrs, color="gray", linestyle="--", alpha=0.5)

    ax.set_yscale("log")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model Size")
    ax.set_ylabel("Best Learning Rate")
    ax.set_title("Model Size vs Best Learning Rate")
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, lr in enumerate(best_lrs):
        ax.annotate(
            f"{lr:.0e}",
            (i, lr),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=10,
        )

    save_path = os.path.join(PLOTS_DIR, "model_size_vs_best_lr.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


def plot_val_loss_vs_mmlu(post_eval_results: dict) -> None:
    """
    Plot 5: Validation Loss vs MMLU Accuracy (scatter).
    Each point is one (model, LR) run. Colored by model size.
    Shows the relationship between validation loss and downstream performance.
    """
    fig, ax = plt.subplots()

    for model_name in MODELS:
        if model_name not in post_eval_results:
            continue

        data = post_eval_results[model_name]
        size_label = data["model_size"]
        runs = data["runs"]

        val_losses = []
        accuracies = []
        for lr_str, run in runs.items():
            if run.get("val_loss") is not None:
                val_losses.append(run["val_loss"])
                accuracies.append(run["mmlu_accuracy"] * 100)

        ax.scatter(
            val_losses, accuracies,
            marker=MARKERS[size_label],
            color=COLORS[size_label],
            label=size_label,
            s=100,
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )

    ax.set_xlabel("Best Validation Loss")
    ax.set_ylabel("MMLU-PRO-Ita Accuracy (%)")
    ax.set_title("Validation Loss vs MMLU-PRO-Ita Accuracy")
    ax.legend(title="Model Size")
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(PLOTS_DIR, "val_loss_vs_mmlu_accuracy.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


def generate_all_plots() -> None:
    """Generate all five analysis plots."""
    ensure_dirs()

    sweep_results, post_eval_results = load_results()

    if sweep_results:
        plot_lr_vs_val_loss(sweep_results)
        plot_model_size_vs_val_loss(sweep_results)
        plot_model_size_vs_best_lr(sweep_results)
    else:
        print("[plot] Cannot generate sweep-based plots: no sweep results")

    if post_eval_results:
        plot_lr_vs_mmlu_accuracy(post_eval_results)
        plot_val_loss_vs_mmlu(post_eval_results)
    else:
        print("[plot] Cannot generate MMLU plots: no post-eval results")

    print("\n[plot] All plots generated!")


if __name__ == "__main__":
    generate_all_plots()
