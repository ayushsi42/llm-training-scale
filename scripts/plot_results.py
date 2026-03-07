#!/usr/bin/env python3
"""
plot_results.py — Generate all required analysis plots from WandB experiment results.

Fetches data directly from three WandB projects:
    • lr-transfer-sweep      → sweep_results (LR sweep training runs)
    • lr-transfer-post-eval  → post_eval_results (MMLU evaluation runs)

Produces five matplotlib figures:
    1. LR vs Best Validation Loss (log-scale LR)
    2. LR vs MMLU Accuracy (log-scale LR)
    3. Model Size vs Best Validation Loss
    4. Model Size vs Best Learning Rate (log-scale LR)
    5. Validation Loss vs MMLU Accuracy (scatter)

Usage:
    python scripts/plot_results.py [--entity ENTITY] [--offline]

Options:
    --entity ENTITY   WandB entity/team name (overrides config.WANDB_ENTITY)
    --offline         Load from local JSON cache instead of WandB (fallback)
"""

import argparse
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
    WANDB_PROJECT_SWEEP,
    WANDB_PROJECT_POST_EVAL,
    WANDB_ENTITY,
)
from src.utils import ensure_dirs


# ── Plot styling ──────────────────────────────────────────────────────────────
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

# WandB run config / summary keys to look for
# The training scripts are expected to log these under run.config / run.summary
_SWEEP_LR_KEYS = ["learning_rate", "lr"]
_SWEEP_LOSS_KEYS = ["best_val_loss", "val/loss_best", "eval/loss"]
_POST_EVAL_LR_KEYS = ["learning_rate", "lr"]
_POST_EVAL_MMLU_KEYS = ["mmlu_accuracy", "mmlu/accuracy", "eval/accuracy"]
_POST_EVAL_VALLOSS_KEYS = ["val_loss", "best_val_loss", "val/loss_best", "eval/loss"]


def _pick(d: dict, keys: list, default=None):
    """Return the first value found in *d* for any key in *keys*."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _model_name_from_run(run) -> str | None:
    """
    Infer the canonical model name (HuggingFace hub ID) from a WandB run.
    Checks run.config for common key names.
    """
    cfg = dict(run.config)
    for key in ("model_name", "model", "base_model", "model_id", "pretrained_model"):
        if key in cfg and cfg[key]:
            candidate = cfg[key]
            # Match against known models
            for m in MODELS:
                if m in str(candidate) or str(candidate) in m:
                    return m
    # Fallback: try the run name (e.g. "SmolLM2-135M-Instruct_lr1e-4")
    for m in MODELS:
        short = m.split("/")[-1]   # e.g. "SmolLM2-135M-Instruct"
        if short in run.name or short.lower() in run.name.lower():
            return m
    return None


def _fetch_sweep_results(entity: str | None) -> dict:
    """
    Query the lr-transfer-sweep WandB project and build a sweep_results dict
    with the same structure previously expected from sweep_results.json:

    {
        "<model_name>": {
            "model_size": "135M",
            "runs": {
                "<lr_str>": {
                    "learning_rate": <float>,
                    "best_val_loss": <float>,
                }
            }
        }
    }
    """
    import wandb
    api = wandb.Api()
    path = f"{entity}/{WANDB_PROJECT_SWEEP}" if entity else WANDB_PROJECT_SWEEP

    print(f"[wandb] Fetching sweep runs from: {path}")
    try:
        runs = api.runs(path)
    except Exception as exc:
        print(f"[wandb] Error fetching sweep runs: {exc}")
        return {}

    sweep_results: dict = {}
    skipped = 0

    for run in runs:
        if run.state not in ("finished", "crashed"):
            # Only use completed runs
            continue

        model_name = _model_name_from_run(run)
        if model_name is None:
            skipped += 1
            continue

        cfg = dict(run.config)
        summary = dict(run.summary)

        lr = _pick(cfg, _SWEEP_LR_KEYS) or _pick(summary, _SWEEP_LR_KEYS)
        if lr is None:
            skipped += 1
            continue

        best_val_loss = _pick(summary, _SWEEP_LOSS_KEYS)
        if best_val_loss is None:
            # Try scanning history for the minimum eval loss
            try:
                hist = run.scan_history(keys=["eval/loss", "val/loss", "val_loss"])
                losses = [row.get("eval/loss") or row.get("val/loss") or row.get("val_loss")
                          for row in hist if any(k in row for k in ("eval/loss", "val/loss", "val_loss"))]
                if losses:
                    best_val_loss = min(v for v in losses if v is not None)
            except Exception:
                pass

        if best_val_loss is None:
            skipped += 1
            continue

        size_label = MODEL_SIZE_LABELS[model_name]
        lr_str = str(lr)

        if model_name not in sweep_results:
            sweep_results[model_name] = {"model_size": size_label, "runs": {}}

        sweep_results[model_name]["runs"][lr_str] = {
            "learning_rate": float(lr),
            "best_val_loss": float(best_val_loss),
        }

    print(f"[wandb] Sweep: {sum(len(v['runs']) for v in sweep_results.values())} runs loaded "
          f"across {len(sweep_results)} models ({skipped} skipped)")
    return sweep_results


def _fetch_post_eval_results(entity: str | None) -> dict:
    """
    Query the lr-transfer-post-eval WandB project and build a post_eval_results dict:

    {
        "<model_name>": {
            "model_size": "135M",
            "runs": {
                "<lr_str>": {
                    "learning_rate": <float>,
                    "mmlu_accuracy": <float>,   # [0, 1]
                    "val_loss": <float | None>,
                }
            }
        }
    }
    """
    import wandb
    api = wandb.Api()
    path = f"{entity}/{WANDB_PROJECT_POST_EVAL}" if entity else WANDB_PROJECT_POST_EVAL

    print(f"[wandb] Fetching post-eval runs from: {path}")
    try:
        runs = api.runs(path)
    except Exception as exc:
        print(f"[wandb] Error fetching post-eval runs: {exc}")
        return {}

    post_eval_results: dict = {}
    skipped = 0

    for run in runs:
        if run.state not in ("finished", "crashed"):
            continue

        model_name = _model_name_from_run(run)
        if model_name is None:
            skipped += 1
            continue

        cfg = dict(run.config)
        summary = dict(run.summary)

        lr = _pick(cfg, _POST_EVAL_LR_KEYS) or _pick(summary, _POST_EVAL_LR_KEYS)
        if lr is None:
            skipped += 1
            continue

        mmlu_accuracy = _pick(summary, _POST_EVAL_MMLU_KEYS)
        if mmlu_accuracy is None:
            skipped += 1
            continue

        # Normalise to [0, 1] if stored as percentage
        if mmlu_accuracy > 1.0:
            mmlu_accuracy /= 100.0

        val_loss = _pick(summary, _POST_EVAL_VALLOSS_KEYS)

        size_label = MODEL_SIZE_LABELS[model_name]
        lr_str = str(lr)

        if model_name not in post_eval_results:
            post_eval_results[model_name] = {"model_size": size_label, "runs": {}}

        post_eval_results[model_name]["runs"][lr_str] = {
            "learning_rate": float(lr),
            "mmlu_accuracy": float(mmlu_accuracy),
            "val_loss": float(val_loss) if val_loss is not None else None,
        }

    print(f"[wandb] Post-eval: {sum(len(v['runs']) for v in post_eval_results.values())} runs loaded "
          f"across {len(post_eval_results)} models ({skipped} skipped)")
    return post_eval_results


def load_results_from_wandb(entity: str | None = None) -> tuple[dict, dict]:
    """
    Fetch sweep and post-eval results directly from WandB.
    Returns (sweep_results, post_eval_results).
    """
    effective_entity = entity or WANDB_ENTITY
    sweep_results = _fetch_sweep_results(effective_entity)
    post_eval_results = _fetch_post_eval_results(effective_entity)
    return sweep_results, post_eval_results


def load_results_from_local() -> tuple[dict, dict]:
    """
    Fallback: load sweep and post-eval results from local JSON files.
    Returns (sweep_results, post_eval_results).
    """
    sweep_path = os.path.join(RESULTS_DIR, "sweep_results.json")
    post_eval_path = os.path.join(RESULTS_DIR, "post_eval_results.json")

    sweep_results = {}
    post_eval_results = {}

    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            sweep_results = json.load(f)
        print(f"[local] Loaded sweep results from {sweep_path}")
    else:
        print(f"[local] Warning: {sweep_path} not found")

    if os.path.exists(post_eval_path):
        with open(post_eval_path) as f:
            post_eval_results = json.load(f)
        print(f"[local] Loaded post-eval results from {post_eval_path}")
    else:
        print(f"[local] Warning: {post_eval_path} not found")

    return sweep_results, post_eval_results


# ── Plot functions ────────────────────────────────────────────────────────────

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


# ── Main entrypoint ───────────────────────────────────────────────────────────

def generate_all_plots(entity: str | None = None, offline: bool = False) -> None:
    """Fetch data from WandB (or local cache) and generate all five analysis plots."""
    ensure_dirs()

    if offline:
        print("[plot] Offline mode: loading from local JSON files")
        sweep_results, post_eval_results = load_results_from_local()
    else:
        try:
            import wandb  # noqa: F401
        except ImportError:
            print("[plot] wandb not installed — falling back to local files. "
                  "Install with: pip install wandb")
            sweep_results, post_eval_results = load_results_from_local()
        else:
            sweep_results, post_eval_results = load_results_from_wandb(entity=entity)

    if sweep_results:
        plot_lr_vs_val_loss(sweep_results)
        plot_model_size_vs_val_loss(sweep_results)
        plot_model_size_vs_best_lr(sweep_results)
    else:
        print("[plot] Cannot generate sweep-based plots: no sweep results found")

    if post_eval_results:
        plot_lr_vs_mmlu_accuracy(post_eval_results)
        plot_val_loss_vs_mmlu(post_eval_results)
    else:
        print("[plot] Cannot generate MMLU plots: no post-eval results found")

    print("\n[plot] All plots generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate analysis plots from WandB data")
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity/team name (overrides config.WANDB_ENTITY)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load from local JSON files instead of WandB",
    )
    args = parser.parse_args()
    generate_all_plots(entity=args.entity, offline=args.offline)
