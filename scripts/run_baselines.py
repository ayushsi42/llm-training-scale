#!/usr/bin/env python3
"""
run_baselines.py — Zero-shot baseline evaluation for all three models
on MMLU-PRO-Ita.

Evaluates each model without any fine-tuning and logs results to WandB.

Usage:
    python scripts/run_baselines.py
"""

import json
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from config import (
    MODELS,
    MODEL_SIZE_LABELS,
    WANDB_PROJECT_BASELINES,
    WANDB_ENTITY,
    RESULTS_DIR,
    SEED,
)
from src.utils import set_seed, ensure_dirs
from src.evaluate_mmlu import evaluate_mmlu


def run_baselines() -> dict:
    """
    Evaluate all base models zero-shot on MMLU-PRO-Ita.
    Returns a dictionary mapping model names to their results.
    """
    set_seed(SEED)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baselines] Device: {device}")

    all_results = {}

    for model_name in MODELS:
        size_label = MODEL_SIZE_LABELS[model_name]
        print(f"\n{'='*60}")
        print(f"[baselines] Evaluating {size_label} ({model_name})")
        print(f"{'='*60}")

        # ── Initialize WandB run ─────────────────────────────────────
        wandb.init(
            project=WANDB_PROJECT_BASELINES,
            name=f"{size_label}_baseline",
            entity=WANDB_ENTITY,
            config={
                "model_name": model_name,
                "model_size": size_label,
                "evaluation": "zero-shot",
                "dataset": "MMLU-PRO-Ita",
                "seed": SEED,
            },
            reinit=True,
        )

        # ── Load model ───────────────────────────────────────────────
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Evaluate ─────────────────────────────────────────────────
        results = evaluate_mmlu(model, tokenizer, device)

        # ── Log to WandB ─────────────────────────────────────────────
        wandb.log({
            "mmlu/overall_accuracy": results["overall_accuracy"],
            "mmlu/num_correct": results["num_correct"],
            "mmlu/num_total": results["num_total"],
        })

        # Log per-subject accuracies
        for subject, acc in results["subject_accuracy"].items():
            wandb.log({f"mmlu/subject/{subject}": acc})

        wandb.summary["mmlu_accuracy"] = results["overall_accuracy"]
        wandb.finish()

        # ── Store results ────────────────────────────────────────────
        all_results[model_name] = {
            "model_size": size_label,
            "overall_accuracy": results["overall_accuracy"],
            "num_correct": results["num_correct"],
            "num_total": results["num_total"],
            "subject_accuracy": results["subject_accuracy"],
        }

        # Free memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save results to disk ─────────────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, "baselines.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[baselines] Results saved to {results_path}")

    # ── Print summary table ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print(f"{'-'*60}")
    for model_name, res in all_results.items():
        print(f"{res['model_size']:<20} {res['overall_accuracy']:>10.4f} "
              f"{res['num_correct']:>10} {res['num_total']:>10}")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    run_baselines()
