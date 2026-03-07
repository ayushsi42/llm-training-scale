#!/usr/bin/env python3
"""
run_post_eval.py — Post-training evaluation on MMLU-PRO-Ita.

For each model:
    - Load the best checkpoint (lowest val loss from sweep)
    - Evaluate on MMLU-PRO-Ita
    - Also evaluate all 30 checkpoints (each model × LR)
    - Log results to WandB
    - Save structured comparison table

Usage:
    python scripts/run_post_eval.py [--sizes 135M 360M 1.7B]
"""

import argparse

import json
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import wandb

from config import (
    MODELS,
    MODEL_SIZE_LABELS,
    LR_GRID,
    WANDB_PROJECT_POST_EVAL,
    WANDB_ENTITY,
    RESULTS_DIR,
    SEED,
)
from src.utils import set_seed, ensure_dirs, get_checkpoint_path
from src.evaluate_mmlu import evaluate_mmlu


def load_finetuned_model(
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
):
    """
    Load a fine-tuned LoRA model from checkpoint.

    Returns (model, tokenizer).
    """
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    if not torch.cuda.is_available():
        model = model.to(device)

    # Load tokenizer from checkpoint (same as base, but saved there)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_post_eval(selected_sizes: list[str] | None = None, batch_size: int = 8) -> dict:
    """
    Evaluate all fine-tuned checkpoints on MMLU-PRO-Ita.

    Logs results to WandB.

    Returns results dict with MMLU accuracy per model per LR.
    """
    set_seed(SEED)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[post-eval] Device: {device}")

    # ── Evaluate checkpoints ───────────────────────────────────────────
    all_results = {}

    for model_name in MODELS:
        size_label = MODEL_SIZE_LABELS[model_name]

        if selected_sizes and size_label not in selected_sizes:
            continue

        model_results = {}

        print(f"\n{'='*60}")
        print(f"[post-eval] Model: {size_label}")
        print(f"{'='*60}")

        # Evaluate ALL LR checkpoints for this model
        for lr in LR_GRID:
            lr_str = f"{lr:.0e}"
            ckpt_path = get_checkpoint_path(model_name, lr)

            if not os.path.exists(ckpt_path):
                print(f"  [post-eval] Skipping LR={lr_str}: checkpoint not found")
                continue

            print(f"\n  [post-eval] Evaluating {size_label} LR={lr_str}...")

            # Initialize WandB run
            wandb.init(
                project=WANDB_PROJECT_POST_EVAL,
                name=f"{size_label}_lr{lr_str}",
                group=size_label,
                entity=WANDB_ENTITY,
                config={
                    "model_name": model_name,
                    "model_size": size_label,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "evaluation": "post-finetuning",
                    "dataset": "MMLU-PRO-Ita",
                    "seed": SEED,
                },
                reinit=True,
            )

            # Load fine-tuned model
            model, tokenizer = load_finetuned_model(model_name, ckpt_path, device)

            # Evaluate
            mmlu_results = evaluate_mmlu(model, tokenizer, device, batch_size=batch_size)

            # Log to WandB
            wandb.log({
                "mmlu/overall_accuracy": mmlu_results["overall_accuracy"],
                "mmlu/num_correct": mmlu_results["num_correct"],
                "mmlu/num_total": mmlu_results["num_total"],
            })
            for subject, acc in mmlu_results["subject_accuracy"].items():
                wandb.log({f"mmlu/subject/{subject}": acc})

            wandb.summary["mmlu_accuracy"] = mmlu_results["overall_accuracy"]
            wandb.finish()

            model_results[lr_str] = {
                "learning_rate": lr,
                "mmlu_accuracy": mmlu_results["overall_accuracy"],
                "num_correct": mmlu_results["num_correct"],
                "num_total": mmlu_results["num_total"],
            }

            # Free memory
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[model_name] = {
            "model_size": size_label,
            "runs": model_results,
        }

    # ── Save results ─────────────────────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, "post_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[post-eval] Results saved to {results_path}")

    # ── Print comparison table ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Model':<10} {'LR':>10} {'MMLU Acc':>10}")
    print(f"{'-'*70}")
    for model_name, data in all_results.items():
        for lr_str, run_data in data["runs"].items():
            print(f"{data['model_size']:<10} {lr_str:>10} "
                  f"{run_data.get('mmlu_accuracy', 'N/A'):>10.4f}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run post-training evaluation on MMLU-PRO-Ita")
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["135M", "360M", "1.7B"],
        help="Specify which model sizes to evaluate. If not provided, evaluates all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 4).",
    )
    args = parser.parse_args()

    run_post_eval(selected_sizes=args.sizes, batch_size=args.batch_size)
