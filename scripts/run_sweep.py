#!/usr/bin/env python3
"""
run_sweep.py — Orchestrate the full learning rate sweep across all models.

For each model × each LR (3 × 10 = 30 runs):
    - Fine-tune with LoRA
    - Track via WandB
    - Save best checkpoint and results

Usage:
    python scripts/run_sweep.py
"""

import argparse
import json
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from config import (
    MODELS,
    MODEL_SIZE_LABELS,
    LR_GRID,
    WANDB_PROJECT_SWEEP,
    RESULTS_DIR,
    SEED,
)
from src.utils import set_seed, ensure_dirs
from src.data import prepare_datasets
from src.train import train


def run_sweep(target_model: str = None) -> dict:
    """
    Run the full LR sweep for all models (or a single target model).

    Returns a nested dictionary:
        results[model_name][lr_str] = {
            "best_val_loss": float,
            "best_checkpoint_path": str,
            "final_global_step": int,
            "stopped_early": bool,
        }
    """
    set_seed(SEED)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models_to_run = [target_model] if target_model else MODELS
    
    print(f"[sweep] Device: {device}")
    print(f"[sweep] Models: {len(models_to_run)}")
    print(f"[sweep] LR grid: {LR_GRID}")
    print(f"[sweep] Total runs: {len(models_to_run) * len(LR_GRID)}")

    all_results = {}
    run_count = 0
    total_runs = len(models_to_run) * len(LR_GRID)

    for model_name in models_to_run:
        # If model is not in config, use its base name as label
        size_label = MODEL_SIZE_LABELS.get(model_name, model_name.split('/')[-1])
        print(f"\n{'='*70}")
        print(f"[sweep] Model: {size_label} ({model_name})")
        print(f"{'='*70}")

        model_results = {}

        # Pre-load dataset once per model (avoids re-downloading for each LR)
        print(f"[sweep] Loading dataset for {size_label}...")
        ds_dict, tokenizer = prepare_datasets(model_name)

        for lr in LR_GRID:
            run_count += 1
            lr_str = f"{lr:.0e}"
            print(f"\n[sweep] Run {run_count}/{total_runs}: "
                  f"{size_label} LR={lr_str}")

            # Train with this specific LR
            result = train(
                model_name=model_name,
                lr=lr,
                wandb_project=WANDB_PROJECT_SWEEP,
                wandb_run_name=f"{size_label}_lr{lr_str}",
                wandb_group=size_label,
                device=device,
                ds_dict=ds_dict,
                tokenizer_preloaded=tokenizer,
            )

            model_results[lr_str] = {
                "learning_rate": lr,
                "best_val_loss": result["best_val_loss"],
                "best_checkpoint_path": result["best_checkpoint_path"],
                "final_global_step": result["final_global_step"],
                "stopped_early": result["stopped_early"],
            }

            # Retrieve existing results to not overwrite other models' data
            results_path = os.path.join(RESULTS_DIR, "sweep_results.json")
            saved_results = {}
            if os.path.exists(results_path):
                try:
                    with open(results_path, "r") as f:
                        saved_results = json.load(f)
                except json.JSONDecodeError:
                    pass
            
            # Update with current model's results
            saved_results[model_name] = {
                "model_size": size_label,
                "runs": model_results,
            }
            all_results.update(saved_results)
            _save_results(all_results)

        # ── Print per-model summary ──────────────────────────────────
        print(f"\n[sweep] Summary for {size_label}:")
        print(f"{'LR':>12} {'Val Loss':>12} {'Steps':>8} {'Early?':>8}")
        print(f"{'-'*44}")
        for lr_str, res in model_results.items():
            print(f"{lr_str:>12} {res['best_val_loss']:>12.4f} "
                  f"{res['final_global_step']:>8} "
                  f"{'Yes' if res['stopped_early'] else 'No':>8}")

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("[sweep] All runs complete!")
    print(f"{'='*70}")

    # Print best LR per model
    for model_name, data in all_results.items():
        runs = data["runs"]
        best_lr, best_loss = min(
            ((r["learning_rate"], r["best_val_loss"]) for r in runs.values()),
            key=lambda x: x[1],
        )
        print(f"  {data['model_size']:>6}: Best LR = {best_lr:.0e}, "
              f"Val Loss = {best_loss:.4f}")

    return all_results


def _save_results(results: dict) -> None:
    """Save current sweep results to disk (incremental checkpoint)."""
    results_path = os.path.join(RESULTS_DIR, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Orchestrate learning rate sweep")
    parser.add_argument(
        "--model", 
        type=str, 
        default=None, 
        help="Optional: Run sweep for a single model only (e.g., HuggingFaceTB/SmolLM2-135M-Instruct). If not specified, runs all models in config."
    )
    args = parser.parse_args()
    
    run_sweep(target_model=args.model)

if __name__ == "__main__":
    main()
