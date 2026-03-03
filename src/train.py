"""
train.py — LoRA fine-tuning loop with early stopping, validation,
checkpoint saving, and WandB logging.

This module provides the core training function used by the sweep runner.
It can also be run standalone for a single training run via scripts/run_single.py.
"""

import os
import math
import json
import time
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import wandb

from config import (
    LORA_R,
    LORA_ALPHA,
    LORA_TARGET_MODULES,
    LORA_BIAS,
    LORA_DROPOUT,
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    WEIGHT_DECAY,
    SCHEDULER_TYPE,
    WARMUP_STEPS,
    MAX_SEQ_LEN,
    NUM_EPOCHS,
    VAL_STEPS,
    EARLY_STOPPING_PATIENCE,
    SEED,
)
from src.utils import set_seed, count_trainable_params, format_params, get_checkpoint_path
from src.data import prepare_datasets, get_dataloaders


def setup_model(
    model_name: str,
    lr: float,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Load base model, apply LoRA, and create optimizer + scheduler.

    Returns: (model, tokenizer, optimizer)
    The scheduler is created later once we know the total training steps.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model in appropriate precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Move to device if no device_map
    if not torch.cuda.is_available():
        model = model.to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
        lora_dropout=LORA_DROPOUT,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Report trainable parameters
    trainable, total = count_trainable_params(model)
    print(f"[train] Trainable params: {format_params(trainable)} / {format_params(total)}")
    print(f"[train] Trainable ratio:  {trainable / total:.4%}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    return model, tokenizer, optimizer


@torch.no_grad()
def validate(model, val_loader: DataLoader, device: torch.device) -> float:
    """
    Compute average validation loss over the entire validation set.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else float("inf")
    model.train()
    return avg_loss


def train(
    model_name: str,
    lr: float,
    wandb_project: str,
    wandb_run_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    device: Optional[torch.device] = None,
    ds_dict=None,
    tokenizer_preloaded=None,
) -> dict:
    """
    Full training run for a single model + learning rate combination.

    Args:
        model_name: HuggingFace model identifier.
        lr: Learning rate for this run.
        wandb_project: WandB project name for logging.
        wandb_run_name: Optional custom WandB run name.
        wandb_group: Optional WandB group (e.g., model size).
        device: Torch device. If None, auto-detect.
        ds_dict: Pre-loaded dataset dict (to avoid re-loading for each LR).
        tokenizer_preloaded: Pre-loaded tokenizer.

    Returns:
        Dictionary with:
            - best_val_loss: float
            - best_checkpoint_path: str
            - final_global_step: int
            - stopped_early: bool
    """
    set_seed(SEED)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ─────────────────────────────────────────────────────────────
    if ds_dict is None:
        ds_dict, tokenizer_preloaded = prepare_datasets(model_name)

    train_loader, val_loader = get_dataloaders(ds_dict)

    # ── Model + Optimizer ────────────────────────────────────────────────
    model, tokenizer, optimizer = setup_model(model_name, lr, device)

    # ── Scheduler ────────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_training_steps = steps_per_epoch * NUM_EPOCHS

    lr_scheduler = get_scheduler(
        name=SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_training_steps,
    )

    print(f"[train] Steps per epoch: {steps_per_epoch}")
    print(f"[train] Total training steps: {total_training_steps}")
    print(f"[train] Learning rate: {lr}")

    # ── Checkpoint path ──────────────────────────────────────────────────
    ckpt_path = get_checkpoint_path(model_name, lr)
    os.makedirs(ckpt_path, exist_ok=True)

    # ── WandB init ───────────────────────────────────────────────────────
    from config import WANDB_ENTITY, MODEL_SIZE_LABELS

    size_label = MODEL_SIZE_LABELS.get(model_name, model_name)
    if wandb_run_name is None:
        wandb_run_name = f"{size_label}_lr{lr:.0e}"

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        group=wandb_group or size_label,
        entity=WANDB_ENTITY,
        config={
            "model_name": model_name,
            "model_size": size_label,
            "learning_rate": lr,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_target_modules": LORA_TARGET_MODULES,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": SCHEDULER_TYPE,
            "warmup_steps": WARMUP_STEPS,
            "max_seq_len": MAX_SEQ_LEN,
            "num_epochs": NUM_EPOCHS,
            "val_steps": VAL_STEPS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "seed": SEED,
        },
        reinit=True,
    )

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    stopped_early = False
    accumulated_loss = 0.0
    micro_steps = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\n[train] === Epoch {epoch + 1}/{NUM_EPOCHS} ===")

        for batch_idx, batch in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch + 1}", leave=False
        )):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM_STEPS  # Scale loss for accumulation
            loss.backward()

            accumulated_loss += outputs.loss.item()
            micro_steps += 1

            # Gradient accumulation step
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                avg_train_loss = accumulated_loss / micro_steps

                # Log training metrics
                wandb.log({
                    "train/loss": avg_train_loss,
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                    "train/epoch": epoch + (batch_idx + 1) / len(train_loader),
                })

                accumulated_loss = 0.0
                micro_steps = 0

                # ── Validation ───────────────────────────────────────
                if global_step % VAL_STEPS == 0:
                    val_loss = validate(model, val_loader, device)

                    wandb.log({
                        "val/loss": val_loss,
                        "val/global_step": global_step,
                    })

                    print(f"  [step {global_step}] val_loss={val_loss:.4f}  "
                          f"best={best_val_loss:.4f}")

                    # Check for improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0

                        # Save best checkpoint
                        model.save_pretrained(ckpt_path)
                        tokenizer.save_pretrained(ckpt_path)
                        print(f"  [step {global_step}] ✓ New best! Saved to {ckpt_path}")
                    else:
                        patience_counter += 1
                        print(f"  [step {global_step}] ✗ No improvement "
                              f"({patience_counter}/{EARLY_STOPPING_PATIENCE})")

                    # Early stopping check
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f"\n[train] Early stopping at step {global_step}")
                        stopped_early = True
                        break

        if stopped_early:
            break

    # ── Final validation if not already done ─────────────────────────────
    if global_step % VAL_STEPS != 0 and not stopped_early:
        val_loss = validate(model, val_loader, device)
        wandb.log({"val/loss": val_loss, "val/global_step": global_step})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

    # ── Log final results ────────────────────────────────────────────────
    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["final_global_step"] = global_step
    wandb.summary["stopped_early"] = stopped_early

    wandb.finish()

    # ── Cleanup GPU memory ───────────────────────────────────────────────
    del model, optimizer, lr_scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "model_name": model_name,
        "learning_rate": lr,
        "best_val_loss": best_val_loss,
        "best_checkpoint_path": ckpt_path,
        "final_global_step": global_step,
        "stopped_early": stopped_early,
    }

    print(f"[train] Done. best_val_loss={best_val_loss:.4f}, steps={global_step}")
    return result
