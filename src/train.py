"""
train.py — LoRA fine-tuning using TRL's SFTTrainer with early stopping,
validation, checkpoint saving, and WandB logging.

This module provides the core training function used by the sweep runner.
It can also be run standalone for a single training run via scripts/run_single.py.
"""

import os
import math
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
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
from src.utils import set_seed, get_checkpoint_path
from src.data import prepare_datasets


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
    Full training run for a single model + learning rate combination
    using TRL's SFTTrainer.

    Args:
        model_name: HuggingFace model identifier.
        lr: Learning rate for this run.
        wandb_project: WandB project name for logging.
        wandb_run_name: Optional custom WandB run name.
        wandb_group: Optional WandB group (e.g., model size).
        device: Torch device (used for logging only; SFTTrainer handles placement).
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

    # Load tokenizer if not preloaded
    if tokenizer_preloaded is None:
        tokenizer_preloaded = AutoTokenizer.from_pretrained(model_name)
        if tokenizer_preloaded.pad_token is None:
            tokenizer_preloaded.pad_token = tokenizer_preloaded.eos_token
            tokenizer_preloaded.pad_token_id = tokenizer_preloaded.eos_token_id

    tokenizer = tokenizer_preloaded

    # Set max sequence length on tokenizer — SFTTrainer uses this as default
    tokenizer.model_max_length = MAX_SEQ_LEN

    # ── Model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # ── LoRA Config ──────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
        lora_dropout=LORA_DROPOUT,
        task_type="CAUSAL_LM",
    )

    # ── Checkpoint path ──────────────────────────────────────────────────
    ckpt_path = get_checkpoint_path(model_name, lr)
    os.makedirs(ckpt_path, exist_ok=True)

    # Use a temporary output dir for HF Trainer checkpoints (intermediate)
    hf_output_dir = os.path.join(ckpt_path, "trainer_output")

    # ── WandB config ─────────────────────────────────────────────────────
    from config import WANDB_ENTITY, MODEL_SIZE_LABELS

    size_label = MODEL_SIZE_LABELS.get(model_name, model_name)
    if wandb_run_name is None:
        wandb_run_name = f"{size_label}_lr{lr:.0e}"

    # Set WandB environment variables so Trainer picks them up
    os.environ["WANDB_PROJECT"] = wandb_project
    if WANDB_ENTITY:
        os.environ["WANDB_ENTITY"] = WANDB_ENTITY

    # ── SFT Config (replaces TrainingArguments) ────────────────────────
    sft_config = SFTConfig(
        output_dir=hf_output_dir,
        run_name=wandb_run_name,

        # Batch size
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        # Optimizer
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,

        # Scheduler
        lr_scheduler_type=SCHEDULER_TYPE,
        warmup_steps=WARMUP_STEPS,

        # Training length
        num_train_epochs=NUM_EPOCHS,

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=VAL_STEPS,
        save_strategy="steps",
        save_steps=VAL_STEPS,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Precision
        bf16=torch.cuda.is_available(),
        fp16=False,

        # Logging
        logging_steps=10,
        report_to="wandb",

        # Reproducibility
        seed=SEED,
        data_seed=SEED,

        # Misc
        remove_unused_columns=False,
    )

    # ── SFTTrainer ───────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
            ),
        ],
    )

    print(f"[train] Learning rate: {lr}")
    print(f"[train] Checkpoint path: {ckpt_path}")

    # ── Train ────────────────────────────────────────────────────────────
    train_result = trainer.train()

    # ── Determine results ────────────────────────────────────────────────
    # Get best eval loss from trainer state
    best_val_loss = trainer.state.best_metric if trainer.state.best_metric is not None else float("inf")
    final_global_step = trainer.state.global_step

    # Check if early stopping triggered
    stopped_early = final_global_step < (
        math.ceil(len(ds_dict["train"]) / (BATCH_SIZE * GRAD_ACCUM_STEPS)) * NUM_EPOCHS
    )

    # ── Save best LoRA adapter to our checkpoint path ────────────────────
    trainer.model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"[train] ✓ Best model saved to {ckpt_path}")

    # ── Log final results ────────────────────────────────────────────────
    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["final_global_step"] = final_global_step
    wandb.summary["stopped_early"] = stopped_early

    wandb.finish()

    # ── Cleanup ──────────────────────────────────────────────────────────
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "model_name": model_name,
        "learning_rate": lr,
        "best_val_loss": best_val_loss,
        "best_checkpoint_path": ckpt_path,
        "final_global_step": final_global_step,
        "stopped_early": stopped_early,
    }

    print(f"[train] Done. best_val_loss={best_val_loss:.4f}, steps={final_global_step}")
    return result
