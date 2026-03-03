"""
utils.py — Utility functions for reproducibility, directory management,
and model introspection.
"""

import os
import random

import numpy as np
import torch

from config import (
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    MODEL_SIZE_LABELS,
    MODEL_PARAM_COUNTS,
)


def set_seed(seed: int) -> None:
    """
    Set random seed for full reproducibility across Python, NumPy, PyTorch,
    and CUDA. Also enables deterministic algorithms where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Request deterministic behavior (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch >= 2.0 deterministic mode
    try:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except Exception:
        pass  # Graceful fallback if not supported


def ensure_dirs() -> None:
    """Create all required output directories."""
    for d in [OUTPUT_DIR, CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)


def get_model_size_label(model_name: str) -> str:
    """Return human-readable model size label (e.g., '135M')."""
    return MODEL_SIZE_LABELS.get(model_name, model_name.split("/")[-1])


def get_model_param_count(model_name: str) -> int:
    """Return approximate total parameter count."""
    return MODEL_PARAM_COUNTS.get(model_name, 0)


def count_trainable_params(model) -> tuple[int, int]:
    """
    Count trainable and total parameters in a model.
    Returns (trainable, total).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def format_params(n: int) -> str:
    """Format a parameter count for display (e.g., 1_234_567 -> '1.23M')."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def get_checkpoint_path(model_name: str, lr: float) -> str:
    """Return the checkpoint directory path for a given model + LR run."""
    size_label = get_model_size_label(model_name)
    lr_str = f"{lr:.0e}".replace("+", "").replace("-", "m")
    return os.path.join(CHECKPOINT_DIR, f"{size_label}_lr{lr_str}")
