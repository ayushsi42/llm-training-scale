"""
config.py — Central configuration for the LR Transfer experiment.

All hyperparameters, model definitions, dataset references, and paths
are defined here. Only the learning rate varies across runs.
"""

import os

# ============================================================================
# Reproducibility
# ============================================================================
SEED = 42

# ============================================================================
# Models
# ============================================================================
MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]

# Human-readable labels for plotting / logging
MODEL_SIZE_LABELS = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": "135M",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "360M",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "1.7B",
}

# Approximate parameter counts (for x-axis in plots)
MODEL_PARAM_COUNTS = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": 135_000_000,
    "HuggingFaceTB/SmolLM2-360M-Instruct": 360_000_000,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 1_700_000_000,
}

# ============================================================================
# Datasets
# ============================================================================
TRAIN_DATASET = "cosimoiaia/Loquace-102k"
EVAL_DATASET = "efederici/MMLU-Pro-ita"

# Subset size for training data (set to None to use full dataset)
TRAIN_SUBSET_SIZE = 4_000

# Validation split ratio
VAL_SPLIT_RATIO = 0.15

# ============================================================================
# LoRA Configuration (FIXED — do not change)
# ============================================================================
LORA_R = 32
LORA_ALPHA = 128
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_BIAS = "none"
LORA_DROPOUT = 0.0

# ============================================================================
# Training Hyperparameters (FIXED — do not change except LR)
# ============================================================================
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8          # effective batch size = 4 * 8 = 32
WEIGHT_DECAY = 0.01
SCHEDULER_TYPE = "cosine"
WARMUP_STEPS = 100
MAX_SEQ_LEN = 512
NUM_EPOCHS = 5

# ============================================================================
# Validation & Early Stopping
# ============================================================================
VAL_STEPS = 100                # validate every N global steps
EARLY_STOPPING_PATIENCE = 3   # stop after N consecutive non-improving evals

# ============================================================================
# Learning Rate Sweep Grid
# ============================================================================
LR_GRID = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]

# ============================================================================
# WandB
# ============================================================================
WANDB_PROJECT_BASELINES = "lr-transfer-baselines"
WANDB_PROJECT_SWEEP = "lr-transfer-sweep"
WANDB_PROJECT_POST_EVAL = "lr-transfer-post-eval"
WANDB_ENTITY = None  # Set to your WandB entity/team, or None for default

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
