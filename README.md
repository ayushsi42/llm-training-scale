# Learning Rate Transfer Across Model Sizes

Does the optimal learning rate transfer across model sizes? This project investigates
that question by running a controlled learning rate sweep with LoRA fine-tuning on
three SmolLM2 model variants (135M, 360M, 1.7B) using Italian instruction data, then
evaluating on MMLU-PRO-Ita.

## Project Structure

```
llm-training-scale/
├── config.py                 # All hyperparameters, paths, LR grid
├── requirements.txt          # Python dependencies
├── README.md
│
├── src/                      # Core library modules
│   ├── __init__.py
│   ├── utils.py              # Seed setting, reproducibility, helpers
│   ├── data.py               # Data loading, train/val split, tokenization
│   ├── evaluate_mmlu.py      # MMLU-PRO-Ita evaluation (log-likelihood)
│   └── train.py              # LoRA fine-tuning with early stopping
│
├── scripts/                  # Executable entry points
│   ├── run_baselines.py      # Step 1: Zero-shot MMLU evaluation
│   ├── run_sweep.py          # Step 2: Full LR sweep (30 runs)
│   ├── run_post_eval.py      # Step 3: Post-fine-tuning MMLU eval
│   ├── plot_results.py       # Step 4: Generate analysis plots
│   └── run_single.py         # Train a single model + LR (utility)
│
└── outputs/                  # Generated at runtime
    ├── checkpoints/          # Best model checkpoints per run
    ├── results/              # JSON result files
    └── plots/                # Matplotlib figures (PNG)
```

## Setup

```bash
pip install -r requirements.txt
wandb login
```

## Experiment Pipeline

Run all steps sequentially from the **project root**:

```bash
# 1. Baseline — zero-shot MMLU-PRO-Ita evaluation (3 models)
python scripts/run_baselines.py

# 2. LR Sweep — fine-tune all models across 10 learning rates (30 runs)
python scripts/run_sweep.py

# 3. Post-Training Evaluation — MMLU-PRO-Ita on all fine-tuned checkpoints
python scripts/run_post_eval.py

# 4. Plotting — generate all 4 analysis figures
python scripts/plot_results.py
```

For a single standalone training run:

```bash
python scripts/run_single.py --model HuggingFaceTB/SmolLM2-135M-Instruct --lr 1e-4
```

## Models

| Model | Parameters |
|-------|-----------|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B |

## Datasets

| Dataset | Role |
|---------|------|
| `cosimoiaia/Loquace-102k` | Training (Italian instruction pairs) |
| `efederici/MMLU-Pro-ita` | Evaluation (Italian MMLU-Pro, 10 options) |

## Fixed Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 32 |
| LoRA alpha | 128 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Batch size (effective) | 32 (4 × 8 accumulation) |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Scheduler | Cosine with 100 warmup steps |
| Max sequence length | 512 |
| Epochs | 3 |
| Early stopping patience | 3 validations |
| Validation frequency | Every 100 steps |

## Learning Rate Grid

```
[5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
```

## Output Plots

1. **LR vs Validation Loss** — log-scale x-axis, one curve per model
2. **LR vs MMLU Accuracy** — log-scale x-axis, one curve per model
3. **Model Size vs Best Validation Loss** — bar chart
4. **Model Size vs Best Learning Rate** — log-scale y-axis

## WandB Projects

| Project | Contents |
|---------|----------|
| `lr-transfer-baselines` | Zero-shot MMLU evaluations |
| `lr-transfer-sweep` | All 30 training runs |
| `lr-transfer-post-eval` | Post-fine-tuning MMLU evaluations |

## Configuration

Edit `config.py` to adjust:
- `TRAIN_SUBSET_SIZE` — number of training samples (default: 20k)
- `BATCH_SIZE` / `GRAD_ACCUM_STEPS` — adjust for your GPU memory
- `WANDB_ENTITY` — your WandB team/entity
