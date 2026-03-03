#!/usr/bin/env python3
"""
run_single.py — Train a single model with a given learning rate.

Convenience script for standalone training runs.

Usage:
    python scripts/run_single.py --model HuggingFaceTB/SmolLM2-135M-Instruct --lr 1e-4
"""

import argparse
import json
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Train a single model with a given LR")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--wandb-project", type=str, default="lr-transfer-sweep")
    args = parser.parse_args()

    result = train(
        model_name=args.model,
        lr=args.lr,
        wandb_project=args.wandb_project,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
