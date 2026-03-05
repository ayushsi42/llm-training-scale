"""
data.py — Data loading, splitting, and formatting
for the Loquace-102k Italian instruction dataset.

Creates a dedicated train/validation split for fine-tuning.
SFTTrainer handles tokenization and collation internally, so we only
return chat-formatted messages (list-of-dicts).
"""

from functools import partial

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

from config import (
    TRAIN_DATASET,
    TRAIN_SUBSET_SIZE,
    VAL_SPLIT_RATIO,
    SEED,
)


def load_raw_dataset(subset_size: int | None = TRAIN_SUBSET_SIZE) -> Dataset:
    """
    Load the Loquace-102k dataset from HuggingFace.
    Optionally select a random subset for tractable experimentation.
    """
    ds = load_dataset(TRAIN_DATASET, split="train")

    if subset_size is not None and subset_size < len(ds):
        ds = ds.shuffle(seed=SEED).select(range(subset_size))
        print(f"[data] Using subset of {subset_size} samples from {TRAIN_DATASET}")
    else:
        print(f"[data] Using full dataset: {len(ds)} samples")

    return ds


def split_dataset(ds: Dataset, val_ratio: float = VAL_SPLIT_RATIO) -> DatasetDict:
    """
    Split dataset into train and validation sets.
    Returns a DatasetDict with 'train' and 'validation' keys.
    """
    split = ds.train_test_split(test_size=val_ratio, seed=SEED)
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })


def format_instruction(example: dict) -> dict:
    """
    Format a single example as a chat-style instruction/response pair
    using the messages format expected by SFTTrainer.

    Expected Loquace-102k columns: 'instruction', 'output'
    (may also contain 'input' for context).

    Returns a 'messages' column containing a list of role/content dicts.
    """
    instruction = example.get("instruction", "")
    context = example.get("input", "")
    response = example.get("output", "")

    # Include context in the user message if present
    if context and context.strip():
        user_message = f"{instruction}\n\n{context}"
    else:
        user_message = instruction

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response},
    ]

    return {"messages": messages}


def prepare_datasets(
    model_name: str,
    subset_size: int | None = TRAIN_SUBSET_SIZE,
) -> tuple[DatasetDict, AutoTokenizer]:
    """
    Full data preparation pipeline:
    1. Load raw dataset (with optional subset)
    2. Split into train/validation
    3. Format as chat messages (for SFTTrainer)

    Returns (dataset_dict, tokenizer).
    The dataset_dict contains a 'messages' column in each split.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and split
    raw_ds = load_raw_dataset(subset_size)
    ds_dict = split_dataset(raw_ds)

    print(f"[data] Train size: {len(ds_dict['train'])}")
    print(f"[data] Validation size: {len(ds_dict['validation'])}")

    # Format instructions as chat messages
    ds_dict = ds_dict.map(format_instruction, desc="Formatting instructions")

    return ds_dict, tokenizer
