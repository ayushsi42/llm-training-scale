"""
data.py — Data loading, splitting, formatting, and tokenization
for the Loquace-102k Italian instruction dataset.

Creates a dedicated train/validation split for fine-tuning.
"""

from functools import partial

from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import (
    TRAIN_DATASET,
    TRAIN_SUBSET_SIZE,
    VAL_SPLIT_RATIO,
    MAX_SEQ_LEN,
    BATCH_SIZE,
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


def format_instruction(example: dict, tokenizer) -> dict:
    """
    Format a single example as a chat-style instruction/response pair
    using the model's chat template.

    Expected Loquace-102k columns: 'instruction', 'output'
    (may also contain 'input' for context).
    """
    # Build the conversation in chat format
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

    # Apply chat template to get the full formatted text
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def tokenize_function(examples: dict, tokenizer) -> dict:
    """
    Tokenize formatted text with padding and truncation.
    Sets up labels for causal language modeling (labels = input_ids).
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        return_tensors=None,  # Return lists for dataset mapping
    )

    # For causal LM: labels are the same as input_ids
    # Padding tokens will be ignored in loss computation (-100)
    labels = []
    for input_ids, attention_mask in zip(
        tokenized["input_ids"], tokenized["attention_mask"]
    ):
        label = [
            token_id if mask == 1 else -100
            for token_id, mask in zip(input_ids, attention_mask)
        ]
        labels.append(label)

    tokenized["labels"] = labels
    return tokenized


def prepare_datasets(
    model_name: str,
    subset_size: int | None = TRAIN_SUBSET_SIZE,
) -> tuple[DatasetDict, AutoTokenizer]:
    """
    Full data preparation pipeline:
    1. Load raw dataset (with optional subset)
    2. Split into train/validation
    3. Format as chat instructions
    4. Tokenize

    Returns (dataset_dict, tokenizer).
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

    # Format instructions using the chat template
    format_fn = partial(format_instruction, tokenizer=tokenizer)
    ds_dict = ds_dict.map(format_fn, desc="Formatting instructions")

    # Tokenize
    tokenize_fn = partial(tokenize_function, tokenizer=tokenizer)
    ds_dict = ds_dict.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=ds_dict["train"].column_names,
        desc="Tokenizing",
    )

    # Set format for PyTorch
    ds_dict.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return ds_dict, tokenizer


def get_dataloaders(
    ds_dict: DatasetDict,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from the prepared dataset.
    Training data is shuffled; validation data is not.
    """
    train_loader = DataLoader(
        ds_dict["train"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Deterministic — no multiprocess data loading
    )

    val_loader = DataLoader(
        ds_dict["validation"],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, val_loader
