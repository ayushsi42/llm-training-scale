"""
evaluate_mmlu.py — Zero-shot and post-finetuning evaluation on MMLU-PRO-Ita.

Implements log-likelihood scoring: for each question with 10 answer options,
compute the log-probability of each option and select the most likely one.
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

from config import EVAL_DATASET, SEED


# MMLU-Pro uses letters A-J for 10 options
OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def load_mmlu_pro_ita() -> list[dict]:
    """
    Load MMLU-PRO-Ita dataset and return as a list of structured examples.

    Each example contains:
        - question: str
        - options: list[str]  (up to 10 answer choices)
        - answer: str         (correct option letter, e.g. "A")
        - subject: str        (category/subject)
    """
    ds = load_dataset(EVAL_DATASET, split="test")

    examples = []
    for row in ds:
        example = {
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"],
            "subject": row.get("category", row.get("subject", "unknown")),
        }
        examples.append(example)

    print(f"[eval] Loaded {len(examples)} MMLU-PRO-Ita examples")
    return examples


def format_mmlu_prompt(question: str, options: list[str]) -> str:
    """
    Format an MMLU question with its answer options for zero-shot evaluation.

    Returns a prompt string that ends with a cue for the model to output
    the answer letter.
    """
    prompt = f"Domanda: {question}\n\n"
    for i, option in enumerate(options):
        letter = OPTION_LETTERS[i]
        prompt += f"{letter}. {option}\n"
    prompt += "\nRisposta: "
    return prompt


def compute_option_logprobs(
    model,
    tokenizer,
    prompt: str,
    options: list[str],
    device: torch.device,
) -> list[float]:
    """
    Compute log-probability of each option letter as the next token
    following the prompt.

    Returns a list of log-probabilities, one per option.
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits for the last position (where the answer should go)
        last_logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)
        log_probs = F.log_softmax(last_logits, dim=-1)

    # Get log-prob for each option letter token
    option_logprobs = []
    for i in range(len(options)):
        letter = OPTION_LETTERS[i]
        # Tokenize just the letter to get its token ID
        letter_ids = tokenizer.encode(letter, add_special_tokens=False)
        # Use the first token if the letter maps to multiple tokens
        token_id = letter_ids[0]
        option_logprobs.append(log_probs[token_id].item())

    return option_logprobs


def evaluate_mmlu(
    model,
    tokenizer,
    device: torch.device | None = None,
    max_examples: int | None = None,
) -> dict:
    """
    Run full MMLU-PRO-Ita evaluation.

    Args:
        model: The language model (base or fine-tuned).
        tokenizer: The corresponding tokenizer.
        device: Device to run on. If None, auto-detect.
        max_examples: Limit evaluation to N examples (for debugging).

    Returns:
        Dictionary with:
            - overall_accuracy: float
            - subject_accuracy: dict[str, float]
            - num_correct: int
            - num_total: int
            - per_subject_counts: dict[str, dict]
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    examples = load_mmlu_pro_ita()

    if max_examples is not None:
        examples = examples[:max_examples]

    # Track results per subject
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    total_correct = 0
    total = 0

    for ex in tqdm(examples, desc="Evaluating MMLU-PRO-Ita"):
        question = ex["question"]
        options = ex["options"]
        correct_answer = ex["answer"]
        subject = ex["subject"]

        # Format prompt and compute log-probs
        prompt = format_mmlu_prompt(question, options)
        logprobs = compute_option_logprobs(model, tokenizer, prompt, options, device)

        # Select the option with highest log-probability
        predicted_idx = max(range(len(logprobs)), key=lambda i: logprobs[i])
        predicted_answer = OPTION_LETTERS[predicted_idx]

        # Check correctness
        is_correct = predicted_answer == correct_answer
        if is_correct:
            total_correct += 1
            subject_correct[subject] += 1

        total += 1
        subject_total[subject] += 1

    # Compute accuracies
    overall_accuracy = total_correct / total if total > 0 else 0.0

    subject_accuracy = {}
    per_subject_counts = {}
    for subj in sorted(subject_total.keys()):
        acc = subject_correct[subj] / subject_total[subj]
        subject_accuracy[subj] = acc
        per_subject_counts[subj] = {
            "correct": subject_correct[subj],
            "total": subject_total[subj],
            "accuracy": acc,
        }

    results = {
        "overall_accuracy": overall_accuracy,
        "subject_accuracy": subject_accuracy,
        "num_correct": total_correct,
        "num_total": total,
        "per_subject_counts": per_subject_counts,
    }

    print(f"[eval] Overall accuracy: {overall_accuracy:.4f} ({total_correct}/{total})")
    return results
