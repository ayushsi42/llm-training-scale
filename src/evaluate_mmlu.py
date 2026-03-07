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


def compute_option_logprobs_batch(
    model,
    tokenizer,
    prompts: list[str],
    options_list: list[list[str]],
    device: torch.device,
) -> list[list[float]]:
    """
    Compute log-probability of each option letter as the next token
    following the prompt for a batch of prompts.

    Returns a list of lists of log-probabilities, one per option per prompt.
    """
    # Tokenize the prompt
    # Need padding to ensure sequences in the batch have the same length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        
        # We must index using the attention mask because of padding.
        # This properly finds the last actual token in each sequence.
        attention_mask = inputs["attention_mask"]
        last_token_idx = attention_mask.sum(dim=1) - 1
        
        batch_size = outputs.logits.size(0)
        last_logits = outputs.logits[torch.arange(batch_size, device=device), last_token_idx, :]
        log_probs = F.log_softmax(last_logits, dim=-1) # shape: (batch_size, vocab_size)

    # Get log-prob for each option letter token
    # We assume options are identical letters ('A', 'B', etc.) for all questions in MMLU
    batch_option_logprobs = []
    
    # Pre-compute token IDs for A-J
    letter_token_ids = []
    for letter in OPTION_LETTERS:
        letter_ids = tokenizer.encode(letter, add_special_tokens=False)
        letter_token_ids.append(letter_ids[0])

    for i in range(len(prompts)):
        option_logprobs = []
        # We assume the number of options is the length of the list provided in `options_list[i]`
        num_opts = len(options_list[i])
        for j in range(num_opts):
            token_id = letter_token_ids[j]
            option_logprobs.append(log_probs[i, token_id].item())
        batch_option_logprobs.append(option_logprobs)

    return batch_option_logprobs


def evaluate_mmlu(
    model,
    tokenizer,
    device: torch.device | None = None,
    max_examples: int | None = None,
    batch_size: int = 4,
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

    for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating MMLU-PRO-Ita"):
        batch = examples[i:i + batch_size]
        
        prompts = []
        options_list = []
        correct_answers = []
        subjects = []
        
        for ex in batch:
            question = ex["question"]
            options = ex["options"]
            prompts.append(format_mmlu_prompt(question, options))
            options_list.append(options)
            correct_answers.append(ex["answer"])
            subjects.append(ex["subject"])

        # Compute log-probs for the batch
        batch_logprobs = compute_option_logprobs_batch(model, tokenizer, prompts, options_list, device)

        for j in range(len(batch)):
            logprobs = batch_logprobs[j]
            correct_answer = correct_answers[j]
            subject = subjects[j]
            
            # Select the option with highest log-probability
            predicted_idx = max(range(len(logprobs)), key=lambda k: logprobs[k])
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
