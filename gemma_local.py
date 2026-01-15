#!/usr/bin/env python3
"""
Local Gemma 3 4B inference with logits access.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from linkedin_data import get_pairs
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

MODEL_ID = "google/gemma-3-1b-it"  # 1B fits in 8GB without quantization

# Global model and tokenizer
model = None
tokenizer = None


def load_model():
    global model, tokenizer
    print(f"Loading {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def get_ab_logits(prompt: str) -> tuple[float, float]:
    """Get log probabilities for A and B tokens."""
    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :].float()  # Convert to float32 for stability

        # Get token IDs for A and B
        a_id = tokenizer.encode("A", add_special_tokens=False)[0]
        b_id = tokenizer.encode("B", add_special_tokens=False)[0]

        # Get log probs
        log_probs = torch.log_softmax(logits, dim=-1)
        a_logprob = log_probs[a_id].item()
        b_logprob = log_probs[b_id].item()

    return a_logprob, b_logprob


def generate_response(prompt: str, max_new_tokens: int = 32) -> str:
    """Generate text response from the model."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def parse_ab_response(response: str) -> int:
    """Parse A/B response. Returns 0 for A, 1 for B, -1 for unparseable."""
    response_upper = response.upper().strip()

    # Check for clear A or B
    if response_upper.startswith("A") or response_upper == "A":
        return 0
    if response_upper.startswith("B") or response_upper == "B":
        return 1

    # Check last line
    lines = response.strip().split('\n')
    last_line = lines[-1].strip().upper()
    if "A" in last_line and "B" not in last_line:
        return 0
    if "B" in last_line and "A" not in last_line:
        return 1

    # Count mentions
    a_count = response_upper.count(" A ") + response_upper.count(" A.") + response_upper.count("(A)")
    b_count = response_upper.count(" B ") + response_upper.count(" B.") + response_upper.count("(B)")

    if a_count > b_count:
        return 0
    if b_count > a_count:
        return 1

    return -1


def predict_from_generation(post_a: str, post_b: str) -> int:
    """Predict which post wins using text generation."""
    prompt = f"""You're scrolling LinkedIn. You see these two posts. Which one makes you want to engage more?

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Trust your gut reaction - which post would you like/comment on? Answer: A or B."""

    response = generate_response(prompt)
    return parse_ab_response(response)


def predict_from_logits(post_a: str, post_b: str) -> tuple[int, float]:
    """Predict which post wins based on A/B logits. Returns (prediction, confidence)."""
    prompt = f"""You're scrolling LinkedIn. You see these two posts. Which one makes you want to engage more?

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Trust your gut reaction - which post would you like/comment on? Answer: A or B."""

    a_logprob, b_logprob = get_ab_logits(prompt)

    # Convert to probabilities
    import math
    a_prob = math.exp(a_logprob)
    b_prob = math.exp(b_logprob)

    # Normalize
    total = a_prob + b_prob
    a_prob /= total
    b_prob /= total

    if a_prob > b_prob:
        return 0, a_prob
    else:
        return 1, b_prob


def predict_swapped_gen(post_a: str, post_b: str) -> int:
    """Predict with position swapping using generation."""
    # First direction
    pred1 = predict_from_generation(post_a, post_b)

    # Swapped
    pred2_raw = predict_from_generation(post_b, post_a)
    pred2 = 1 - pred2_raw if pred2_raw != -1 else -1  # Flip

    if pred1 == pred2:
        return pred1
    elif pred1 == -1:
        return pred2
    elif pred2 == -1:
        return pred1
    else:
        return -1  # Disagreement


def evaluate_generation(pairs, swapped: bool = False):
    """Evaluate on pairs using text generation."""
    correct = 0
    total = 0
    unparseable = 0

    for i, pair in enumerate(pairs):
        if swapped:
            pred = predict_swapped_gen(pair.post_a.text, pair.post_b.text)
        else:
            pred = predict_from_generation(pair.post_a.text, pair.post_b.text)

        if pred == -1:
            unparseable += 1
            continue

        if pred == pair.label:
            correct += 1
        total += 1

        if (i + 1) % 10 == 0:
            acc = correct / total if total > 0 else 0
            print(f"  [{i+1}/{len(pairs)}] {correct}/{total} = {acc*100:.1f}%")

    return correct / total if total > 0 else 0, unparseable


def run():
    print("=" * 70)
    print("Gemma 3 4B Local - Generation-based Prediction")
    print("=" * 70)

    load_model()

    train_pairs, test_pairs = get_pairs()
    test_subset = test_pairs[:100]
    print(f"\nTest set: {len(test_subset)} pairs")

    # Test generation
    print("\nTesting generation...")
    test_response = generate_response("Say hello")
    print(f"Response: '{test_response[:50]}...'")

    print("\n" + "-" * 60)
    print("Direct (generation-based)")
    print("-" * 60)
    acc, unparse = evaluate_generation(test_subset, swapped=False)
    print(f"Direct: {acc*100:.1f}% ({unparse} unparseable)")

    print("\n" + "-" * 60)
    print("Swapped (bias-corrected)")
    print("-" * 60)
    acc_swap, unparse_swap = evaluate_generation(test_subset, swapped=True)
    print(f"Swapped: {acc_swap*100:.1f}% ({unparse_swap} unparseable)")

    print(f"\nRandom baseline: 50.0%")


if __name__ == "__main__":
    random.seed(42)
    run()
