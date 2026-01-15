#!/usr/bin/env python3
"""
Prompt engineering for engagement prediction.

Test many different prompt framings with chain-of-thought reasoning.
Multi-threaded for speed.
"""

import random
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from linkedin_data import get_pairs

API_KEY = "AIzaSyCWrrgydQ-_kaS92B3vvXrstJcZnPgiE20"
MODEL = "gemma-3-4b-it"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

MAX_WORKERS = 20


def call_gemini(prompt: str, temperature: float = 0.0, max_tokens: int = 1024, max_retries: int = 8) -> str:
    """Call Gemini API with exponential backoff."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = 2 * (2 ** attempt) + random.random()
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"].strip()
            return ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            return ""
    return ""


def parse_ab_response(response: str) -> int:
    """Parse A/B response. Returns 0 for A, 1 for B, -1 for unparseable."""
    response = response.strip()

    # Check last line first (after reasoning)
    lines = response.strip().split('\n')
    last_line = lines[-1].strip().upper() if lines else ""

    # Look for clear A or B at the end
    if last_line in ["A", "A.", "**A**", "POST A", "ANSWER: A", "FINAL ANSWER: A"]:
        return 0
    if last_line in ["B", "B.", "**B**", "POST B", "ANSWER: B", "FINAL ANSWER: B"]:
        return 1

    # Check for A or B anywhere in last few lines
    check_text = ' '.join(lines[-3:]).upper() if len(lines) >= 3 else response.upper()

    # Strong signals
    if "FINAL ANSWER: A" in check_text or "ANSWER IS A" in check_text or "CHOOSE A" in check_text:
        return 0
    if "FINAL ANSWER: B" in check_text or "ANSWER IS B" in check_text or "CHOOSE B" in check_text:
        return 1
    if "POST A WILL" in check_text and "POST B WILL" not in check_text:
        return 0
    if "POST B WILL" in check_text and "POST A WILL" not in check_text:
        return 1

    # Count mentions in conclusion area
    a_count = check_text.count(" A ") + check_text.count(" A.") + check_text.count("(A)")
    b_count = check_text.count(" B ") + check_text.count(" B.") + check_text.count("(B)")

    if a_count > b_count:
        return 0
    if b_count > a_count:
        return 1

    # Last resort: check whole response
    full_upper = response.upper()
    if "A IS BETTER" in full_upper or "A WILL GET MORE" in full_upper:
        return 0
    if "B IS BETTER" in full_upper or "B WILL GET MORE" in full_upper:
        return 1

    return -1


# ============== PROMPT STRATEGIES ==============

def prompt_direct(post_a: str, post_b: str) -> str:
    return f"""Which LinkedIn post will get MORE engagement (likes, comments, shares)?

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Reply with only "A" or "B"."""


def prompt_cot_simple(post_a: str, post_b: str) -> str:
    return f"""Which LinkedIn post will get MORE engagement (likes, comments, shares)?

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Think step by step about what makes content engaging on LinkedIn, then give your final answer as just "A" or "B"."""


def prompt_cot_criteria(post_a: str, post_b: str) -> str:
    return f"""Analyze which LinkedIn post will get MORE engagement.

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Consider these factors:
1. Emotional hook - does it grab attention?
2. Relatability - can professionals connect with it?
3. Value - does it teach something or provide insight?
4. Shareability - would someone want to share this?
5. Discussion potential - does it invite comments?

Analyze each post on these criteria, then conclude with your final answer: "A" or "B"."""


def prompt_algorithm(post_a: str, post_b: str) -> str:
    return f"""You are LinkedIn's engagement prediction algorithm. Your job is to predict which content will perform better.

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Based on patterns in viral LinkedIn content, which post will get more engagement? Explain your reasoning, then state your prediction: A or B."""


def prompt_marketer(post_a: str, post_b: str) -> str:
    return f"""You are a social media marketing expert with 10 years of experience optimizing LinkedIn content.

A client shows you two posts and asks which will perform better:

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Based on your expertise, analyze both posts and predict which will get more engagement. End with your recommendation: A or B."""


def prompt_viral(post_a: str, post_b: str) -> str:
    return f"""Which of these posts is more likely to go VIRAL on LinkedIn?

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Consider: hook strength, emotional resonance, professional relevance, and shareability.
Think through your analysis, then answer: A or B."""


def prompt_audience(post_a: str, post_b: str) -> str:
    return f"""Imagine showing these two LinkedIn posts to 1000 professionals.

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Which post would get more total reactions (likes, comments, shares) from this audience?

Reason through what would resonate with professionals, then give your answer: A or B."""


def prompt_specific_metrics(post_a: str, post_b: str) -> str:
    return f"""Predict engagement for these LinkedIn posts.

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

For each post, estimate:
- Likes (1-100 scale)
- Comments (1-50 scale)
- Shares (1-20 scale)

Then calculate total engagement and determine which post wins.
Final answer: A or B."""


def prompt_first_impression(post_a: str, post_b: str) -> str:
    return f"""You're scrolling LinkedIn. You see these two posts. Which one makes you want to engage more?

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Trust your gut reaction - which post would you like/comment on? Answer: A or B."""


def prompt_contrast(post_a: str, post_b: str) -> str:
    return f"""Compare and contrast these LinkedIn posts for engagement potential.

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

STRENGTHS of Post A:
WEAKNESSES of Post A:
STRENGTHS of Post B:
WEAKNESSES of Post B:

WINNER (more engagement): A or B"""


def prompt_psychology(post_a: str, post_b: str) -> str:
    return f"""Using psychology of social media engagement, analyze these posts:

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Consider:
- Social proof triggers
- Emotional valence (positive/negative/surprising)
- Identity reinforcement (does it make the sharer look good?)
- Cognitive ease (easy to process?)
- Call to action (implicit or explicit?)

Which post has stronger psychological engagement drivers? Answer: A or B."""


def prompt_betting(post_a: str, post_b: str) -> str:
    return f"""If you had to bet $100 on which LinkedIn post gets more engagement:

Post A: "{post_a[:400]}"

Post B: "{post_b[:400]}"

Which would you bet on and why? Make your choice: A or B."""


def prompt_ab_test(post_a: str, post_b: str) -> str:
    return f"""You're running an A/B test on LinkedIn. These are the two variants:

Variant A: "{post_a[:400]}"

Variant B: "{post_b[:400]}"

Based on your understanding of what drives LinkedIn engagement, predict which variant will win the test.

Analyze key differences, then declare the winner: A or B."""


def prompt_detailed_reasoning(post_a: str, post_b: str) -> str:
    return f"""I need you to carefully analyze which LinkedIn post will get more engagement.

=== POST A ===
{post_a[:500]}

=== POST B ===
{post_b[:500]}

Please think through this systematically:

1. FIRST IMPRESSIONS
   - What's your initial reaction to each post?
   - Which one grabs attention faster?

2. CONTENT ANALYSIS
   - What's the main message of each?
   - Is it valuable/insightful?
   - Is it relatable to professionals?

3. ENGAGEMENT TRIGGERS
   - Does it provoke emotion?
   - Does it invite discussion?
   - Is it shareable?

4. LINKEDIN-SPECIFIC FACTORS
   - Does it fit LinkedIn's professional context?
   - Would engaging with it make someone look good?

5. PREDICTION
   - Weigh the factors above
   - Make your prediction

FINAL ANSWER (A or B):"""


# ============== EVALUATION ==============

STRATEGIES = {
    "direct": prompt_direct,
    "cot_simple": prompt_cot_simple,
    "cot_criteria": prompt_cot_criteria,
    "algorithm": prompt_algorithm,
    "marketer": prompt_marketer,
    "viral": prompt_viral,
    "audience": prompt_audience,
    "metrics": prompt_specific_metrics,
    "first_impression": prompt_first_impression,
    "contrast": prompt_contrast,
    "psychology": prompt_psychology,
    "betting": prompt_betting,
    "ab_test": prompt_ab_test,
    "detailed": prompt_detailed_reasoning,
}


def evaluate_single(args):
    """Evaluate single pair with a strategy."""
    pair, strategy_fn, temperature = args
    prompt = strategy_fn(pair.post_a.text, pair.post_b.text)

    response = call_gemini(prompt, temperature=temperature, max_tokens=1024)
    pred = parse_ab_response(response)

    return pred, pair.label, response


def evaluate_strategy(pairs, strategy_name: str, strategy_fn, temperature: float = 0.0):
    """Evaluate a strategy on all pairs."""
    args_list = [(pair, strategy_fn, temperature) for pair in pairs]

    correct = 0
    total = 0
    unparseable = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_single, args): i
                   for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            pred, label, response = future.result()

            if pred == -1:
                unparseable += 1
                continue

            if pred == label:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    return acc, total, unparseable


def evaluate_swapped_single(args):
    """Evaluate single pair with swapped positions."""
    pair, strategy_fn, temperature = args

    # First direction
    prompt1 = strategy_fn(pair.post_a.text, pair.post_b.text)
    response1 = call_gemini(prompt1, temperature=temperature, max_tokens=1024)
    pred1 = parse_ab_response(response1)

    # Swapped direction
    prompt2 = strategy_fn(pair.post_b.text, pair.post_a.text)
    response2 = call_gemini(prompt2, temperature=temperature, max_tokens=1024)
    pred2_raw = parse_ab_response(response2)

    # Flip pred2 (if it said A, that means B wins in original order)
    pred2 = 1 - pred2_raw if pred2_raw != -1 else -1

    # Combine predictions
    if pred1 == pred2:
        final_pred = pred1
    elif pred1 == -1:
        final_pred = pred2
    elif pred2 == -1:
        final_pred = pred1
    else:
        # Disagreement - return -1 or random
        final_pred = -1

    return final_pred, pair.label


def evaluate_swapped(pairs, strategy_name: str, strategy_fn, temperature: float = 0.0):
    """Evaluate a strategy with swapped positions to cancel bias."""
    args_list = [(pair, strategy_fn, temperature) for pair in pairs]

    correct = 0
    total = 0
    unparseable = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_swapped_single, args): i
                   for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            pred, label = future.result()

            if pred == -1:
                unparseable += 1
                continue

            if pred == label:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    return acc, total, unparseable


def run():
    print("=" * 70)
    print(f"{MODEL} - First Impression Test")
    print("=" * 70)

    train_pairs, test_pairs = get_pairs()
    test_subset = test_pairs[:100]
    print(f"\nTest set: {len(test_subset)} pairs")

    # Test API
    print("\nTesting API connection...")
    test_response = call_gemini("Say 'hello'", temperature=0, max_tokens=10)
    if not test_response:
        print("API connection failed!")
        return
    print(f"API works: '{test_response}'")

    # Just test first_impression
    fn = prompt_first_impression
    name = "first_impression"

    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print("=" * 60)

    # Direct evaluation
    acc, total, unparse = evaluate_strategy(test_subset, name, fn, temperature=0.0)
    print(f"  Direct:  {acc*100:.1f}% ({total} valid, {unparse} unparseable)")

    # Swapped evaluation
    acc_swap, total_swap, unparse_swap = evaluate_swapped(test_subset, name, fn, temperature=0.0)
    print(f"  Swapped: {acc_swap*100:.1f}% ({total_swap} valid, {unparse_swap} unparseable)")

    print(f"\nRandom baseline: 50.0%")


if __name__ == "__main__":
    random.seed(42)
    run()
