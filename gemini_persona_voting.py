#!/usr/bin/env python3
"""
Persona voting using Gemini API.

Test if LLMs can predict engagement via persona simulation.
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

MAX_WORKERS = 10  # Concurrent API calls


def call_gemini(prompt: str, temperature: float = 0.7, max_tokens: int = 10, max_retries: int = 5) -> str:
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
            resp = requests.post(API_URL, json=payload, timeout=30)
            if resp.status_code == 429:
                wait = 5 * (2 ** attempt) + random.random()
                print(f"  Rate limited, waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            # Handle response structure safely
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"].strip()
            return ""
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                wait = 2 ** attempt + random.random()
                time.sleep(wait)
                continue
            return ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            return ""

    return ""


def would_like(post: str, persona: str, temperature: float = 0.7) -> bool:
    """Ask if persona would like the post."""
    prompt = f"""You are {persona}

You're scrolling through LinkedIn and see this post:

"{post[:400]}"

Would you react to (like) this post? Answer only YES or NO."""

    response = call_gemini(prompt, temperature=temperature, max_tokens=4096)
    response_lower = response.lower()

    if "yes" in response_lower and "no" not in response_lower:
        return True
    elif "no" in response_lower:
        return False
    else:
        return False  # Default to no if ambiguous


def vote_on_pair(post_a: str, post_b: str, persona: str,
                 n_votes: int = 4, temperature: float = 0.7) -> tuple:
    """Have persona vote on both posts multiple times."""
    votes_a = sum(would_like(post_a, persona, temperature) for _ in range(n_votes))
    votes_b = sum(would_like(post_b, persona, temperature) for _ in range(n_votes))

    if votes_a > votes_b:
        pred = 0  # A wins
    elif votes_b > votes_a:
        pred = 1  # B wins
    else:
        pred = random.choice([0, 1])  # Tie-break

    return votes_a, votes_b, pred


def direct_compare(post_a: str, post_b: str, temperature: float = 0.0) -> int:
    """Direct A vs B comparison."""
    prompt = f"""Which LinkedIn post will get MORE engagement (likes, comments, shares)?

Post A: "{post_a[:350]}"

Post B: "{post_b[:350]}"

Reply with only "A" or "B"."""

    response = call_gemini(prompt, temperature=temperature, max_tokens=4096)
    response_upper = response.upper()

    if "A" in response_upper and "B" not in response_upper:
        return 0
    elif "B" in response_upper and "A" not in response_upper:
        return 1
    else:
        return -1  # Unparseable


def swapped_compare(post_a: str, post_b: str, temperature: float = 0.0) -> int:
    """Compare twice with swapped positions to cancel positional bias."""
    pred1 = direct_compare(post_a, post_b, temperature)
    time.sleep(0.1)  # Small delay between calls
    pred2_raw = direct_compare(post_b, post_a, temperature)

    # Convert: if raw says A (which is now B), that means B wins
    pred2 = 1 - pred2_raw if pred2_raw != -1 else -1

    if pred1 == pred2:
        return pred1
    elif pred1 == -1:
        return pred2
    elif pred2 == -1:
        return pred1
    else:
        return random.choice([0, 1])  # Disagreement


def evaluate_single_persona(args):
    """Evaluate a single pair with persona voting."""
    pair, persona, n_votes, temperature = args
    votes_a, votes_b, pred = vote_on_pair(
        pair.post_a.text, pair.post_b.text,
        persona, n_votes=n_votes, temperature=temperature
    )
    return pred == pair.label


def evaluate_persona(pairs, persona: str, n_votes: int = 4, temperature: float = 0.7):
    """Evaluate accuracy with persona voting (multi-threaded)."""
    args_list = [(pair, persona, n_votes, temperature) for pair in pairs]

    correct = 0
    total = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_single_persona, args): i
                   for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            if future.result():
                correct += 1
            total += 1

            if total % 10 == 0:
                print(f"  [{total}/{len(pairs)}] {correct}/{total} = {correct/total*100:.1f}%")

    return correct / total


def evaluate_single_direct(args):
    """Evaluate a single pair with direct comparison."""
    pair, swapped = args
    compare_fn = swapped_compare if swapped else direct_compare
    pred = compare_fn(pair.post_a.text, pair.post_b.text)
    return pred, pair.label


def evaluate_direct(pairs, swapped: bool = False):
    """Evaluate accuracy with direct comparison (multi-threaded)."""
    args_list = [(pair, swapped) for pair in pairs]

    correct = 0
    total = 0
    unparseable = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_single_direct, args): i
                   for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            pred, label = future.result()

            if pred == -1:
                unparseable += 1
                continue

            if pred == label:
                correct += 1
            total += 1

            if total % 10 == 0:
                acc = correct / total if total > 0 else 0
                print(f"  [{total}/{len(pairs)}] {correct}/{total} = {acc*100:.1f}%")

    acc = correct / total if total > 0 else 0
    return acc, unparseable


def run():
    print("=" * 70)
    print(f"{MODEL} - Persona Voting (Multi-threaded)")
    print("=" * 70)

    train_pairs, test_pairs = get_pairs()

    # Use larger subset now with better API key
    test_subset = test_pairs[:100]
    print(f"\nTest set: {len(test_subset)} pairs")

    # Check API works
    print("\nTesting API connection...")
    test_response = call_gemini("Say 'hello' and nothing else.", temperature=0, max_tokens=4096)
    if not test_response:
        print("API connection failed!")
        return
    print(f"API works: '{test_response}'")

    results = {}

    # Test 1: Direct comparison
    print("\n" + "-" * 70)
    print("Direct Comparison (A vs B)")
    print("-" * 70)
    acc, unparse = evaluate_direct(test_subset, swapped=False)
    results["direct"] = acc
    print(f"Direct: {acc*100:.1f}% (unparseable: {unparse})")

    # Test 2: Swapped comparison
    print("\n" + "-" * 70)
    print("Swapped Comparison (cancels positional bias)")
    print("-" * 70)
    acc, unparse = evaluate_direct(test_subset, swapped=True)
    results["swapped"] = acc
    print(f"Swapped: {acc*100:.1f}% (unparseable: {unparse})")

    # Test 3: Persona voting (just one persona to start)
    persona = "a professional who uses LinkedIn daily for networking"
    print("\n" + "-" * 70)
    print(f"Persona Voting: '{persona[:50]}...'")
    print("-" * 70)
    acc = evaluate_persona(test_subset, persona, n_votes=4, temperature=0.7)
    results["persona_voting"] = acc
    print(f"Accuracy: {acc*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:30s}: {acc*100:.1f}%")

    print(f"\nRandom baseline: 50.0%")


if __name__ == "__main__":
    random.seed(42)
    run()
