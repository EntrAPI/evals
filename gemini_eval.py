#!/usr/bin/env python3
"""
Evaluate Gemini 2.5 Flash Lite on tweet engagement prediction.
"""

import json
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "AIzaSyCWrrgydQ-_kaS92B3vvXrstJcZnPgiE20"
MODEL = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

PAIRS_FILE = "data/twitter_clean_no_replies.json"

def predict_engagement(tweet_a: str, tweet_b: str) -> int | None:
    """Ask Gemini which tweet will get more engagement. Returns 0 for A, 1 for B, None for error."""

    prompt = f"""You are predicting Twitter engagement. Both tweets are from the SAME user, posted at different times. Which tweet got more engagement (likes + retweets)?

Tweet A:
{tweet_a}

Tweet B:
{tweet_b}

Reply with ONLY "A" or "B" - nothing else."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 5,
        }
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        text = result["candidates"][0]["content"]["parts"][0]["text"].strip().upper()

        if "A" in text and "B" not in text:
            return 0
        elif "B" in text and "A" not in text:
            return 1
        else:
            print(f"  Ambiguous response: {text}")
            return None

    except Exception as e:
        print(f"  Error: {e}")
        return None


def run():
    print("=" * 70)
    print("Gemini 2.5 Flash Lite Engagement Prediction")
    print("=" * 70)

    # Load pairs
    print("\nLoading test pairs...")
    with open(PAIRS_FILE) as f:
        data = json.load(f)

    test_pairs = data['test']
    print(f"Total test pairs: {len(test_pairs):,}")

    # Sample for testing (API rate limits + cost)
    random.seed(42)
    sample_size = 500
    sample = random.sample(test_pairs, min(sample_size, len(test_pairs)))
    print(f"Evaluating on {len(sample)} samples...")

    # Evaluate
    correct = 0
    total = 0
    errors = 0

    results_by_ratio = {
        "1.5-2.0": {"correct": 0, "total": 0},
        "2.0-3.0": {"correct": 0, "total": 0},
        "3.0-5.0": {"correct": 0, "total": 0},
        "5.0+": {"correct": 0, "total": 0},
    }

    for i, pair in enumerate(sample):
        tweet_a = pair['tweet_a']['text']
        tweet_b = pair['tweet_b']['text']
        label = pair['label']  # 0 = A wins, 1 = B wins
        ratio = pair['ratio']

        pred = predict_engagement(tweet_a, tweet_b)

        if pred is not None:
            total += 1
            if pred == label:
                correct += 1

            # Track by ratio
            if ratio < 2.0:
                bucket = "1.5-2.0"
            elif ratio < 3.0:
                bucket = "2.0-3.0"
            elif ratio < 5.0:
                bucket = "3.0-5.0"
            else:
                bucket = "5.0+"

            results_by_ratio[bucket]["total"] += 1
            if pred == label:
                results_by_ratio[bucket]["correct"] += 1
        else:
            errors += 1

        if (i + 1) % 50 == 0:
            acc = correct / total * 100 if total > 0 else 0
            print(f"  Progress: {i+1}/{len(sample)} | Acc: {acc:.1f}% ({correct}/{total}) | Errors: {errors}")

        # Rate limiting
        time.sleep(0.1)

    # Final results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    acc = correct / total * 100 if total > 0 else 0
    print(f"\nOverall accuracy: {acc:.1f}% ({correct}/{total})")
    print(f"Errors/skipped: {errors}")

    print("\nAccuracy by engagement ratio:")
    for bucket, stats in results_by_ratio.items():
        if stats["total"] > 0:
            bucket_acc = stats["correct"] / stats["total"] * 100
            print(f"  {bucket}x: {bucket_acc:.1f}% ({stats['correct']}/{stats['total']})")

    print(f"\nRandom baseline: 50.0%")
    print(f"Improvement over random: +{acc - 50:.1f}pp")


if __name__ == "__main__":
    run()
