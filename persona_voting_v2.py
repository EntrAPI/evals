#!/usr/bin/env python3
"""
Persona voting v2 - trying different approaches to make it work.

Variations:
1. Rating scale (1-10) instead of YES/NO
2. Direct comparison with position swapping to cancel bias
3. Few-shot examples
4. Better engagement-focused prompting
"""

import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from linkedin_data import get_pairs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


class EngagementPredictor:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        print(f"\nLoading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(DEVICE)
        self.model.eval()
        print("  Ready.")

    def generate(self, prompt: str, max_tokens: int = 20, temperature: float = 0.3) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response

    def rate_engagement(self, post: str, temperature: float = 0.3) -> float:
        """Rate a post's engagement potential on a 1-10 scale."""
        prompt = f"""Rate how much engagement (likes, comments, shares) this LinkedIn post will get.

Post: "{post[:300]}"

Rate from 1-10 where:
1 = Very low engagement (boring, generic)
5 = Average engagement
10 = Viral potential (highly engaging, shareable)

Reply with ONLY a number from 1 to 10."""

        response = self.generate(prompt, max_tokens=5, temperature=temperature)

        # Parse the number
        for word in response.split():
            try:
                num = float(word.replace(',', '.'))
                if 1 <= num <= 10:
                    return num
            except:
                continue
        return 5.0  # Default to middle

    def compare_direct(self, post_a: str, post_b: str, temperature: float = 0.3) -> int:
        """Direct comparison - which post will get more engagement?"""
        prompt = f"""Which LinkedIn post will get MORE engagement (likes, comments, shares)?

Post 1: "{post_a[:250]}"

Post 2: "{post_b[:250]}"

Consider: emotional appeal, value provided, shareability, relatability.

Reply with ONLY "1" or "2"."""

        response = self.generate(prompt, max_tokens=5, temperature=temperature)

        if "1" in response and "2" not in response:
            return 0
        elif "2" in response and "1" not in response:
            return 1
        return -1  # Unparseable

    def compare_swapped(self, post_a: str, post_b: str, temperature: float = 0.3) -> int:
        """Compare twice with swapped positions, combine results."""
        # First comparison: A=1, B=2
        pred1 = self.compare_direct(post_a, post_b, temperature)

        # Second comparison: A=2, B=1 (swapped)
        pred2_raw = self.compare_direct(post_b, post_a, temperature)
        # Convert: if raw says 1 (which is now B), that means B wins
        pred2 = 1 - pred2_raw if pred2_raw != -1 else -1

        # Combine
        if pred1 == pred2:
            return pred1
        elif pred1 == -1:
            return pred2
        elif pred2 == -1:
            return pred1
        else:
            # Disagreement - return -1 or random
            return random.choice([0, 1])

    def rate_with_criteria(self, post: str, temperature: float = 0.3) -> float:
        """Rate with specific engagement criteria."""
        prompt = f"""Analyze this LinkedIn post for engagement potential.

Post: "{post[:300]}"

Rate each criterion 1-10:
1. Emotional resonance (does it evoke feelings?):
2. Practical value (useful tips/insights?):
3. Shareability (would people share this?):
4. Discussion potential (invites comments?):

Then give an OVERALL score 1-10.

End your response with "OVERALL: X" where X is 1-10."""

        response = self.generate(prompt, max_tokens=150, temperature=temperature)

        # Parse overall score
        if "overall:" in response.lower():
            after_overall = response.lower().split("overall:")[-1]
            for word in after_overall.split():
                try:
                    num = float(word.replace(',', '.'))
                    if 1 <= num <= 10:
                        return num
                except:
                    continue
        return 5.0

    def cleanup(self):
        del self.model
        del self.tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def get_pair_texts(pair):
    if hasattr(pair, 'tweet_a'):
        return pair.tweet_a.text, pair.tweet_b.text
    else:
        return pair.post_a.text, pair.post_b.text


def test_rating_approach(predictor, pairs, n_samples=4):
    """Test rating each post and comparing scores."""
    print(f"\n--- Rating Approach ({n_samples} samples per post) ---")
    correct = 0
    total = 0

    for i, pair in enumerate(pairs):
        text_a, text_b = get_pair_texts(pair)

        # Get multiple ratings and average
        scores_a = [predictor.rate_engagement(text_a, temperature=0.5) for _ in range(n_samples)]
        scores_b = [predictor.rate_engagement(text_b, temperature=0.5) for _ in range(n_samples)]

        avg_a = sum(scores_a) / len(scores_a)
        avg_b = sum(scores_b) / len(scores_b)

        pred = 0 if avg_a > avg_b else 1

        if pred == pair.label:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pairs)}] {correct}/{total} = {correct/total*100:.1f}%")

    acc = correct / total
    print(f"Rating accuracy: {acc*100:.1f}%")
    return acc


def test_direct_comparison(predictor, pairs):
    """Test direct A vs B comparison."""
    print(f"\n--- Direct Comparison ---")
    correct = 0
    total = 0
    unparseable = 0

    for i, pair in enumerate(pairs):
        text_a, text_b = get_pair_texts(pair)
        pred = predictor.compare_direct(text_a, text_b, temperature=0.0)

        if pred == -1:
            unparseable += 1
            continue

        if pred == pair.label:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pairs)}] {correct}/{total} = {correct/total*100:.1f}%")

    acc = correct / total if total > 0 else 0
    print(f"Direct comparison: {acc*100:.1f}% (unparseable: {unparseable})")
    return acc


def test_swapped_comparison(predictor, pairs):
    """Test comparison with position swapping to cancel bias."""
    print(f"\n--- Swapped Comparison (cancels positional bias) ---")
    correct = 0
    total = 0

    for i, pair in enumerate(pairs):
        text_a, text_b = get_pair_texts(pair)
        pred = predictor.compare_swapped(text_a, text_b, temperature=0.0)

        if pred == pair.label:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pairs)}] {correct}/{total} = {correct/total*100:.1f}%")

    acc = correct / total
    print(f"Swapped comparison: {acc*100:.1f}%")
    return acc


def test_criteria_rating(predictor, pairs, n_samples=2):
    """Test rating with specific criteria."""
    print(f"\n--- Criteria-Based Rating ---")
    correct = 0
    total = 0

    for i, pair in enumerate(pairs):
        text_a, text_b = get_pair_texts(pair)

        scores_a = [predictor.rate_with_criteria(text_a, temperature=0.3) for _ in range(n_samples)]
        scores_b = [predictor.rate_with_criteria(text_b, temperature=0.3) for _ in range(n_samples)]

        avg_a = sum(scores_a) / len(scores_a)
        avg_b = sum(scores_b) / len(scores_b)

        pred = 0 if avg_a > avg_b else 1

        if pred == pair.label:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pairs)}] {correct}/{total} = {correct/total*100:.1f}%")

    acc = correct / total
    print(f"Criteria rating: {acc*100:.1f}%")
    return acc


def run():
    print("=" * 70)
    print("Engagement Prediction - Approach Variations")
    print("=" * 70)

    train_pairs, test_pairs = get_pairs()

    # Use subset for faster testing
    test_subset = test_pairs[:100]
    print(f"\nTest set: {len(test_subset)} pairs")

    predictor = EngagementPredictor()

    results = {}

    # Test 1: Simple rating
    results["rating_4x"] = test_rating_approach(predictor, test_subset, n_samples=4)
    torch.cuda.empty_cache()

    # Test 2: Direct comparison
    results["direct"] = test_direct_comparison(predictor, test_subset)
    torch.cuda.empty_cache()

    # Test 3: Swapped comparison
    results["swapped"] = test_swapped_comparison(predictor, test_subset)
    torch.cuda.empty_cache()

    # Test 4: Criteria-based rating
    results["criteria_2x"] = test_criteria_rating(predictor, test_subset, n_samples=2)
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}: {acc*100:.1f}%")

    print(f"\nRandom baseline: 50.0%")

    predictor.cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    run()
