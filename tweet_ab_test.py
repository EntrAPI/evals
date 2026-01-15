#!/usr/bin/env python3
"""
Tweet A/B Testing Experiment

Tests whether Qwen 2.5 0.5B can predict which of two tweets will perform better
(more engagement: likes + retweets).

Uses the Tesla Twitter dataset from Hugging Face.
"""

import random
import re
from dataclasses import dataclass

import torch
from datasets import load_dataset
from langdetect import detect, LangDetectException
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Tweet:
    text: str
    likes: int
    retweets: int
    replies: int

    @property
    def engagement(self) -> int:
        return self.likes + self.retweets

    def __repr__(self):
        return f"Tweet(eng={self.engagement}, text={self.text[:50]}...)"


def is_english(text: str) -> bool:
    """Use langdetect for proper language detection."""
    # Remove URLs and mentions for cleaner detection
    clean = re.sub(r'https?://\S+', '', text)
    clean = re.sub(r'@\w+', '', clean).strip()
    if len(clean) < 20:
        return False
    try:
        return detect(clean) == 'en'
    except LangDetectException:
        return False


def load_tweets(min_high_engagement: int = 5) -> list[Tweet]:
    """Load tweets from the Tesla dataset with quality filters."""
    print("Loading dataset...")
    ds = load_dataset("hugginglearners/twitter-dataset-tesla", split="train")

    tweets = []
    for row in ds:
        text = row["tweet"]

        # Skip retweets and replies
        if row["retweet"] or text.startswith("RT"):
            continue

        # Skip tweets starting with @ (replies)
        if text.startswith("@"):
            continue

        # Skip very short tweets
        if len(text) < 30:
            continue

        # Skip non-English
        if not is_english(text):
            continue

        # Skip tweets that are mostly URLs
        url_pattern = r'https?://\S+'
        text_without_urls = re.sub(url_pattern, '', text).strip()
        if len(text_without_urls) < 20:
            continue

        tweets.append(Tweet(
            text=text,
            likes=int(row["nlikes"] or 0),
            retweets=int(row["nretweets"] or 0),
            replies=int(row["nreplies"] or 0),
        ))

    print(f"Loaded {len(tweets)} quality tweets (filtered from {len(ds)})")

    # Show likes distribution
    likes = [t.likes for t in tweets]
    print(f"Likes distribution:")
    print(f"  0: {sum(1 for l in likes if l == 0)}")
    print(f"  1-9: {sum(1 for l in likes if 1 <= l <= 9)}")
    print(f"  10-49: {sum(1 for l in likes if 10 <= l <= 49)}")
    print(f"  50-99: {sum(1 for l in likes if 50 <= l <= 99)}")
    print(f"  100+: {sum(1 for l in likes if l >= 100)}")

    return tweets


def create_pairs(tweets: list[Tweet], n_pairs: int = 100, min_high: int = 50) -> list[tuple[Tweet, Tweet, int]]:
    """
    Create pairs of tweets for A/B testing.
    Pairs high-engagement tweets (>=min_high) with zero engagement tweets.
    Returns list of (tweet_a, tweet_b, winner) where winner is 0 or 1.
    """
    # Filter by likes specifically (not total engagement)
    high_engagement = [t for t in tweets if t.likes >= min_high]
    low_engagement = [t for t in tweets if t.likes == 0 and t.retweets == 0]

    print(f"High engagement tweets (>={min_high} likes): {len(high_engagement)}")
    print(f"Zero engagement tweets: {len(low_engagement)}")

    random.shuffle(high_engagement)
    random.shuffle(low_engagement)

    pairs = []
    for i in range(min(n_pairs, len(high_engagement), len(low_engagement))):
        high = high_engagement[i]
        low = low_engagement[i]

        # Randomly assign which is A vs B to avoid position bias
        if random.random() < 0.5:
            pairs.append((high, low, 0))  # high is A, so winner is 0
        else:
            pairs.append((low, high, 1))  # high is B, so winner is 1

    print(f"Created {len(pairs)} tweet pairs for A/B testing")
    return pairs


class LLMPredictor:
    def __init__(self, model_id: str = MODEL_ID):
        print(f"\nLoading model: {model_id}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        print(f"Model loaded on {self.device}")

    def score_tweet(self, tweet: str) -> float | None:
        """Score a single tweet's engagement potential (1-10)."""
        messages = [
            {"role": "system", "content": "Rate tweet engagement potential from 1-10. Output only a number."},
            {"role": "user", "content": f"Rate this tweet's viral potential (1=low, 10=high):\n\n{tweet}\n\nScore:"}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Extract number from response
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        if numbers:
            return float(numbers[0])
        return None

    def predict(self, tweet_a: str, tweet_b: str) -> tuple[int | None, str]:
        """
        Score each tweet independently and compare.
        Returns (prediction, raw_response) where prediction is 0 for A, 1 for B, or None.
        """
        score_a = self.score_tweet(tweet_a)
        score_b = self.score_tweet(tweet_b)

        if score_a is None or score_b is None:
            return None, f"scores: {score_a}, {score_b}"

        if score_a > score_b:
            return 0, f"{score_a:.1f} vs {score_b:.1f}"
        elif score_b > score_a:
            return 1, f"{score_a:.1f} vs {score_b:.1f}"
        else:
            # Tie - random choice
            return random.choice([0, 1]), f"{score_a:.1f} vs {score_b:.1f} (tie)"


def run_experiment(n_pairs: int = 50, show_examples: int = 5):
    """Run the A/B testing experiment."""
    print(f"\n{'='*60}")
    print(f"Tweet A/B Testing Experiment")
    print(f"Model: {MODEL_ID}")
    print(f"{'='*60}\n")

    # Load data
    tweets = load_tweets()
    pairs = create_pairs(tweets, n_pairs=n_pairs, min_high=50)

    if not pairs:
        print("Error: No valid pairs created")
        return

    # Load model
    predictor = LLMPredictor()

    # Run predictions
    correct = 0
    total = 0
    errors = 0
    examples = []
    picked_first = 0
    picked_second = 0
    actual_first = 0
    actual_second = 0

    print(f"\nRunning {len(pairs)} predictions...\n")

    for i, (t1, t2, actual_winner) in enumerate(pairs):
        prediction, raw_response = predictor.predict(t1.text, t2.text)

        # Track position stats
        if actual_winner == 0:
            actual_first += 1
        else:
            actual_second += 1

        if prediction == 0:
            picked_first += 1
        elif prediction == 1:
            picked_second += 1

        if prediction is None:
            errors += 1
            status = "PARSE ERROR"
            is_correct = None
        elif prediction == actual_winner:
            correct += 1
            total += 1
            status = "✓ CORRECT"
            is_correct = True
        else:
            total += 1
            status = "✗ WRONG"
            is_correct = False

        # Store examples
        if len(examples) < show_examples:
            examples.append({
                "tweet_a": t1.text[:100],
                "tweet_b": t2.text[:100],
                "likes_a": t1.likes,
                "likes_b": t2.likes,
                "winner": "First" if actual_winner == 0 else "Second",
                "prediction": raw_response,
                "correct": is_correct,
            })

        # Show progress every 10
        if (i + 1) % 10 == 0 or i == len(pairs) - 1:
            acc = correct / total * 100 if total > 0 else 0
            print(f"[{i+1}/{len(pairs)}] {status} | Running accuracy: {acc:.1f}%")

    # Show examples
    print(f"\n{'='*60}")
    print("EXAMPLE PREDICTIONS")
    print(f"{'='*60}")
    for j, ex in enumerate(examples):
        print(f"\n--- Example {j+1} ---")
        print(f"Tweet A ({ex['likes_a']} likes): {ex['tweet_a']}...")
        print(f"Tweet B ({ex['likes_b']} likes): {ex['tweet_b']}...")
        print(f"Actual winner: {ex['winner']} | Model said: '{ex['prediction']}' | {'✓' if ex['correct'] else '✗' if ex['correct'] is False else '?'}")

    # Final results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {correct/total*100:.1f}%" if total > 0 else "N/A")
    print(f"Parse errors: {errors}")
    print(f"Random baseline: 50.0%")
    print(f"\nPosition bias check:")
    total_picks = picked_first + picked_second
    print(f"  Model picked First:  {picked_first} times ({picked_first/total_picks*100:.0f}%)" if total_picks > 0 else "N/A")
    print(f"  Model picked Second: {picked_second} times ({picked_second/total_picks*100:.0f}%)" if total_picks > 0 else "N/A")
    print(f"  Actual winner First:  {actual_first} times ({actual_first/len(pairs)*100:.0f}%)")
    print(f"  Actual winner Second: {actual_second} times ({actual_second/len(pairs)*100:.0f}%)")
    print(f"{'='*60}\n")

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0,
        "errors": errors,
        "model": MODEL_ID,
    }


if __name__ == "__main__":
    import sys
    random.seed(42)

    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_experiment(n_pairs=n_pairs)
