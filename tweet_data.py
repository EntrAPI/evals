#!/usr/bin/env python3
"""
Tweet dataset loader - handles downloading, filtering, and splitting.

Dataset: Twitter/X Posts Sample (1000+ posts)
Source: https://github.com/luminati-io/Twitter-X-dataset-samples

Task: A/B comparison - given two tweets, predict which gets more engagement.

Usage:
    python tweet_data.py              # Download and process data
    python tweet_data.py --stats      # Show dataset statistics
    python tweet_data.py --examples   # Show example pairs
"""

import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_FILE = DATA_DIR / "twitter_posts.csv"
PROCESSED_FILE = DATA_DIR / "tweets_processed.json"
PAIRS_FILE = DATA_DIR / "tweet_pairs.json"

DOWNLOAD_URL = "https://raw.githubusercontent.com/luminati-io/Twitter-X-dataset-samples/main/twitter-posts.csv"


@dataclass
class Tweet:
    id: str
    text: str
    likes: int
    retweets: int
    replies: int
    views: int
    lang: str

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "likes": self.likes,
            "retweets": self.retweets,
            "replies": self.replies,
            "views": self.views,
            "lang": self.lang,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def download_data():
    """Download the dataset from GitHub."""
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from {DOWNLOAD_URL}...")
    urllib.request.urlretrieve(DOWNLOAD_URL, RAW_FILE)
    print(f"Saved to {RAW_FILE}")
    return True


def load_raw_data() -> list[dict] | None:
    """Load raw CSV data from file."""
    if not RAW_FILE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Try to download
        try:
            download_data()
        except Exception as e:
            print(f"Download failed: {e}")
            print(f"\nManually download from: {DOWNLOAD_URL}")
            print(f"Save to: {RAW_FILE}")
            return None

    print(f"Loading raw data from {RAW_FILE}...")
    data = []
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    print(f"Loaded {len(data)} raw tweets")
    return data


def process_tweets(raw_data: list[dict]) -> list[Tweet]:
    """Filter and process raw tweets."""
    tweets = []
    skipped = {"short": 0, "no_text": 0, "duplicate": 0}
    seen_ids = set()

    for row in raw_data:
        # Get text (column is 'description' in this dataset)
        text = row.get("description", "").strip()
        if not text:
            skipped["no_text"] += 1
            continue
        if len(text) < 30:
            skipped["short"] += 1
            continue

        # Get engagement metrics
        likes = int(row.get("likes", 0) or 0)
        retweets = int(row.get("reposts", 0) or 0)
        replies = int(row.get("replies", 0) or 0)
        views = int(row.get("views", 0) or 0)

        tweet_id = row.get("id", "")
        if tweet_id in seen_ids:
            skipped["duplicate"] += 1
            continue
        seen_ids.add(tweet_id)

        tweets.append(Tweet(
            id=tweet_id,
            text=text,
            likes=likes,
            retweets=retweets,
            replies=replies,
            views=views,
            lang="en",  # Dataset doesn't have lang, assume English
        ))

    print(f"\nProcessed {len(tweets)} valid tweets")
    print(f"Skipped: {skipped}")
    return tweets


@dataclass
class TweetPair:
    tweet_a: Tweet
    tweet_b: Tweet
    label: int  # 0 = A wins, 1 = B wins

    def to_dict(self):
        return {
            "tweet_a": self.tweet_a.to_dict(),
            "tweet_b": self.tweet_b.to_dict(),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            tweet_a=Tweet.from_dict(d["tweet_a"]),
            tweet_b=Tweet.from_dict(d["tweet_b"]),
            label=d["label"],
        )


def engagement_score(tweet: Tweet) -> float:
    """Combined engagement score."""
    # Weighted combination: likes matter most, then retweets, then replies
    # Normalize by views if available to get engagement rate
    if tweet.views > 0:
        return (tweet.likes + 0.5 * tweet.retweets + 0.2 * tweet.replies) / tweet.views * 1000
    return tweet.likes + 0.5 * tweet.retweets + 0.2 * tweet.replies


def create_pairs(tweets: list[Tweet], seed: int = 42, min_gap: float = 5,
                 train_ratio: float = 0.8, max_pairs: int = 5000, use_engagement_rate: bool = False) -> dict:
    """
    Create A/B comparison pairs.

    Each pair has two tweets with different engagement levels.
    Label: 0 = tweet_a has more engagement, 1 = tweet_b has more engagement.

    Args:
        tweets: List of processed tweets
        seed: Random seed for reproducibility
        min_gap: Minimum difference in engagement score
        train_ratio: Fraction of pairs for training
        max_pairs: Maximum number of pairs to generate
        use_engagement_rate: If True, use engagement/views rate; else use likes

    Returns dict with train_pairs and test_pairs.
    """
    random.seed(seed)

    # Score each tweet
    if use_engagement_rate:
        scores = {t.id: engagement_score(t) for t in tweets}
        score_name = "engagement rate"
    else:
        scores = {t.id: t.likes for t in tweets}
        score_name = "likes"

    # Sort tweets by score
    sorted_tweets = sorted(tweets, key=lambda t: scores[t.id])

    score_values = [scores[t.id] for t in sorted_tweets]
    print(f"\n{score_name} distribution:")
    print(f"  Min: {score_values[0]:.2f}")
    print(f"  Max: {score_values[-1]:.2f}")
    print(f"  Median: {score_values[len(score_values)//2]:.2f}")

    # Generate pairs with sufficient gap
    pairs = []
    used_ids = set()

    # Sample pairs randomly
    indices = list(range(len(sorted_tweets)))
    random.shuffle(indices)

    for i in indices:
        if len(pairs) >= max_pairs:
            break

        t1 = sorted_tweets[i]
        if t1.id in used_ids:
            continue

        score1 = scores[t1.id]

        # Find a tweet with sufficient gap
        candidates = [
            j for j in range(len(sorted_tweets))
            if j != i
            and sorted_tweets[j].id not in used_ids
            and abs(scores[sorted_tweets[j].id] - score1) >= min_gap
        ]

        if not candidates:
            continue

        j = random.choice(candidates)
        t2 = sorted_tweets[j]

        # Randomly assign to A or B position
        if random.random() < 0.5:
            tweet_a, tweet_b = t1, t2
        else:
            tweet_a, tweet_b = t2, t1

        # Label: which one has higher score
        label = 0 if scores[tweet_a.id] > scores[tweet_b.id] else 1

        pairs.append(TweetPair(tweet_a=tweet_a, tweet_b=tweet_b, label=label))
        used_ids.add(t1.id)
        used_ids.add(t2.id)

    print(f"\nGenerated {len(pairs)} pairs (min gap: {min_gap})")

    # Shuffle and split
    random.shuffle(pairs)
    n_train = int(len(pairs) * train_ratio)
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Test: {len(test_pairs)} pairs")

    # Verify balance
    train_labels = [p.label for p in train_pairs]
    test_labels = [p.label for p in test_pairs]
    print(f"\nLabel balance:")
    print(f"  Train: {sum(train_labels)}/{len(train_labels)} = {sum(train_labels)/len(train_labels)*100:.1f}% B wins")
    print(f"  Test: {sum(test_labels)}/{len(test_labels)} = {sum(test_labels)/len(test_labels)*100:.1f}% B wins")

    return {
        "train": [p.to_dict() for p in train_pairs],
        "test": [p.to_dict() for p in test_pairs],
        "seed": seed,
        "min_gap": min_gap,
    }


def save_processed(tweets: list[Tweet]):
    """Save processed tweets to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = [t.to_dict() for t in tweets]
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved processed tweets to {PROCESSED_FILE}")


def save_pairs(pairs_data: dict):
    """Save pairs to disk."""
    with open(PAIRS_FILE, "w", encoding="utf-8") as f:
        json.dump(pairs_data, f, indent=2)
    print(f"Saved pairs to {PAIRS_FILE}")


def load_pairs() -> dict | None:
    """Load pairs from disk."""
    if not PAIRS_FILE.exists():
        return None
    with open(PAIRS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pairs() -> tuple[list[TweetPair], list[TweetPair]]:
    """
    Main API: Get train and test pairs.

    Returns:
        train_pairs, test_pairs
    """
    data = load_pairs()
    if data is None:
        print("No pairs found. Run: python tweet_data.py")
        sys.exit(1)

    train_pairs = [TweetPair.from_dict(p) for p in data["train"]]
    test_pairs = [TweetPair.from_dict(p) for p in data["test"]]

    return train_pairs, test_pairs


def show_stats():
    """Show dataset statistics."""
    data = load_pairs()
    if data is None:
        print("No pairs found. Run: python tweet_data.py")
        return

    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    train_pairs = [TweetPair.from_dict(p) for p in data["train"]]
    test_pairs = [TweetPair.from_dict(p) for p in data["test"]]

    print(f"\nSeed: {data['seed']}")
    print(f"Min gap: {data['min_gap']}")

    print(f"\nTrain: {len(train_pairs)} pairs")
    print(f"Test: {len(test_pairs)} pairs")

    # Label balance
    train_labels = [p.label for p in train_pairs]
    test_labels = [p.label for p in test_pairs]
    print(f"\nLabel balance (should be ~50%):")
    print(f"  Train: {sum(train_labels)/len(train_labels)*100:.1f}% B wins")
    print(f"  Test: {sum(test_labels)/len(test_labels)*100:.1f}% B wins")

    # Like gap distribution
    train_gaps = [abs(p.tweet_a.likes - p.tweet_b.likes) for p in train_pairs]
    test_gaps = [abs(p.tweet_a.likes - p.tweet_b.likes) for p in test_pairs]
    print(f"\nLike gap distribution:")
    print(f"  Train: min={min(train_gaps)}, max={max(train_gaps)}, mean={sum(train_gaps)/len(train_gaps):.1f}")
    print(f"  Test: min={min(test_gaps)}, max={max(test_gaps)}, mean={sum(test_gaps)/len(test_gaps):.1f}")


def show_examples():
    """Show example pairs."""
    data = load_pairs()
    if data is None:
        print("No pairs found. Run: python tweet_data.py")
        return

    print("=" * 70)
    print("EXAMPLE PAIRS")
    print("=" * 70)

    train_pairs = [TweetPair.from_dict(p) for p in data["train"][:5]]

    for i, pair in enumerate(train_pairs):
        print(f"\nPair {i+1} (Label: {'B' if pair.label else 'A'} wins)")
        print("-" * 70)

        text_a = pair.tweet_a.text[:80] + "..." if len(pair.tweet_a.text) > 80 else pair.tweet_a.text
        text_b = pair.tweet_b.text[:80] + "..." if len(pair.tweet_b.text) > 80 else pair.tweet_b.text

        winner_a = " <-- WINNER" if pair.label == 0 else ""
        winner_b = " <-- WINNER" if pair.label == 1 else ""

        print(f"  A [{pair.tweet_a.likes:>4} likes]{winner_a}")
        print(f"    {text_a}")
        print(f"  B [{pair.tweet_b.likes:>4} likes]{winner_b}")
        print(f"    {text_b}")


def main():
    if "--stats" in sys.argv:
        show_stats()
        return

    if "--examples" in sys.argv:
        show_examples()
        return

    # Check for existing pairs
    if PAIRS_FILE.exists():
        print(f"Pairs already exist at {PAIRS_FILE}")
        print("Use --stats to view statistics, --examples to view samples")
        print("Delete the file to regenerate pairs")
        return

    # Load raw data (will auto-download if needed)
    raw_data = load_raw_data()
    if raw_data is None:
        return

    # Process
    tweets = process_tweets(raw_data)
    save_processed(tweets)

    # Create and save pairs
    pairs_data = create_pairs(tweets)
    save_pairs(pairs_data)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nData saved to {DATA_DIR}/")
    print("Run with --stats or --examples to inspect the data")


if __name__ == "__main__":
    main()
