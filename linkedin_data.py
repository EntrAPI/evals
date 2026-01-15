#!/usr/bin/env python3
"""
LinkedIn dataset loader - A/B comparison for post engagement.

Dataset: LinkedIn Influencers Data
Source: https://www.kaggle.com/datasets/shreyasajal/linkedin-influencers-data
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_FILE = DATA_DIR / "influencers_data.csv"
PAIRS_FILE = DATA_DIR / "linkedin_pairs.json"


@dataclass
class Post:
    id: str
    text: str
    reactions: int
    comments: int
    author: str

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "reactions": self.reactions,
            "comments": self.comments,
            "author": self.author,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class PostPair:
    post_a: Post
    post_b: Post
    label: int  # 0 = A wins, 1 = B wins

    def to_dict(self):
        return {
            "post_a": self.post_a.to_dict(),
            "post_b": self.post_b.to_dict(),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            post_a=Post.from_dict(d["post_a"]),
            post_b=Post.from_dict(d["post_b"]),
            label=d["label"],
        )


def load_and_process() -> list[Post]:
    """Load and filter LinkedIn posts."""
    print(f"Loading {RAW_FILE}...")
    df = pd.read_csv(RAW_FILE, low_memory=False)

    posts = []
    for idx, row in df.iterrows():
        text = row.get("content")
        if pd.isna(text) or len(str(text)) < 50:
            continue

        reactions = row.get("reactions", 0)
        if pd.isna(reactions):
            continue

        comments = row.get("comments", 0)
        if pd.isna(comments):
            comments = 0

        posts.append(Post(
            id=str(idx),
            text=str(text),
            reactions=int(reactions),
            comments=int(comments),
            author=str(row.get("name", "Unknown")),
        ))

    print(f"Loaded {len(posts)} valid posts")
    return posts


def create_pairs(posts: list[Post], seed: int = 42, min_gap: int = 10,
                 train_ratio: float = 0.8, max_pairs: int = 2000) -> dict:
    """Create A/B comparison pairs."""
    random.seed(seed)

    # Sort by reactions
    sorted_posts = sorted(posts, key=lambda p: p.reactions)

    scores = {p.id: p.reactions for p in posts}

    print(f"\nReactions distribution:")
    print(f"  Min: {sorted_posts[0].reactions}")
    print(f"  Max: {sorted_posts[-1].reactions}")
    print(f"  Median: {sorted_posts[len(sorted_posts)//2].reactions}")

    # Generate pairs
    pairs = []
    used_ids = set()
    indices = list(range(len(sorted_posts)))
    random.shuffle(indices)

    for i in indices:
        if len(pairs) >= max_pairs:
            break

        p1 = sorted_posts[i]
        if p1.id in used_ids:
            continue

        candidates = [
            j for j in range(len(sorted_posts))
            if j != i
            and sorted_posts[j].id not in used_ids
            and abs(sorted_posts[j].reactions - p1.reactions) >= min_gap
        ]

        if not candidates:
            continue

        j = random.choice(candidates)
        p2 = sorted_posts[j]

        if random.random() < 0.5:
            post_a, post_b = p1, p2
        else:
            post_a, post_b = p2, p1

        label = 0 if post_a.reactions > post_b.reactions else 1

        pairs.append(PostPair(post_a=post_a, post_b=post_b, label=label))
        used_ids.add(p1.id)
        used_ids.add(p2.id)

    print(f"\nGenerated {len(pairs)} pairs (min gap: {min_gap})")

    random.shuffle(pairs)
    n_train = int(len(pairs) * train_ratio)
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    print(f"  Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    train_labels = [p.label for p in train_pairs]
    test_labels = [p.label for p in test_pairs]
    print(f"  Train balance: {sum(train_labels)/len(train_labels)*100:.1f}% B wins")
    print(f"  Test balance: {sum(test_labels)/len(test_labels)*100:.1f}% B wins")

    return {
        "train": [p.to_dict() for p in train_pairs],
        "test": [p.to_dict() for p in test_pairs],
        "seed": seed,
        "min_gap": min_gap,
    }


def save_pairs(data: dict):
    with open(PAIRS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {PAIRS_FILE}")


def load_pairs() -> dict | None:
    if not PAIRS_FILE.exists():
        return None
    with open(PAIRS_FILE) as f:
        return json.load(f)


def get_pairs() -> tuple[list[PostPair], list[PostPair]]:
    """Main API: Get train and test pairs."""
    data = load_pairs()
    if data is None:
        print("No pairs found. Run: python linkedin_data.py")
        import sys
        sys.exit(1)

    train = [PostPair.from_dict(p) for p in data["train"]]
    test = [PostPair.from_dict(p) for p in data["test"]]
    return train, test


def show_examples():
    data = load_pairs()
    if not data:
        print("No data. Run: python linkedin_data.py")
        return

    print("=" * 70)
    print("EXAMPLE PAIRS")
    print("=" * 70)

    for i, p in enumerate(data["train"][:3]):
        pair = PostPair.from_dict(p)
        print(f"\nPair {i+1} (Label: {'B' if pair.label else 'A'} wins)")
        print("-" * 70)
        print(f"  A [{pair.post_a.reactions:>5} reactions] {pair.post_a.text[:80]}...")
        print(f"  B [{pair.post_b.reactions:>5} reactions] {pair.post_b.text[:80]}...")


def main():
    import sys
    if "--examples" in sys.argv:
        show_examples()
        return

    if PAIRS_FILE.exists():
        print(f"Pairs exist at {PAIRS_FILE}")
        print("Use --examples to view, or delete to regenerate")
        return

    posts = load_and_process()
    pairs_data = create_pairs(posts)
    save_pairs(pairs_data)
    print("\nDone!")


if __name__ == "__main__":
    main()
