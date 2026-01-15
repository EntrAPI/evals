#!/usr/bin/env python3
"""
Compare different pooling strategies for tweet embeddings.
"""

import random
import re
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from datasets import load_dataset
from langdetect import detect, LangDetectException
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


@dataclass
class Tweet:
    text: str
    likes: int


def is_english(text: str) -> bool:
    clean = re.sub(r'https?://\S+', '', text)
    clean = re.sub(r'@\w+', '', clean).strip()
    if len(clean) < 20:
        return False
    try:
        return detect(clean) == 'en'
    except LangDetectException:
        return False


def load_tweets() -> list[Tweet]:
    print("Loading dataset...")
    ds = load_dataset("hugginglearners/twitter-dataset-tesla", split="train")

    tweets = []
    for row in ds:
        text = row["tweet"]
        if row["retweet"] or text.startswith("RT") or text.startswith("@"):
            continue
        if len(text) < 30:
            continue
        if not is_english(text):
            continue
        text_without_urls = re.sub(r'https?://\S+', '', text).strip()
        if len(text_without_urls) < 20:
            continue

        tweets.append(Tweet(text=text, likes=int(row["nlikes"] or 0)))

    print(f"Loaded {len(tweets)} tweets")
    return tweets


# ============================================================
# Pooling Functions
# ============================================================

def mean_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    """Mean over all tokens."""
    return hidden_states.mean(dim=0)


def max_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    """Max over all tokens."""
    return hidden_states.max(dim=0).values


def min_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    """Min over all tokens."""
    return hidden_states.min(dim=0).values


def first_token(hidden_states: torch.Tensor) -> torch.Tensor:
    """First token only."""
    return hidden_states[0, :]


def last_token(hidden_states: torch.Tensor) -> torch.Tensor:
    """Last token only."""
    return hidden_states[-1, :]


def mean_max_concat(hidden_states: torch.Tensor) -> torch.Tensor:
    """Concatenate mean and max pooling."""
    return torch.cat([hidden_states.mean(dim=0), hidden_states.max(dim=0).values])


def mean_std_concat(hidden_states: torch.Tensor) -> torch.Tensor:
    """Concatenate mean and std."""
    return torch.cat([hidden_states.mean(dim=0), hidden_states.std(dim=0)])


def mean_min_max_concat(hidden_states: torch.Tensor) -> torch.Tensor:
    """Concatenate mean, min, and max."""
    return torch.cat([
        hidden_states.mean(dim=0),
        hidden_states.min(dim=0).values,
        hidden_states.max(dim=0).values
    ])


def weighted_mean(hidden_states: torch.Tensor) -> torch.Tensor:
    """Linearly weighted mean (later tokens weighted more)."""
    seq_len = hidden_states.shape[0]
    weights = torch.arange(1, seq_len + 1, device=hidden_states.device, dtype=hidden_states.dtype)
    weights = weights / weights.sum()
    return (hidden_states * weights.unsqueeze(1)).sum(dim=0)


def attention_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    """Self-attention pooling (query = mean, keys = all tokens)."""
    query = hidden_states.mean(dim=0, keepdim=True)  # [1, hidden]
    scores = torch.matmul(query, hidden_states.T)  # [1, seq_len]
    weights = torch.softmax(scores / (hidden_states.shape[1] ** 0.5), dim=-1)  # [1, seq_len]
    return (weights @ hidden_states).squeeze(0)  # [hidden]


POOLING_FUNCTIONS = {
    "mean": mean_pool,
    "max": max_pool,
    "min": min_pool,
    "first_token": first_token,
    "last_token": last_token,
    "mean+max": mean_max_concat,
    "mean+std": mean_std_concat,
    "mean+min+max": mean_min_max_concat,
    "weighted_mean": weighted_mean,
    "attention": attention_pool,
}


class EmbeddingExtractor:
    def __init__(self, model_id: str = MODEL_ID, target_layer: int = None):
        print(f"\nLoading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(DEVICE)
        self.model.eval()

        n_layers = self.model.config.num_hidden_layers
        self.target_layer = target_layer if target_layer else n_layers // 2
        self.hidden_dim = self.model.config.hidden_size
        print(f"Extracting from layer {self.target_layer}/{n_layers}")

        self.captured_activation = None
        self._register_hook()

    def _register_hook(self):
        layer = self.model.model.layers[self.target_layer]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output[0]

            if hidden_states.dim() == 3:
                self.captured_activation = hidden_states[0, :, :].detach()
            elif hidden_states.dim() == 2:
                self.captured_activation = hidden_states.detach()

        layer.register_forward_hook(hook_fn)

    def get_raw_embeddings(self, text: str) -> torch.Tensor:
        """Get all token embeddings [seq_len, hidden_dim]."""
        messages = [
            {"role": "system", "content": "Analyze this tweet."},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)

        with torch.no_grad():
            self.model(**inputs)

        return self.captured_activation.float()


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_and_evaluate(X_train: torch.Tensor, y_train: torch.Tensor,
                       X_test: torch.Tensor, y_test: torch.Tensor,
                       epochs: int = 1000, lr: float = 0.01) -> tuple[float, float]:
    """Train logistic regression and return train/test accuracy."""
    # Normalize
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    model = LogisticRegression(X_train.shape[1]).to(DEVICE)
    nn.init.xavier_uniform_(model.linear.weight)
    nn.init.zeros_(model.linear.bias)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_norm).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        train_preds = (model(X_train_norm).squeeze() > 0.5).float()
        test_preds = (model(X_test_norm).squeeze() > 0.5).float()
        train_acc = (train_preds == y_train).float().mean().item()
        test_acc = (test_preds == y_test).float().mean().item()

    return train_acc, test_acc


def run_experiment():
    print("=" * 60)
    print("Pooling Strategy Comparison")
    print("=" * 60)

    tweets = load_tweets()

    # Split by engagement - sort by text to ensure determinism
    high = sorted([t for t in tweets if t.likes >= 50], key=lambda t: t.text)
    medium = sorted([t for t in tweets if 10 <= t.likes < 50], key=lambda t: t.text)
    zero = sorted([t for t in tweets if t.likes == 0], key=lambda t: t.text)

    # Use fixed indices after sorting (deterministic)
    random.seed(42)
    high_idx = list(range(len(high)))
    medium_idx = list(range(len(medium)))
    zero_idx = list(range(len(zero)))
    random.shuffle(high_idx)
    random.shuffle(medium_idx)
    random.shuffle(zero_idx)

    # Train: 10-49 likes vs 0 likes
    n_train = min(50, len(medium))
    train_tweets = [medium[i] for i in medium_idx[:n_train]] + [zero[i] for i in zero_idx[:n_train]]
    train_labels = [1.0] * n_train + [0.0] * n_train

    # Test: 50+ likes vs 0 likes (use DIFFERENT zero tweets)
    n_test = len(high)
    test_tweets = [high[i] for i in high_idx] + [zero[i] for i in zero_idx[n_train:n_train + n_test]]
    test_labels = [1.0] * n_test + [0.0] * n_test

    print(f"\nTrain: {len(train_tweets)} tweets ({n_train} high, {n_train} low)")
    print(f"Test: {len(test_tweets)} tweets ({n_test} high, {n_test} low)")

    # Extract raw embeddings
    extractor = EmbeddingExtractor()

    print("\nExtracting embeddings...")
    train_raw = []
    for i, tweet in enumerate(train_tweets):
        emb = extractor.get_raw_embeddings(tweet.text)
        train_raw.append(emb)
        if (i + 1) % 25 == 0:
            print(f"  Train: [{i+1}/{len(train_tweets)}]")

    test_raw = []
    for i, tweet in enumerate(test_tweets):
        emb = extractor.get_raw_embeddings(tweet.text)
        test_raw.append(emb)
        if (i + 1) % 25 == 0:
            print(f"  Test: [{i+1}/{len(test_tweets)}]")

    y_train = torch.tensor(train_labels, device=DEVICE)
    y_test = torch.tensor(test_labels, device=DEVICE)

    # Test each pooling strategy
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"{'Pooling Method':<20} {'Features':>10} {'Train Acc':>12} {'Test Acc':>12}")
    print("-" * 60)

    results = []
    for name, pool_fn in POOLING_FUNCTIONS.items():
        # Apply pooling
        X_train = torch.stack([pool_fn(emb) for emb in train_raw])
        X_test = torch.stack([pool_fn(emb) for emb in test_raw])

        train_acc, test_acc = train_and_evaluate(X_train, y_train, X_test, y_test)
        results.append((name, X_train.shape[1], train_acc, test_acc))
        print(f"{name:<20} {X_train.shape[1]:>10} {train_acc*100:>11.1f}% {test_acc*100:>11.1f}%")

    # Sort by test accuracy
    print("\n" + "=" * 60)
    print("Ranked by Test Accuracy")
    print("=" * 60)
    for name, features, train_acc, test_acc in sorted(results, key=lambda x: -x[3]):
        print(f"{name:<20} {test_acc*100:>6.1f}%")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    run_experiment()
