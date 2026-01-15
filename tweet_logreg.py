#!/usr/bin/env python3
"""
Tweet A/B Testing with Logistic Regression on Activations (GPU)

1. Get model embeddings for high/low engagement tweets
2. Train logistic regression on steering set (10-49 likes vs 0)
3. Test on test set (50+ likes vs 0)

All on GPU.
"""

import random
import re
from dataclasses import dataclass

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
    retweets: int


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

        tweets.append(Tweet(
            text=text,
            likes=int(row["nlikes"] or 0),
            retweets=int(row["nretweets"] or 0),
        ))

    print(f"Loaded {len(tweets)} tweets")
    return tweets


class EmbeddingExtractor:
    def __init__(self, model_id: str = MODEL_ID, target_layer: int = None, max_tokens: int = 64):
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
        self.max_tokens = max_tokens
        print(f"Model has {n_layers} layers, extracting from layer {self.target_layer}")
        print(f"Max tokens: {max_tokens}, hidden dim: {self.hidden_dim}")
        print(f"Flattened embedding size: {max_tokens * self.hidden_dim}")

        self.captured_activation = None
        self._register_hook()

    def _register_hook(self):
        layer = self.model.model.layers[self.target_layer]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output[0]

            # Keep all token embeddings [seq_len, hidden_dim]
            if hidden_states.dim() == 3:
                self.captured_activation = hidden_states[0, :, :].detach()  # [seq_len, hidden_dim]
            elif hidden_states.dim() == 2:
                self.captured_activation = hidden_states.detach()

        layer.register_forward_hook(hook_fn)

    def get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for a single tweet - all tokens, padded/truncated to max_tokens."""
        messages = [
            {"role": "system", "content": "Analyze this tweet."},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_tokens).to(DEVICE)

        with torch.no_grad():
            self.model(**inputs)

        # Pad or truncate to fixed size
        emb = self.captured_activation.float()  # [seq_len, hidden_dim]
        seq_len = emb.shape[0]

        if seq_len < self.max_tokens:
            # Pad with zeros
            padding = torch.zeros(self.max_tokens - seq_len, self.hidden_dim, device=DEVICE)
            emb = torch.cat([emb, padding], dim=0)
        elif seq_len > self.max_tokens:
            # Truncate
            emb = emb[:self.max_tokens, :]

        # Flatten to 1D
        return emb.flatten()  # [max_tokens * hidden_dim]


class LogisticRegression(nn.Module):
    """Simple logistic regression on GPU."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_logreg(X_train: torch.Tensor, y_train: torch.Tensor,
                 epochs: int = 2000, lr: float = 0.01, weight_decay: float = 0.1) -> LogisticRegression:
    """Train logistic regression on GPU with L2 regularization."""
    input_dim = X_train.shape[1]

    # Normalize inputs
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train_norm = (X_train - mean) / std

    print(f"  Input dim: {input_dim}")
    print(f"  X stats - mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"  X_norm stats - mean: {X_train_norm.mean():.4f}, std: {X_train_norm.std():.4f}")

    model = LogisticRegression(input_dim).to(DEVICE)

    # Initialize weights small
    nn.init.xavier_uniform_(model.linear.weight)
    nn.init.zeros_(model.linear.bias)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_norm).squeeze()
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 400 == 0:
            with torch.no_grad():
                preds = (outputs > 0.5).float()
                acc = (preds == y_train).float().mean()
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {acc.item()*100:.1f}%")

    # Store normalization params for evaluation
    model.mean = mean
    model.std = std
    return model


def evaluate(model: LogisticRegression, X: torch.Tensor, y: torch.Tensor) -> dict:
    """Evaluate model on GPU."""
    model.eval()
    with torch.no_grad():
        # Apply same normalization
        X_norm = (X - model.mean) / model.std
        outputs = model(X_norm).squeeze()
        preds = (outputs > 0.5).float()

        acc = (preds == y).float().mean().item()

        # Per-class metrics
        tp = ((preds == 1) & (y == 1)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        tn = ((preds == 0) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def run_experiment():
    print("=" * 60)
    print("Tweet Classification with Logistic Regression (GPU)")
    print("=" * 60)

    tweets = load_tweets()

    # Split tweets by engagement level
    high_engagement = [t for t in tweets if t.likes >= 50]
    medium_engagement = [t for t in tweets if 10 <= t.likes < 50]
    zero_engagement = [t for t in tweets if t.likes == 0]

    print(f"\nTweet counts:")
    print(f"  50+ likes (test high): {len(high_engagement)}")
    print(f"  10-49 likes (train high): {len(medium_engagement)}")
    print(f"  0 likes: {len(zero_engagement)}")

    random.shuffle(medium_engagement)
    random.shuffle(zero_engagement)
    random.shuffle(high_engagement)

    # Training set: 10-49 likes vs 0 likes
    n_train = min(50, len(medium_engagement))
    train_high = medium_engagement[:n_train]
    train_low = zero_engagement[:n_train]

    # Test set: 50+ likes vs 0 likes
    n_test = len(high_engagement)
    test_high = high_engagement
    test_low = zero_engagement[n_train:n_train + n_test]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_high)} high + {len(train_low)} low = {len(train_high) + len(train_low)}")
    print(f"  Test: {len(test_high)} high + {len(test_low)} low = {len(test_high) + len(test_low)}")

    # Extract embeddings (on GPU)
    extractor = EmbeddingExtractor()

    print("\nExtracting training embeddings...")
    X_train_list = []
    y_train_list = []
    for i, tweet in enumerate(train_high + train_low):
        emb = extractor.get_embedding(tweet.text)
        X_train_list.append(emb)
        y_train_list.append(1.0 if tweet.likes >= 10 else 0.0)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(train_high) + len(train_low)}]")

    print("\nExtracting test embeddings...")
    X_test_list = []
    y_test_list = []
    for i, tweet in enumerate(test_high + test_low):
        emb = extractor.get_embedding(tweet.text)
        X_test_list.append(emb)
        y_test_list.append(1.0 if tweet.likes >= 50 else 0.0)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(test_high) + len(test_low)}]")

    # Stack tensors (all on GPU)
    X_train = torch.stack(X_train_list)
    y_train = torch.tensor(y_train_list, device=DEVICE)
    X_test = torch.stack(X_test_list)
    y_test = torch.tensor(y_test_list, device=DEVICE)

    print(f"\nFeature shapes (on {DEVICE}):")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")

    # Train logistic regression
    print("\n" + "=" * 60)
    print("Training Logistic Regression")
    print("=" * 60)

    model = train_logreg(X_train, y_train, epochs=1000, lr=0.01)

    # Evaluate
    train_metrics = evaluate(model, X_train, y_train)
    test_metrics = evaluate(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nTrain accuracy: {train_metrics['accuracy']*100:.1f}%")
    print(f"Test accuracy: {test_metrics['accuracy']*100:.1f}%")
    print(f"Random baseline: 50.0%")

    print(f"\nTest set details:")
    print(f"  Precision (high engagement): {test_metrics['precision']*100:.1f}%")
    print(f"  Recall (high engagement): {test_metrics['recall']*100:.1f}%")
    print(f"  F1 score: {test_metrics['f1']*100:.1f}%")
    print(f"  Confusion: TP={test_metrics['tp']}, FP={test_metrics['fp']}, TN={test_metrics['tn']}, FN={test_metrics['fn']}")

    # Get weights for interpretability
    with torch.no_grad():
        weights = model.linear.weight.squeeze().cpu()
        top_positive = torch.argsort(weights, descending=True)[:10]
        top_negative = torch.argsort(weights)[:10]

    print(f"\nTop 10 positive weight dimensions (predict high engagement):")
    for i, idx in enumerate(top_positive):
        print(f"  {i+1}. Dim {idx.item()}: {weights[idx].item():.4f}")

    print(f"\nTop 10 negative weight dimensions (predict low engagement):")
    for i, idx in enumerate(top_negative):
        print(f"  {i+1}. Dim {idx.item()}: {weights[idx].item():.4f}")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    run_experiment()
