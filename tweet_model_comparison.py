#!/usr/bin/env python3
"""
Compare different models with top 5 pooling strategies.
"""

import gc
import random
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
from datasets import load_dataset
from langdetect import detect, LangDetectException
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Models to test (small enough for 8GB VRAM)
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3.5-mini-instruct",
]

# Top 5 pooling methods
def weighted_mean(hidden_states: torch.Tensor) -> torch.Tensor:
    seq_len = hidden_states.shape[0]
    weights = torch.arange(1, seq_len + 1, device=hidden_states.device, dtype=hidden_states.dtype)
    weights = weights / weights.sum()
    return (hidden_states * weights.unsqueeze(1)).sum(dim=0)

def mean_std_concat(hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.cat([hidden_states.mean(dim=0), hidden_states.std(dim=0)])

def mean_min_max_concat(hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.cat([hidden_states.mean(dim=0), hidden_states.min(dim=0).values, hidden_states.max(dim=0).values])

def mean_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states.mean(dim=0)

def last_token(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states[-1, :]

POOLING_METHODS = {
    "weighted_mean": weighted_mean,
    "mean+std": mean_std_concat,
    "mean+min+max": mean_min_max_concat,
    "mean": mean_pool,
    "last_token": last_token,
}


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


def get_data_split(tweets: list[Tweet]):
    """Get deterministic train/test split."""
    high = sorted([t for t in tweets if t.likes >= 50], key=lambda t: t.text)
    medium = sorted([t for t in tweets if 10 <= t.likes < 50], key=lambda t: t.text)
    zero = sorted([t for t in tweets if t.likes == 0], key=lambda t: t.text)

    random.seed(42)
    high_idx = list(range(len(high)))
    medium_idx = list(range(len(medium)))
    zero_idx = list(range(len(zero)))
    random.shuffle(high_idx)
    random.shuffle(medium_idx)
    random.shuffle(zero_idx)

    n_train = min(50, len(medium))
    train_tweets = [medium[i] for i in medium_idx[:n_train]] + [zero[i] for i in zero_idx[:n_train]]
    train_labels = [1.0] * n_train + [0.0] * n_train

    n_test = len(high)
    test_tweets = [high[i] for i in high_idx] + [zero[i] for i in zero_idx[n_train:n_train + n_test]]
    test_labels = [1.0] * n_test + [0.0] * n_test

    return train_tweets, train_labels, test_tweets, test_labels


class EmbeddingExtractor:
    def __init__(self, model_id: str):
        print(f"  Loading {model_id}...")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(DEVICE)
        self.model.eval()

        # Get model config
        n_layers = self.model.config.num_hidden_layers
        self.target_layer = n_layers // 2
        self.hidden_dim = self.model.config.hidden_size
        print(f"    Layers: {n_layers}, Hidden: {self.hidden_dim}, Using layer: {self.target_layer}")

        self.captured_activation = None
        self._register_hook()

    def _register_hook(self):
        # Handle different model architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layer = self.model.model.layers[self.target_layer]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layer = self.model.transformer.h[self.target_layer]
        else:
            # Try to find layers
            for name, module in self.model.named_modules():
                if f'.{self.target_layer}.' in name or name.endswith(f'.{self.target_layer}'):
                    layer = module
                    break
            else:
                raise ValueError(f"Cannot find layers in model {self.model_id}")

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if hidden_states.dim() == 3:
                self.captured_activation = hidden_states[0, :, :].detach()
            elif hidden_states.dim() == 2:
                self.captured_activation = hidden_states.detach()

        layer.register_forward_hook(hook_fn)

    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get all token embeddings."""
        # Simple prompt without chat template for compatibility
        prompt = f"Tweet: {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)

        with torch.no_grad():
            self.model(**inputs)

        return self.captured_activation.float()

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train logistic regression and return test accuracy."""
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    model = LogisticRegression(X_train.shape[1]).to(DEVICE)
    nn.init.xavier_uniform_(model.linear.weight)
    nn.init.zeros_(model.linear.bias)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_norm).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_preds = (model(X_test_norm).squeeze() > 0.5).float()
        test_acc = (test_preds == y_test).float().mean().item()

    return test_acc


def run_experiment():
    print("=" * 70)
    print("Model & Pooling Comparison")
    print("=" * 70)

    tweets = load_tweets()
    train_tweets, train_labels, test_tweets, test_labels = get_data_split(tweets)

    print(f"\nTrain: {len(train_tweets)} tweets")
    print(f"Test: {len(test_tweets)} tweets")

    y_train = torch.tensor(train_labels, device=DEVICE)
    y_test = torch.tensor(test_labels, device=DEVICE)

    results = {}

    for model_id in MODELS:
        print(f"\n{'='*70}")
        print(f"Model: {model_id}")
        print("=" * 70)

        try:
            extractor = EmbeddingExtractor(model_id)

            # Extract embeddings
            print("  Extracting embeddings...")
            train_raw = []
            for i, tweet in enumerate(train_tweets):
                emb = extractor.get_embeddings(tweet.text)
                train_raw.append(emb)
                if (i + 1) % 50 == 0:
                    print(f"    Train: [{i+1}/{len(train_tweets)}]")

            test_raw = []
            for i, tweet in enumerate(test_tweets):
                emb = extractor.get_embeddings(tweet.text)
                test_raw.append(emb)
                if (i + 1) % 50 == 0:
                    print(f"    Test: [{i+1}/{len(test_tweets)}]")

            # Test each pooling method
            model_results = {}
            for pool_name, pool_fn in POOLING_METHODS.items():
                X_train = torch.stack([pool_fn(emb) for emb in train_raw])
                X_test = torch.stack([pool_fn(emb) for emb in test_raw])

                acc = train_and_evaluate(X_train, y_train, X_test, y_test)
                model_results[pool_name] = acc
                print(f"    {pool_name}: {acc*100:.1f}%")

            results[model_id] = model_results

            # Cleanup
            extractor.cleanup()

        except Exception as e:
            print(f"  ERROR: {e}")
            results[model_id] = {"error": str(e)}
            gc.collect()
            torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY (Test Accuracy %)")
    print("=" * 70)

    # Header
    model_names = [m.split("/")[-1][:20] for m in MODELS]
    header = f"{'Pooling':<15}" + "".join(f"{name:>22}" for name in model_names)
    print(header)
    print("-" * len(header))

    # Rows
    for pool_name in POOLING_METHODS.keys():
        row = f"{pool_name:<15}"
        for model_id in MODELS:
            if model_id in results and pool_name in results[model_id]:
                acc = results[model_id][pool_name]
                row += f"{acc*100:>21.1f}%"
            else:
                row += f"{'ERR':>22}"
        print(row)

    # Best per model
    print("-" * len(header))
    row = f"{'BEST':<15}"
    for model_id in MODELS:
        if model_id in results and "error" not in results[model_id]:
            best = max(results[model_id].values())
            row += f"{best*100:>21.1f}%"
        else:
            row += f"{'ERR':>22}"
    print(row)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    run_experiment()
