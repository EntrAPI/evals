#!/usr/bin/env python3
"""
Compare different classifiers with top 5 pooling strategies.
Includes ensemble methods.
"""

import random
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from langdetect import detect, LangDetectException
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from xgboost import XGBClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

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
    """Get deterministic train/test split - 10-49 likes vs 0 likes for training."""
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

    # Train: ALL medium (10-49 likes) vs equal zero engagement
    n_medium = len(medium)
    n_train_neg = min(n_medium, len(zero) - len(high))  # Leave some zero for test

    train_pos = [medium[i] for i in medium_idx]
    train_neg = [zero[i] for i in zero_idx[:n_train_neg]]
    train_tweets = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)

    # Test: 50+ likes vs 0 likes (use different zero tweets)
    n_test = len(high)
    test_tweets = [high[i] for i in high_idx] + [zero[i] for i in zero_idx[n_train_neg:n_train_neg + n_test]]
    test_labels = [1] * n_test + [0] * n_test

    print(f"  High (50+): {len(high)}, Medium (10-49): {len(medium)}, Zero: {len(zero)}")

    return train_tweets, train_labels, test_tweets, test_labels


class EmbeddingExtractor:
    def __init__(self, model_id: str = MODEL_ID):
        print(f"\nLoading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(DEVICE)
        self.model.eval()

        n_layers = self.model.config.num_hidden_layers
        self.target_layer = n_layers // 2
        print(f"  Layers: {n_layers}, Using layer: {self.target_layer}")

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

    def get_embeddings(self, text: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": "Analyze this tweet."},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)

        with torch.no_grad():
            self.model(**inputs)

        return self.captured_activation.float()


# GPU Logistic Regression
class TorchLogisticRegression(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_torch_logreg(X_train, y_train, X_test, y_test, epochs=1000, lr=0.01, weight_decay=0.01):
    """GPU logistic regression."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    model = TorchLogisticRegression(X_train.shape[1]).to(DEVICE)
    nn.init.xavier_uniform_(model.linear.weight)
    nn.init.zeros_(model.linear.bias)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t).squeeze()
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (model(X_test_t).squeeze() > 0.5).cpu().numpy()
    return (preds == y_test).mean()


def make_voting_hard():
    """Create fresh hard voting ensemble."""
    return VotingClassifier(
        estimators=[
            ('logreg', LogisticRegression(max_iter=1000, C=1.0)),
            ('svm_rbf', SVC(kernel='rbf', C=1.0)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss', verbosity=0)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ],
        voting='hard'
    )


def make_voting_soft():
    """Create fresh soft voting ensemble."""
    return VotingClassifier(
        estimators=[
            ('logreg', LogisticRegression(max_iter=1000, C=1.0)),
            ('svm_rbf', SVC(kernel='rbf', C=1.0, probability=True)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss', verbosity=0)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ],
        voting='soft'
    )


def make_stacking():
    """Create fresh stacking ensemble."""
    return StackingClassifier(
        estimators=[
            ('logreg', LogisticRegression(max_iter=1000, C=1.0)),
            ('svm_rbf', SVC(kernel='rbf', C=1.0, probability=True)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss', verbosity=0)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )


def get_classifiers():
    """Return dict of classifier name -> (classifier factory, uses_gpu)."""
    return {
        "LogReg (GPU)": (lambda: "gpu_logreg", True),
        "LogReg": (lambda: LogisticRegression(max_iter=1000, C=1.0), False),
        "SVM (RBF)": (lambda: SVC(kernel='rbf', C=1.0), False),
        "XGBoost": (lambda: XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss', verbosity=0), False),
        "RandomForest": (lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42), False),
        "Voting (Hard)": (make_voting_hard, False),
        "Voting (Soft)": (make_voting_soft, False),
        "Stacking": (make_stacking, False),
    }


def run_experiment():
    print("=" * 80)
    print("Classifier & Pooling Comparison")
    print("=" * 80)

    tweets = load_tweets()
    train_tweets, train_labels, test_tweets, test_labels = get_data_split(tweets)

    print(f"\nTrain: {len(train_tweets)} tweets")
    print(f"Test: {len(test_tweets)} tweets")

    # Extract embeddings
    extractor = EmbeddingExtractor()

    print("\nExtracting embeddings...")
    train_raw = []
    for i, tweet in enumerate(train_tweets):
        emb = extractor.get_embeddings(tweet.text)
        train_raw.append(emb)
        if (i + 1) % 50 == 0:
            print(f"  Train: [{i+1}/{len(train_tweets)}]")

    test_raw = []
    for i, tweet in enumerate(test_tweets):
        emb = extractor.get_embeddings(tweet.text)
        test_raw.append(emb)
        if (i + 1) % 50 == 0:
            print(f"  Test: [{i+1}/{len(test_tweets)}]")

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    classifiers = get_classifiers()
    results = {name: {} for name in classifiers}

    print("\n" + "=" * 80)
    print("Training classifiers...")
    print("=" * 80)

    for pool_name, pool_fn in POOLING_METHODS.items():
        print(f"\nPooling: {pool_name}")

        # Apply pooling
        X_train = torch.stack([pool_fn(emb) for emb in train_raw]).cpu().numpy()
        X_test = torch.stack([pool_fn(emb) for emb in test_raw]).cpu().numpy()

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for clf_name, (clf_factory, uses_gpu) in classifiers.items():
            if uses_gpu:
                acc = train_torch_logreg(X_train_scaled, y_train, X_test_scaled, y_test)
            else:
                clf = clf_factory()
                clf.fit(X_train_scaled, y_train)
                preds = clf.predict(X_test_scaled)
                acc = (preds == y_test).mean()

            results[clf_name][pool_name] = acc
            print(f"  {clf_name}: {acc*100:.1f}%")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY (Test Accuracy %)")
    print("=" * 80)

    # Header
    pool_names = list(POOLING_METHODS.keys())
    header = f"{'Classifier':<18}" + "".join(f"{p:>14}" for p in pool_names) + f"{'BEST':>10}"
    print(header)
    print("-" * len(header))

    # Rows sorted by best accuracy
    clf_best = [(name, max(results[name].values())) for name in classifiers]
    clf_best.sort(key=lambda x: -x[1])

    for clf_name, _ in clf_best:
        row = f"{clf_name:<18}"
        for pool_name in pool_names:
            acc = results[clf_name][pool_name]
            row += f"{acc*100:>13.1f}%"
        best = max(results[clf_name].values())
        row += f"{best*100:>9.1f}%"
        print(row)

    # Best per pooling
    print("-" * len(header))
    row = f"{'BEST':<18}"
    for pool_name in pool_names:
        best = max(results[clf_name][pool_name] for clf_name in classifiers)
        row += f"{best*100:>13.1f}%"
    overall_best = max(max(r.values()) for r in results.values())
    row += f"{overall_best*100:>9.1f}%"
    print(row)

    print(f"\nOverall best: {overall_best*100:.1f}%")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    run_experiment()
