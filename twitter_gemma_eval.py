#!/usr/bin/env python3
"""
Evaluate Gemma embeddings + classifiers on Twitter same-author pairs.
Uses same embedding model as LinkedIn for fair comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"  # Open model, no auth needed
PAIRS_FILE = "data/twitter_same_author_small.json"
EMBEDDINGS_FILE = "data/twitter_qwen_embeddings.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model and tokenizer
model = None
tokenizer = None


def load_model():
    global model, tokenizer
    print(f"Loading {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def get_embedding(text: str, max_length: int = 512) -> np.ndarray:
    """Extract embedding from model's last hidden state (mean pooling)."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        embedding = hidden.mean(dim=1).squeeze().float().cpu().numpy()

    return embedding


def load_pairs():
    """Load Twitter same-author pairs."""
    print(f"Loading pairs from {PAIRS_FILE}...")
    with open(PAIRS_FILE, 'r') as f:
        data = json.load(f)

    train_pairs = data['train']
    test_pairs = data['test']

    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    return train_pairs, test_pairs


def extract_embeddings(pairs, desc=""):
    """Extract Gemma embeddings for pairs."""
    features = []
    labels = []

    for i, pair in enumerate(pairs):
        text_a = pair['tweet_a']['text']
        text_b = pair['tweet_b']['text']

        emb_a = get_embedding(text_a)
        emb_b = get_embedding(text_b)

        # Concatenate embeddings (same as LinkedIn)
        feature = np.concatenate([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b])
        features.append(feature)
        labels.append(pair['label'])

        if (i + 1) % 50 == 0:
            print(f"  {desc} [{i+1}/{len(pairs)}]")

    return np.array(features), np.array(labels)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def train_model(clf, X_train, y_train, epochs=200, lr=1e-3):
    optimizer = optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        clf.train()
        optimizer.zero_grad()
        logits = clf(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
    return clf


def get_predictions(clf, X):
    clf.eval()
    with torch.no_grad():
        logits = clf(X)
        preds = logits.argmax(dim=1)
    return preds.cpu().numpy()


def run():
    print("=" * 70)
    print("TWITTER SAME-AUTHOR: QWEN 0.5B EMBEDDINGS")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load or extract embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"\nLoading cached embeddings from {EMBEDDINGS_FILE}...")
        data = np.load(EMBEDDINGS_FILE)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
    else:
        train_pairs, test_pairs = load_pairs()
        load_model()

        print(f"\nExtracting train embeddings ({len(train_pairs)} pairs)...")
        X_train, y_train = extract_embeddings(train_pairs, "Train")

        print(f"\nExtracting test embeddings ({len(test_pairs)} pairs)...")
        X_test, y_test = extract_embeddings(test_pairs, "Test")

        print(f"\nSaving embeddings to {EMBEDDINGS_FILE}...")
        np.savez(EMBEDDINGS_FILE, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        # Free GPU memory
        global model, tokenizer
        del model, tokenizer
        torch.cuda.empty_cache()

    print(f"\nFeature shape: {X_train.shape}")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Move to GPU
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=DEVICE)
    y_test_np = y_test_t.cpu().numpy()

    # Normalize
    mean = X_train_t.mean(dim=0)
    std = X_train_t.std(dim=0) + 1e-8
    X_train_norm = (X_train_t - mean) / std
    X_test_norm = (X_test_t - mean) / std

    input_dim = X_train_norm.shape[1]

    # Train models
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (all on GPU)")
    print("=" * 70)

    models_config = [
        ("Ridge", lambda: LogisticRegression(input_dim)),
        ("MLP_256", lambda: MLP(input_dim, [256])),
        ("MLP_512_256", lambda: MLP(input_dim, [512, 256])),
    ]

    results = {}
    all_preds = {}

    for name, model_fn in models_config:
        print(f"\n{name}:")

        preds_runs = []
        accs = []
        for run_i in range(3):
            torch.manual_seed(42 + run_i)
            clf = model_fn().to(DEVICE)
            train_model(clf, X_train_norm, y_train_t)
            preds = get_predictions(clf, X_test_norm)
            preds_runs.append(preds)
            acc = (preds == y_test_np).mean()
            accs.append(acc)

        majority_preds = (np.stack(preds_runs).sum(axis=0) >= 2).astype(int)
        majority_acc = (majority_preds == y_test_np).mean()

        results[name] = majority_acc
        all_preds[name] = majority_preds

        print(f"  Runs: {[f'{a*100:.1f}%' for a in accs]}")
        print(f"  Majority vote: {majority_acc*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON: LLM EMBEDDINGS")
    print("=" * 70)

    print(f"\nTwitter Same-Author ({len(y_test)} test samples):")
    for name, acc in results.items():
        print(f"  {name}: {acc*100:.1f}%")

    print(f"\nLinkedIn Cross-Author (400 test samples):")
    print("  Ridge: ~74%")
    print("  MLP_256: ~75%")
    print("  MLP_512_256: ~75%")

    # Model agreement
    model_names = list(all_preds.keys())
    correct_counts = sum((all_preds[name] == y_test_np).astype(int) for name in model_names)
    all_wrong = (correct_counts == 0).sum()
    print(f"\nAll models wrong: {all_wrong}/{len(y_test_np)} ({all_wrong/len(y_test_np)*100:.1f}%)")


if __name__ == "__main__":
    run()
