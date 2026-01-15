#!/usr/bin/env python3
"""
Evaluate embeddings + classifiers on Twitter same-author pairs.
This controls for follower count - both tweets in each pair are from the same user.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
import os
from sentence_transformers import SentenceTransformer

MODEL_ID = "all-MiniLM-L6-v2"  # Fast, efficient sentence embedding model
PAIRS_FILE = "data/twitter_same_author_small.json"
EMBEDDINGS_FILE = "data/twitter_same_author_embeddings.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pairs():
    """Load Twitter same-author pairs."""
    print(f"Loading pairs from {PAIRS_FILE}...")
    with open(PAIRS_FILE, 'r') as f:
        data = json.load(f)

    train_pairs = data['train']
    test_pairs = data['test']

    print(f"Train: {len(train_pairs):,}, Test: {len(test_pairs):,}")

    return train_pairs, test_pairs


def extract_embeddings(model, pairs, desc=""):
    """Extract embeddings for pairs."""
    texts_a = [p['tweet_a']['text'] for p in pairs]
    texts_b = [p['tweet_b']['text'] for p in pairs]
    labels = [p['label'] for p in pairs]

    print(f"  Encoding {len(texts_a)} tweet pairs...")
    emb_a = model.encode(texts_a, show_progress_bar=True, batch_size=32)
    emb_b = model.encode(texts_b, show_progress_bar=True, batch_size=32)

    # Concatenate embeddings
    features = np.concatenate([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b], axis=1)

    return features, np.array(labels)


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


def train_model(model, X_train, y_train, epochs=200, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
    return model


def get_predictions(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)
    return preds.cpu().numpy(), probs.cpu().numpy()


def run():
    print("=" * 70)
    print("TWITTER SAME-AUTHOR A/B TEST")
    print("=" * 70)
    print("This dataset controls for follower count - both tweets are from same user")

    # Load or extract embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"\nLoading cached embeddings from {EMBEDDINGS_FILE}...")
        data = np.load(EMBEDDINGS_FILE)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
    else:
        train_pairs, test_pairs = load_pairs()

        print(f"\nLoading embedding model {MODEL_ID}...")
        emb_model = SentenceTransformer(MODEL_ID, device=str(DEVICE))

        print("\nExtracting train embeddings...")
        X_train, y_train = extract_embeddings(emb_model, train_pairs, "Train")

        print("\nExtracting test embeddings...")
        X_test, y_test = extract_embeddings(emb_model, test_pairs, "Test")

        print(f"\nSaving embeddings to {EMBEDDINGS_FILE}...")
        np.savez(EMBEDDINGS_FILE, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print(f"\nFeature shape: {X_train.shape}")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Convert to tensors
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

    # Train multiple models
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    models_config = [
        ("Ridge", lambda: LogisticRegression(input_dim)),
        ("MLP_256", lambda: MLP(input_dim, [256])),
        ("MLP_512_256", lambda: MLP(input_dim, [512, 256])),
        ("MLP_256_128", lambda: MLP(input_dim, [256, 128])),
    ]

    results = {}
    all_preds = {}

    for name, model_fn in models_config:
        print(f"\n{name}:")

        # Multiple runs for stability
        preds_runs = []
        accs = []
        for run_i in range(3):
            torch.manual_seed(42 + run_i)
            clf = model_fn().to(DEVICE)
            train_model(clf, X_train_norm, y_train_t)
            preds, probs = get_predictions(clf, X_test_norm)
            preds_runs.append(preds)
            acc = (preds == y_test_np).mean()
            accs.append(acc)

        # Majority vote
        majority_preds = (np.stack(preds_runs).sum(axis=0) >= 2).astype(int)
        majority_acc = (majority_preds == y_test_np).mean()

        results[name] = {
            'mean': np.mean(accs),
            'std': np.std(accs),
            'majority': majority_acc,
        }
        all_preds[name] = majority_preds

        print(f"  Run accs: {[f'{a*100:.1f}%' for a in accs]}")
        print(f"  Mean: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
        print(f"  Majority vote: {majority_acc*100:.1f}%")

    # Analysis by engagement ratio
    print("\n" + "=" * 70)
    print("ACCURACY BY ENGAGEMENT RATIO")
    print("=" * 70)

    # Need to reload pairs to get ratios
    with open(PAIRS_FILE, 'r') as f:
        data = json.load(f)

    test_pairs = data['test']

    ratio_bins = [(1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, float('inf'))]

    for low, high in ratio_bins:
        indices = [i for i, p in enumerate(test_pairs) if low <= p['ratio'] < high]
        if len(indices) == 0:
            continue

        print(f"\nRatio {low:.1f}x - {high:.1f}x ({len(indices)} samples):")
        for name in results:
            preds = all_preds[name][indices]
            labels = y_test_np[indices]
            acc = (preds == labels).mean()
            print(f"  {name}: {acc*100:.1f}%")

    # Model agreement analysis
    print("\n" + "=" * 70)
    print("MODEL AGREEMENT ANALYSIS")
    print("=" * 70)

    model_names = list(all_preds.keys())
    correct_counts = sum((all_preds[name] == y_test_np).astype(int) for name in model_names)

    all_right = (correct_counts == len(model_names))
    all_wrong = (correct_counts == 0)
    contested = ~all_right & ~all_wrong

    print(f"\nAll models correct: {all_right.sum()} ({all_right.sum()/len(y_test_np)*100:.1f}%)")
    print(f"All models wrong: {all_wrong.sum()} ({all_wrong.sum()/len(y_test_np)*100:.1f}%)")
    print(f"Contested: {contested.sum()} ({contested.sum()/len(y_test_np)*100:.1f}%)")

    # Show some examples where all models are wrong
    print("\n--- Examples where ALL models got it wrong ---")
    wrong_indices = np.where(all_wrong)[0][:5]
    for idx in wrong_indices:
        pair = test_pairs[idx]
        print(f"\nSample {idx}: ratio={pair['ratio']:.1f}x, label={y_test_np[idx]}")
        print(f"  Tweet A ({pair['engagement_a']} eng): {pair['tweet_a']['text'][:80]}...")
        print(f"  Tweet B ({pair['engagement_b']} eng): {pair['tweet_b']['text'][:80]}...")
        winner = "A" if pair['label'] == 0 else "B"
        print(f"  Ground truth: {winner} won")

    # Summary comparison with LinkedIn
    print("\n" + "=" * 70)
    print("SUMMARY: TWITTER SAME-AUTHOR vs LINKEDIN CROSS-AUTHOR")
    print("=" * 70)
    print("\nTwitter Same-Author (this run):")
    for name in results:
        print(f"  {name}: {results[name]['majority']*100:.1f}%")

    print("\nLinkedIn Cross-Author (previous results, Gemma embeddings):")
    print("  Ridge: ~74%")
    print("  MLP_256: ~75%")
    print("  MLP_512_256: ~75%")

    print("\nKey insight: Same-author pairs control for follower count.")
    print("If accuracy is similar, embeddings capture content quality.")
    print("If accuracy drops, LinkedIn models were predicting 'who has more followers'.")


if __name__ == "__main__":
    run()
