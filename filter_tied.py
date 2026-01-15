#!/usr/bin/env python3
"""
Filter out exactly tied samples, keep nearly tied.
Re-evaluate models on cleaned data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

EMBEDDINGS_FILE = "data/gemma_1b_embeddings.npz"
PAIRS_FILE = "data/linkedin_pairs.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all_data():
    print("Loading embeddings...")
    data = np.load(EMBEDDINGS_FILE)

    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32, device=DEVICE)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=DEVICE)

    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    print("Loading LinkedIn pairs...")
    with open(PAIRS_FILE, 'r') as f:
        pairs_data = json.load(f)

    train_pairs = pairs_data['train']
    test_pairs = pairs_data['test']

    return X_train_norm, y_train, X_test_norm, y_test, train_pairs, test_pairs


def get_engagement(pair):
    """Extract engagement from pair."""
    if isinstance(pair.get('post_a'), dict):
        eng_a = pair['post_a'].get('reactions', 0) + pair['post_a'].get('comments', 0)
        eng_b = pair['post_b'].get('reactions', 0) + pair['post_b'].get('comments', 0)
    else:
        eng_a = pair.get('engagement_a', pair.get('likes_a', 0))
        eng_b = pair.get('engagement_b', pair.get('likes_b', 0))
    return eng_a, eng_b


def get_post_text(pair):
    """Extract post text from pair."""
    if isinstance(pair.get('post_a'), dict):
        return pair['post_a'].get('text', ''), pair['post_b'].get('text', '')
    return pair.get('post_a', ''), pair.get('post_b', '')


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
    print("FILTER TIED SAMPLES ANALYSIS")
    print("=" * 70)

    X_train, y_train, X_test, y_test, train_pairs, test_pairs = load_all_data()
    input_dim = X_train.shape[1]
    y_test_np = y_test.cpu().numpy()

    # Analyze test set for tied samples
    print("\n" + "=" * 70)
    print("ANALYZING TEST SET FOR TIED SAMPLES")
    print("=" * 70)

    tied_indices = []
    nearly_tied_indices = []  # ratio < 1.5 but not exactly tied
    clear_indices = []  # ratio >= 1.5

    for idx, pair in enumerate(test_pairs):
        eng_a, eng_b = get_engagement(pair)

        if eng_a == eng_b:
            tied_indices.append(idx)
        else:
            ratio = max(eng_a, eng_b) / (min(eng_a, eng_b) + 1)
            if ratio < 1.5:
                nearly_tied_indices.append(idx)
            else:
                clear_indices.append(idx)

    print(f"\nExactly tied (eng_a == eng_b): {len(tied_indices)}")
    print(f"Nearly tied (ratio < 1.5): {len(nearly_tied_indices)}")
    print(f"Clear winner (ratio >= 1.5): {len(clear_indices)}")

    # Show the exactly tied samples
    print("\n--- Exactly Tied Samples (to be removed) ---")
    for idx in tied_indices:
        pair = test_pairs[idx]
        eng_a, eng_b = get_engagement(pair)
        post_a, post_b = get_post_text(pair)
        print(f"\n  Sample {idx}: eng_a={eng_a}, eng_b={eng_b}, label={y_test_np[idx]}")
        print(f"    A: {post_a[:80]}...")
        print(f"    B: {post_b[:80]}...")

    # Show the nearly tied samples (kept)
    print("\n--- Nearly Tied Samples (kept) ---")
    for idx in nearly_tied_indices[:5]:
        pair = test_pairs[idx]
        eng_a, eng_b = get_engagement(pair)
        ratio = max(eng_a, eng_b) / (min(eng_a, eng_b) + 1)
        print(f"  Sample {idx}: eng_a={eng_a}, eng_b={eng_b}, ratio={ratio:.2f}, label={y_test_np[idx]}")

    # Create filtered test set (remove exactly tied)
    keep_mask = np.ones(len(test_pairs), dtype=bool)
    keep_mask[tied_indices] = False

    X_test_filtered = X_test[keep_mask]
    y_test_filtered = y_test[keep_mask]
    y_test_filtered_np = y_test_filtered.cpu().numpy()

    print(f"\n\nFiltered test set: {len(y_test_filtered_np)} samples (removed {len(tied_indices)} tied)")

    # Train and evaluate models on filtered data
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE ON FILTERED DATA")
    print("=" * 70)

    models_config = [
        ("Ridge", lambda: LogisticRegression(input_dim)),
        ("MLP_256", lambda: MLP(input_dim, [256])),
        ("MLP_512_256", lambda: MLP(input_dim, [512, 256])),
    ]

    results_original = {}
    results_filtered = {}
    all_preds_filtered = {}
    all_correct_filtered = {}

    for name, model_fn in models_config:
        print(f"\n{name}:")

        # Train (same as before)
        preds_runs = []
        for run_i in range(3):
            torch.manual_seed(42 + run_i)
            model = model_fn().to(DEVICE)
            train_model(model, X_train, y_train)
            preds, probs = get_predictions(model, X_test)
            preds_runs.append(preds)

        majority_preds = (np.stack(preds_runs).sum(axis=0) >= 2).astype(int)

        # Original accuracy
        acc_original = (majority_preds == y_test_np).mean()
        results_original[name] = acc_original

        # Filtered accuracy
        preds_filtered = majority_preds[keep_mask]
        acc_filtered = (preds_filtered == y_test_filtered_np).mean()
        results_filtered[name] = acc_filtered

        all_preds_filtered[name] = preds_filtered
        all_correct_filtered[name] = (preds_filtered == y_test_filtered_np)

        print(f"  Original (400 samples): {acc_original*100:.1f}%")
        print(f"  Filtered ({len(y_test_filtered_np)} samples): {acc_filtered*100:.1f}%")
        print(f"  Improvement: +{(acc_filtered - acc_original)*100:.1f}pp")

    # Analyze remaining errors on filtered data
    print("\n" + "=" * 70)
    print("REMAINING ERRORS ON FILTERED DATA")
    print("=" * 70)

    model_names = list(all_preds_filtered.keys())
    correct_counts = sum(all_correct_filtered[name].astype(int) for name in model_names)

    all_right = (correct_counts == len(model_names))
    all_wrong = (correct_counts == 0)
    contested = ~all_right & ~all_wrong

    print(f"\nAll models correct: {all_right.sum()} ({all_right.sum()/len(y_test_filtered_np)*100:.1f}%)")
    print(f"All models wrong: {all_wrong.sum()} ({all_wrong.sum()/len(y_test_filtered_np)*100:.1f}%)")
    print(f"Contested: {contested.sum()} ({contested.sum()/len(y_test_filtered_np)*100:.1f}%)")

    # Map back to original indices
    filtered_to_original = np.where(keep_mask)[0]
    all_wrong_orig_indices = filtered_to_original[all_wrong]

    print(f"\n--- Samples ALL models still get wrong (filtered set) ---")

    # Categorize by engagement ratio
    high_ratio_wrong = []
    low_ratio_wrong = []

    for filtered_idx in np.where(all_wrong)[0]:
        orig_idx = filtered_to_original[filtered_idx]
        pair = test_pairs[orig_idx]
        eng_a, eng_b = get_engagement(pair)
        true_label = y_test_filtered_np[filtered_idx]

        if true_label == 0:  # A should win
            ratio = eng_a / (eng_b + 1)
        else:  # B should win
            ratio = eng_b / (eng_a + 1)

        if ratio >= 2.0:
            high_ratio_wrong.append((orig_idx, filtered_idx, eng_a, eng_b, ratio, true_label))
        else:
            low_ratio_wrong.append((orig_idx, filtered_idx, eng_a, eng_b, ratio, true_label))

    print(f"\nHigh ratio wrong (winner has 2x+ engagement): {len(high_ratio_wrong)}")
    print(f"Low ratio wrong (winner has <2x engagement): {len(low_ratio_wrong)}")

    print("\n--- High Ratio Wrong (most suspicious) ---")
    for orig_idx, filt_idx, eng_a, eng_b, ratio, label in sorted(high_ratio_wrong, key=lambda x: -x[4])[:10]:
        pair = test_pairs[orig_idx]
        post_a, post_b = get_post_text(pair)
        print(f"\n  Sample {orig_idx}: label={label}, ratio={ratio:.1f}x")
        print(f"    Engagement: A={eng_a}, B={eng_b}")
        print(f"    A: {post_a[:100]}...")
        print(f"    B: {post_b[:100]}...")

    # Agreement-based accuracy on filtered data
    print("\n" + "=" * 70)
    print("AGREEMENT-BASED ACCURACY (FILTERED)")
    print("=" * 70)

    for agree_threshold in [2, 3]:
        correct = 0
        total = 0

        for idx in range(len(y_test_filtered_np)):
            preds_here = [all_preds_filtered[name][idx] for name in model_names]
            vote = sum(preds_here)

            if vote >= agree_threshold or vote <= (len(model_names) - agree_threshold):
                pred = 1 if vote > len(model_names) / 2 else 0
                if pred == y_test_filtered_np[idx]:
                    correct += 1
                total += 1

        if total > 0:
            print(f"Agreement >= {agree_threshold}: {correct}/{total} = {correct/total*100:.1f}% ({total} samples)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOriginal test set: 400 samples")
    print(f"After removing exactly tied: {len(y_test_filtered_np)} samples")
    print(f"Tied samples removed: {len(tied_indices)}")

    print(f"\nAccuracy comparison:")
    for name in model_names:
        print(f"  {name}: {results_original[name]*100:.1f}% -> {results_filtered[name]*100:.1f}% (+{(results_filtered[name]-results_original[name])*100:.1f}pp)")


if __name__ == "__main__":
    run()
