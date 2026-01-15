#!/usr/bin/env python3
"""
Analyze overlap between different models' predictions.
Which samples do they agree on? Which are contested?
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict

EMBEDDINGS_FILE = "data/gemma_1b_embeddings.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    data = np.load(EMBEDDINGS_FILE)

    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32, device=DEVICE)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=DEVICE)

    # Normalize
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    print(f"Data: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train_norm, y_train, X_test_norm, y_test, X_train, X_test


# ============== MODELS ==============

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


class TinyRecursive(nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        self.hidden_dim = hidden_dim

    def forward(self, x, K=32):
        B = x.shape[0]
        x_proj = self.input_proj(x)
        z = torch.zeros(B, self.hidden_dim, device=x.device)
        for _ in range(K):
            z = z + self.update(torch.cat([z, x_proj], dim=-1))
            z = self.norm(z)
        return self.output(z)


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


def get_predictions(model, X_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)
    return preds.cpu().numpy(), probs.cpu().numpy()


def run():
    print("=" * 70)
    print("Analyzing Model Overlap")
    print("=" * 70)

    X_train, y_train, X_test, y_test, X_train_raw, X_test_raw = load_data()
    input_dim = X_train.shape[1]
    y_test_np = y_test.cpu().numpy()

    # Train multiple models
    models_config = [
        ("Ridge", lambda: LogisticRegression(input_dim), {}),
        ("MLP_256", lambda: MLP(input_dim, [256]), {}),
        ("MLP_512_256", lambda: MLP(input_dim, [512, 256]), {}),
        ("TRM_h8_K32", lambda: TinyRecursive(input_dim, hidden_dim=8), {}),
        ("TRM_h4_K64", lambda: TinyRecursive(input_dim, hidden_dim=4), {}),
    ]

    # Train each model 3 times and take majority vote
    all_predictions = {}
    all_probs = {}

    for name, model_fn, train_kwargs in models_config:
        print(f"\nTraining {name}...")
        preds_runs = []
        probs_runs = []

        for run_i in range(3):
            torch.manual_seed(42 + run_i)
            model = model_fn().to(DEVICE)
            train_model(model, X_train, y_train, **train_kwargs)
            preds, probs = get_predictions(model, X_test)
            preds_runs.append(preds)
            probs_runs.append(probs)

        # Majority vote
        preds_stack = np.stack(preds_runs)
        majority_preds = (preds_stack.sum(axis=0) >= 2).astype(int)
        avg_probs = np.mean(probs_runs, axis=0)

        acc = (majority_preds == y_test_np).mean()
        print(f"  {name}: {acc*100:.1f}%")

        all_predictions[name] = majority_preds
        all_probs[name] = avg_probs

    # Analyze overlap
    print("\n" + "=" * 70)
    print("OVERLAP ANALYSIS")
    print("=" * 70)

    n_test = len(y_test_np)
    model_names = list(all_predictions.keys())

    # For each sample, count how many models got it right
    correct_counts = np.zeros(n_test, dtype=int)
    for name in model_names:
        correct_counts += (all_predictions[name] == y_test_np).astype(int)

    # Categorize samples
    all_correct = (correct_counts == len(model_names))
    all_wrong = (correct_counts == 0)
    some_right = ~all_correct & ~all_wrong

    print(f"\nAll models correct:   {all_correct.sum():3d} ({all_correct.sum()/n_test*100:.1f}%)")
    print(f"All models wrong:     {all_wrong.sum():3d} ({all_wrong.sum()/n_test*100:.1f}%)")
    print(f"Contested (mixed):    {some_right.sum():3d} ({some_right.sum()/n_test*100:.1f}%)")

    # Pairwise agreement
    print("\n" + "-" * 50)
    print("Pairwise Agreement (% of test set)")
    print("-" * 50)
    print(f"{'':20s}", end="")
    for name in model_names:
        print(f"{name[:8]:>10s}", end="")
    print()

    for name1 in model_names:
        print(f"{name1:20s}", end="")
        for name2 in model_names:
            agree = (all_predictions[name1] == all_predictions[name2]).mean()
            print(f"{agree*100:10.1f}", end="")
        print()

    # Which models are most complementary?
    print("\n" + "-" * 50)
    print("Complementary Analysis")
    print("-" * 50)

    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            p1 = all_predictions[name1]
            p2 = all_predictions[name2]
            y = y_test_np

            # Cases where they disagree
            disagree = p1 != p2
            # Among disagreements, who's right?
            p1_right_p2_wrong = disagree & (p1 == y)
            p2_right_p1_wrong = disagree & (p2 == y)

            if disagree.sum() > 0:
                print(f"{name1} vs {name2}:")
                print(f"  Disagree on {disagree.sum()} samples")
                print(f"  {name1} right, {name2} wrong: {p1_right_p2_wrong.sum()}")
                print(f"  {name2} right, {name1} wrong: {p2_right_p1_wrong.sum()}")

                # Oracle (pick correct one when they disagree)
                oracle_preds = np.where(p1 == y, p1, p2)
                oracle_acc = (oracle_preds == y).mean()
                base_acc = max((p1 == y).mean(), (p2 == y).mean())
                print(f"  Best individual: {base_acc*100:.1f}%, Oracle: {oracle_acc*100:.1f}%")
                print()

    # Look at the "all wrong" samples
    print("\n" + "=" * 70)
    print("SAMPLES ALL MODELS GET WRONG")
    print("=" * 70)

    wrong_indices = np.where(all_wrong)[0]
    print(f"\n{len(wrong_indices)} samples that ALL models got wrong:")

    # Show confidence levels for these
    for idx in wrong_indices[:10]:  # Show first 10
        print(f"\n  Sample {idx}: True label = {y_test_np[idx]}")
        for name in model_names:
            prob = all_probs[name][idx]
            pred = all_predictions[name][idx]
            print(f"    {name:15s}: pred={pred}, probs=[{prob[0]:.3f}, {prob[1]:.3f}]")

    # Embedding analysis for hard samples
    print("\n" + "=" * 70)
    print("EMBEDDING ANALYSIS: Easy vs Hard Samples")
    print("=" * 70)

    X_test_np = X_test.cpu().numpy()

    # Compare embedding norms/stats for easy vs hard samples
    easy_indices = np.where(all_correct)[0]
    hard_indices = np.where(all_wrong)[0]

    if len(easy_indices) > 0 and len(hard_indices) > 0:
        easy_norms = np.linalg.norm(X_test_np[easy_indices], axis=1)
        hard_norms = np.linalg.norm(X_test_np[hard_indices], axis=1)

        print(f"\nEmbedding L2 norms:")
        print(f"  Easy samples: mean={easy_norms.mean():.2f}, std={easy_norms.std():.2f}")
        print(f"  Hard samples: mean={hard_norms.mean():.2f}, std={hard_norms.std():.2f}")

        # Variance in embeddings
        easy_var = X_test_np[easy_indices].var(axis=1).mean()
        hard_var = X_test_np[hard_indices].var(axis=1).mean()
        print(f"\nEmbedding variance:")
        print(f"  Easy samples: {easy_var:.4f}")
        print(f"  Hard samples: {hard_var:.4f}")

    # Ensemble prediction
    print("\n" + "=" * 70)
    print("ENSEMBLE PREDICTIONS")
    print("=" * 70)

    # Average probabilities across all models
    avg_probs_all = np.mean([all_probs[name] for name in model_names], axis=0)
    ensemble_preds = avg_probs_all.argmax(axis=1)
    ensemble_acc = (ensemble_preds == y_test_np).mean()

    print(f"\nIndividual model accuracies:")
    for name in model_names:
        acc = (all_predictions[name] == y_test_np).mean()
        print(f"  {name:20s}: {acc*100:.1f}%")

    print(f"\nEnsemble (avg probs): {ensemble_acc*100:.1f}%")

    # Majority vote ensemble
    votes = np.stack([all_predictions[name] for name in model_names])
    majority_preds = (votes.sum(axis=0) >= len(model_names) / 2).astype(int)
    majority_acc = (majority_preds == y_test_np).mean()
    print(f"Ensemble (majority):  {majority_acc*100:.1f}%")

    # Confidence-weighted ensemble
    # Weight by how confident each model is
    weighted_probs = np.zeros_like(avg_probs_all)
    for name in model_names:
        confidence = np.abs(all_probs[name][:, 1] - 0.5)  # Distance from 50%
        weight = confidence.reshape(-1, 1)
        weighted_probs += all_probs[name] * weight
    weighted_preds = weighted_probs.argmax(axis=1)
    weighted_acc = (weighted_preds == y_test_np).mean()
    print(f"Ensemble (conf-weighted): {weighted_acc*100:.1f}%")


if __name__ == "__main__":
    run()
