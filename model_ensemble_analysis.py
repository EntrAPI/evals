#!/usr/bin/env python3
"""
Comprehensive model comparison and ensemble analysis on Twitter same-author embeddings.
All models trained on GPU. Investigates per-sample predictions and oracle ensemble accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from collections import defaultdict

EMBEDDINGS_FILE = "data/twitter_qwen_embeddings.npz"
PAIRS_FILE = "data/twitter_same_author_small.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# MODEL DEFINITIONS (all GPU-compatible PyTorch)
# ============================================================================

class Ridge(nn.Module):
    """L2 regularized logistic regression."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
    def forward(self, x):
        return self.linear(x)


class Lasso(nn.Module):
    """L1 regularized logistic regression (implemented via optimizer)."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
    def forward(self, x):
        return self.linear(x)
    def l1_loss(self):
        return sum(p.abs().sum() for p in self.parameters())


class ElasticNet(nn.Module):
    """L1 + L2 regularized logistic regression."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
    def forward(self, x):
        return self.linear(x)
    def l1_loss(self):
        return sum(p.abs().sum() for p in self.parameters())


class MLP(nn.Module):
    """Multi-layer perceptron with configurable layers."""
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class MLPWithBatchNorm(nn.Module):
    """MLP with batch normalization."""
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class TransformerClassifier(nn.Module):
    """Small transformer encoder for classification."""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: (batch, input_dim) -> treat as single token sequence
        x = self.input_proj(x).unsqueeze(1)  # (batch, 1, d_model)
        x = self.transformer(x)
        x = x.squeeze(1)  # (batch, d_model)
        return self.classifier(x)


class ResidualMLP(nn.Module):
    """MLP with residual connections."""
    def __init__(self, input_dim, hidden_dim=256, num_blocks=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_blocks)
        ])
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.relu(self.input_proj(x))
        for block in self.blocks:
            x = self.relu(x + block(x))
        return self.classifier(x)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_ridge(model, X, y, epochs=300, lr=1e-3, weight_decay=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model


def train_lasso(model, X, y, epochs=300, lr=1e-3, l1_lambda=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y) + l1_lambda * model.l1_loss()
        loss.backward()
        optimizer.step()
    return model


def train_elasticnet(model, X, y, epochs=300, lr=1e-3, l1_lambda=1e-4, weight_decay=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y) + l1_lambda * model.l1_loss()
        loss.backward()
        optimizer.step()
    return model


def train_mlp(model, X, y, epochs=300, lr=1e-3, weight_decay=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model


def get_predictions(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)
    return preds.cpu().numpy(), probs.cpu().numpy()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run():
    print("=" * 80)
    print("COMPREHENSIVE MODEL ENSEMBLE ANALYSIS")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    # Load embeddings
    print("\nLoading embeddings...")
    data = np.load(EMBEDDINGS_FILE)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"Feature dim: {X_train.shape[1]}")

    # Move to GPU
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_np = y_test

    # Normalize
    mean = X_train_t.mean(dim=0)
    std = X_train_t.std(dim=0) + 1e-8
    X_train_norm = (X_train_t - mean) / std
    X_test_norm = (X_test_t - mean) / std

    input_dim = X_train_norm.shape[1]

    # Define all models to test
    models_config = [
        # Linear models
        ("Ridge", lambda: Ridge(input_dim), train_ridge),
        ("Lasso", lambda: Lasso(input_dim), train_lasso),
        ("ElasticNet", lambda: ElasticNet(input_dim), train_elasticnet),

        # MLPs of various sizes
        ("MLP_128", lambda: MLP(input_dim, [128]), train_mlp),
        ("MLP_256", lambda: MLP(input_dim, [256]), train_mlp),
        ("MLP_512", lambda: MLP(input_dim, [512]), train_mlp),
        ("MLP_256_128", lambda: MLP(input_dim, [256, 128]), train_mlp),
        ("MLP_512_256", lambda: MLP(input_dim, [512, 256]), train_mlp),
        ("MLP_512_256_128", lambda: MLP(input_dim, [512, 256, 128]), train_mlp),

        # MLPs with BatchNorm
        ("MLP_BN_256", lambda: MLPWithBatchNorm(input_dim, [256]), train_mlp),
        ("MLP_BN_512_256", lambda: MLPWithBatchNorm(input_dim, [512, 256]), train_mlp),

        # Residual MLP
        ("ResMLP_256", lambda: ResidualMLP(input_dim, 256, num_blocks=2), train_mlp),
        ("ResMLP_512", lambda: ResidualMLP(input_dim, 512, num_blocks=2), train_mlp),

        # Transformer
        ("Transformer_128", lambda: TransformerClassifier(input_dim, d_model=128, nhead=4, num_layers=2), train_mlp),
        ("Transformer_256", lambda: TransformerClassifier(input_dim, d_model=256, nhead=4, num_layers=2), train_mlp),
    ]

    # Train all models (3 runs each for stability)
    print("\n" + "=" * 80)
    print("TRAINING ALL MODELS (3 runs each)")
    print("=" * 80)

    all_results = {}
    all_preds = {}
    all_probs = {}

    for name, model_fn, train_fn in models_config:
        print(f"\n{name}:", end=" ")

        preds_runs = []
        probs_runs = []
        accs = []

        for run_i in range(3):
            torch.manual_seed(42 + run_i)
            model = model_fn().to(DEVICE)
            train_fn(model, X_train_norm, y_train_t)
            preds, probs = get_predictions(model, X_test_norm)
            preds_runs.append(preds)
            probs_runs.append(probs)
            acc = (preds == y_test_np).mean()
            accs.append(acc)

        # Majority vote predictions
        majority_preds = (np.stack(preds_runs).sum(axis=0) >= 2).astype(int)
        avg_probs = np.stack(probs_runs).mean(axis=0)
        majority_acc = (majority_preds == y_test_np).mean()

        all_results[name] = {
            'accs': accs,
            'mean': np.mean(accs),
            'std': np.std(accs),
            'majority_acc': majority_acc,
        }
        all_preds[name] = majority_preds
        all_probs[name] = avg_probs

        print(f"{majority_acc*100:.1f}% (runs: {[f'{a*100:.1f}' for a in accs]})")

    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (sorted by accuracy)")
    print("=" * 80)

    sorted_models = sorted(all_results.items(), key=lambda x: -x[1]['majority_acc'])

    print(f"\n{'Model':<20} {'Majority':<10} {'Mean±Std':<15}")
    print("-" * 45)
    for name, res in sorted_models:
        print(f"{name:<20} {res['majority_acc']*100:>6.1f}%    {res['mean']*100:.1f}% ± {res['std']*100:.1f}%")

    # ========================================================================
    # PER-SAMPLE ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("PER-SAMPLE AGREEMENT ANALYSIS")
    print("=" * 80)

    model_names = list(all_preds.keys())
    n_models = len(model_names)
    n_test = len(y_test_np)

    # Count how many models got each sample correct
    correct_matrix = np.array([all_preds[name] == y_test_np for name in model_names])  # (n_models, n_test)
    correct_counts = correct_matrix.sum(axis=0)  # how many models got each sample right

    print(f"\nDistribution of correct model counts per sample:")
    for i in range(n_models + 1):
        count = (correct_counts == i).sum()
        pct = count / n_test * 100
        bar = "█" * int(pct / 2)
        print(f"  {i:2d} models correct: {count:3d} ({pct:5.1f}%) {bar}")

    all_correct = (correct_counts == n_models).sum()
    all_wrong = (correct_counts == 0).sum()
    some_right = n_test - all_correct - all_wrong

    print(f"\n  All {n_models} models correct: {all_correct} ({all_correct/n_test*100:.1f}%)")
    print(f"  All {n_models} models wrong:   {all_wrong} ({all_wrong/n_test*100:.1f}%)")
    print(f"  Mixed (some right/wrong):  {some_right} ({some_right/n_test*100:.1f}%)")

    # ========================================================================
    # ORACLE ENSEMBLE ACCURACY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ORACLE ENSEMBLE ANALYSIS")
    print("=" * 80)

    # Oracle: if we could pick the best model per sample
    oracle_correct = (correct_counts >= 1).sum()
    oracle_acc = oracle_correct / n_test

    print(f"\nOracle accuracy (at least 1 model correct): {oracle_acc*100:.1f}%")
    print(f"  This is the ceiling if we could perfectly pick the right model per sample.")

    # Best single model
    best_model_name = sorted_models[0][0]
    best_model_acc = sorted_models[0][1]['majority_acc']
    print(f"\nBest single model: {best_model_name} at {best_model_acc*100:.1f}%")
    print(f"Oracle improvement over best: +{(oracle_acc - best_model_acc)*100:.1f}pp")

    # Majority voting ensemble
    ensemble_preds = (np.stack([all_preds[name] for name in model_names]).sum(axis=0) > n_models/2).astype(int)
    ensemble_acc = (ensemble_preds == y_test_np).mean()
    print(f"\nMajority voting ensemble ({n_models} models): {ensemble_acc*100:.1f}%")

    # Weighted ensemble (by accuracy)
    weights = np.array([all_results[name]['majority_acc'] for name in model_names])
    weights = weights / weights.sum()
    weighted_probs = sum(all_probs[name] * w for name, w in zip(model_names, weights))
    weighted_preds = weighted_probs.argmax(axis=1)
    weighted_acc = (weighted_preds == y_test_np).mean()
    print(f"Accuracy-weighted ensemble: {weighted_acc*100:.1f}%")

    # Top-5 ensemble
    top5_names = [name for name, _ in sorted_models[:5]]
    top5_preds = (np.stack([all_preds[name] for name in top5_names]).sum(axis=0) >= 3).astype(int)
    top5_acc = (top5_preds == y_test_np).mean()
    print(f"Top-5 majority ensemble: {top5_acc*100:.1f}%")

    # ========================================================================
    # MODEL DISAGREEMENT ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODEL DISAGREEMENT PATTERNS")
    print("=" * 80)

    # Find samples where models disagree most
    pred_matrix = np.array([all_preds[name] for name in model_names])  # (n_models, n_test)
    disagreement = pred_matrix.std(axis=0)  # higher = more disagreement

    high_disagree_idx = np.argsort(-disagreement)[:10]

    print("\nTop 10 most contested samples (highest model disagreement):")

    # Load pairs for context
    with open(PAIRS_FILE, 'r') as f:
        pairs_data = json.load(f)
    test_pairs = pairs_data['test']

    for rank, idx in enumerate(high_disagree_idx):
        pair = test_pairs[idx]
        votes_for_1 = pred_matrix[:, idx].sum()
        votes_for_0 = n_models - votes_for_1
        true_label = y_test_np[idx]
        n_correct = correct_counts[idx]

        print(f"\n  #{rank+1} Sample {idx}: label={true_label}, {n_correct}/{n_models} correct")
        print(f"      Votes: {votes_for_0} for A, {votes_for_1} for B")
        print(f"      Ratio: {pair['ratio']:.1f}x, Eng: A={pair['engagement_a']}, B={pair['engagement_b']}")
        print(f"      A: {pair['tweet_a']['text'][:70]}...")
        print(f"      B: {pair['tweet_b']['text'][:70]}...")

    # ========================================================================
    # HARD SAMPLES (ALL MODELS WRONG)
    # ========================================================================
    print("\n" + "=" * 80)
    print("HARD SAMPLES (ALL MODELS WRONG)")
    print("=" * 80)

    hard_indices = np.where(correct_counts == 0)[0]
    print(f"\n{len(hard_indices)} samples where ALL {n_models} models are wrong:")

    for idx in hard_indices[:10]:
        pair = test_pairs[idx]
        true_label = y_test_np[idx]
        winner = "A" if true_label == 0 else "B"

        print(f"\n  Sample {idx}: ratio={pair['ratio']:.1f}x, winner={winner}")
        print(f"      Eng: A={pair['engagement_a']}, B={pair['engagement_b']}")
        print(f"      A: {pair['tweet_a']['text'][:80]}...")
        print(f"      B: {pair['tweet_b']['text'][:80]}...")

    # ========================================================================
    # EASY SAMPLES (ALL MODELS CORRECT)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EASY SAMPLES ANALYSIS")
    print("=" * 80)

    easy_indices = np.where(correct_counts == n_models)[0]
    hard_indices = np.where(correct_counts == 0)[0]

    # Compare characteristics
    easy_ratios = [test_pairs[i]['ratio'] for i in easy_indices]
    hard_ratios = [test_pairs[i]['ratio'] for i in hard_indices]

    print(f"\nEasy samples ({len(easy_indices)}):")
    print(f"  Mean engagement ratio: {np.mean(easy_ratios):.1f}x")
    print(f"  Median ratio: {np.median(easy_ratios):.1f}x")

    print(f"\nHard samples ({len(hard_indices)}):")
    print(f"  Mean engagement ratio: {np.mean(hard_ratios):.1f}x")
    print(f"  Median ratio: {np.median(hard_ratios):.1f}x")

    # ========================================================================
    # MODEL CORRELATION MATRIX
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODEL PREDICTION CORRELATION")
    print("=" * 80)

    # Calculate pairwise agreement between models
    agreement_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            agreement_matrix[i, j] = (pred_matrix[i] == pred_matrix[j]).mean()

    print("\nPairwise prediction agreement (showing models that disagree most):")

    # Find most different model pairs
    disagreements = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            disagreements.append((model_names[i], model_names[j], agreement_matrix[i, j]))

    disagreements.sort(key=lambda x: x[2])

    print("\nMost different model pairs:")
    for m1, m2, agree in disagreements[:5]:
        print(f"  {m1} vs {m2}: {agree*100:.1f}% agreement")

    print("\nMost similar model pairs:")
    for m1, m2, agree in disagreements[-5:]:
        print(f"  {m1} vs {m2}: {agree*100:.1f}% agreement")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n  Dataset: {n_test} test samples (same-author Twitter pairs)")
    print(f"  Models tested: {n_models}")
    print(f"\n  Best single model: {best_model_name} ({best_model_acc*100:.1f}%)")
    print(f"  Majority ensemble: {ensemble_acc*100:.1f}%")
    print(f"  Weighted ensemble: {weighted_acc*100:.1f}%")
    print(f"  Oracle (perfect selection): {oracle_acc*100:.1f}%")
    print(f"\n  All models correct: {all_correct}/{n_test} ({all_correct/n_test*100:.1f}%)")
    print(f"  All models wrong: {all_wrong}/{n_test} ({all_wrong/n_test*100:.1f}%)")
    print(f"\n  Potential ensemble gain: +{(oracle_acc - best_model_acc)*100:.1f}pp")
    print(f"  Hard cases (irreducible?): {all_wrong/n_test*100:.1f}%")


if __name__ == "__main__":
    run()
