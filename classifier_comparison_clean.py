#!/usr/bin/env python3
"""
Comprehensive classifier comparison on clean (no-leakage) dataset.
All models trained on GPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from scipy import stats

EMBEDDINGS_FILE = "data/twitter_no_replies_embeddings.npz"
PAIRS_FILE = "data/twitter_clean_no_replies.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class LogisticRegression(nn.Module):
    """Simple logistic regression (linear classifier)."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


class Ridge(nn.Module):
    """L2 regularized logistic regression."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


class Lasso(nn.Module):
    """L1 regularized logistic regression."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)

    def l1_loss(self):
        return sum(p.abs().sum() for p in self.parameters())


class MLP(nn.Module):
    """Multi-layer perceptron."""
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


class MLPBatchNorm(nn.Module):
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


class ResidualBlock(nn.Module):
    """Residual block for ResNet-style MLP."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResidualMLP(nn.Module):
    """MLP with residual connections."""
    def __init__(self, input_dim, hidden_dim=256, num_blocks=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_proj(x))
        x = self.blocks(x)
        return self.classifier(x)


class TransformerClassifier(nn.Module):
    """Transformer encoder for classification."""
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
        x = self.input_proj(x).unsqueeze(1)  # (batch, 1, d_model)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)


class DeepMLP(nn.Module):
    """Deeper MLP with more layers."""
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class WideMLP(nn.Module):
    """Wide but shallow MLP."""
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_standard(model, X, y, epochs=300, lr=1e-3, weight_decay=1e-4):
    """Standard training with AdamW."""
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


def train_lasso(model, X, y, epochs=300, lr=1e-3, l1_lambda=1e-4):
    """Training with L1 regularization."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y) + l1_lambda * model.l1_loss()
        loss.backward()
        optimizer.step()

    return model


def train_with_warmup(model, X, y, epochs=300, lr=1e-3, weight_decay=1e-4, warmup=20):
    """Training with linear warmup."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Linear warmup
        if epoch < warmup:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / warmup

        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    return model


def get_predictions(model, X):
    """Get predictions from model."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
    return preds.cpu().numpy()


# ============================================================================
# MAIN
# ============================================================================

def run():
    print("=" * 80)
    print("CLASSIFIER COMPARISON ON CLEAN DATASET")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    # Load data
    print("\nLoading data...")
    data = np.load(EMBEDDINGS_FILE)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    with open(PAIRS_FILE) as f:
        pairs_data = json.load(f)
    test_pairs = pairs_data['test']

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    print("Split by user - NO DATA LEAKAGE")

    # Prepare tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    # Normalize
    mean = X_train_t.mean(dim=0)
    std = X_train_t.std(dim=0) + 1e-8
    X_train_norm = (X_train_t - mean) / std
    X_test_norm = (X_test_t - mean) / std

    input_dim = X_train_norm.shape[1]
    print(f"Feature dimension: {input_dim}")

    # Define all models to test
    models_config = [
        # Linear models
        ("LogReg", lambda: LogisticRegression(input_dim), train_standard, {"weight_decay": 0}),
        ("Ridge (L2)", lambda: Ridge(input_dim), train_standard, {"weight_decay": 1e-2}),
        ("Lasso (L1)", lambda: Lasso(input_dim), train_lasso, {}),

        # Simple MLPs
        ("MLP_64", lambda: MLP(input_dim, [64]), train_standard, {}),
        ("MLP_128", lambda: MLP(input_dim, [128]), train_standard, {}),
        ("MLP_256", lambda: MLP(input_dim, [256]), train_standard, {}),
        ("MLP_512", lambda: MLP(input_dim, [512]), train_standard, {}),

        # Deeper MLPs
        ("MLP_128_64", lambda: MLP(input_dim, [128, 64]), train_standard, {}),
        ("MLP_256_128", lambda: MLP(input_dim, [256, 128]), train_standard, {}),
        ("MLP_512_256", lambda: MLP(input_dim, [512, 256]), train_standard, {}),
        ("MLP_512_256_128", lambda: MLP(input_dim, [512, 256, 128]), train_standard, {}),

        # MLPs with BatchNorm
        ("MLP_BN_256", lambda: MLPBatchNorm(input_dim, [256]), train_standard, {}),
        ("MLP_BN_512_256", lambda: MLPBatchNorm(input_dim, [512, 256]), train_standard, {}),

        # Wide/Deep variants
        ("WideMLP_1024", lambda: WideMLP(input_dim), train_standard, {}),
        ("DeepMLP", lambda: DeepMLP(input_dim), train_standard, {}),

        # Residual MLPs
        ("ResMLP_256_2blk", lambda: ResidualMLP(input_dim, 256, 2), train_standard, {}),
        ("ResMLP_256_4blk", lambda: ResidualMLP(input_dim, 256, 4), train_standard, {}),
        ("ResMLP_512_2blk", lambda: ResidualMLP(input_dim, 512, 2), train_standard, {}),

        # Transformers
        ("Transformer_64", lambda: TransformerClassifier(input_dim, d_model=64, nhead=4, num_layers=1), train_with_warmup, {}),
        ("Transformer_128", lambda: TransformerClassifier(input_dim, d_model=128, nhead=4, num_layers=2), train_with_warmup, {}),
        ("Transformer_256", lambda: TransformerClassifier(input_dim, d_model=256, nhead=4, num_layers=2), train_with_warmup, {}),
    ]

    # Train and evaluate all models
    print("\n" + "=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)

    results = {}
    all_preds = {}

    for name, model_fn, train_fn, train_kwargs in models_config:
        print(f"\n{name}:", end=" ", flush=True)

        # Run 3 times for stability
        run_accs = []
        run_preds = []

        for seed in range(3):
            torch.manual_seed(42 + seed)
            model = model_fn().to(DEVICE)
            train_fn(model, X_train_norm, y_train_t, **train_kwargs)
            preds = get_predictions(model, X_test_norm)
            acc = (preds == y_test).mean()
            run_accs.append(acc)
            run_preds.append(preds)

        # Majority vote
        majority_preds = (np.stack(run_preds).sum(axis=0) >= 2).astype(int)
        majority_acc = (majority_preds == y_test).mean()

        results[name] = {
            'accs': run_accs,
            'mean': np.mean(run_accs),
            'std': np.std(run_accs),
            'majority': majority_acc,
        }
        all_preds[name] = majority_preds

        print(f"{majority_acc*100:.1f}% (runs: {[f'{a*100:.1f}' for a in run_accs]})")

    # Results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (sorted by accuracy)")
    print("=" * 80)

    sorted_results = sorted(results.items(), key=lambda x: -x[1]['majority'])

    print(f"\n{'Model':<22} {'Majority':<10} {'Mean±Std':<15} {'Runs'}")
    print("-" * 75)
    for name, res in sorted_results:
        runs_str = ", ".join([f"{a*100:.1f}" for a in res['accs']])
        print(f"{name:<22} {res['majority']*100:>6.1f}%    {res['mean']*100:.1f}% ± {res['std']*100:.1f}%    [{runs_str}]")

    # Best vs worst
    best_name, best_res = sorted_results[0]
    worst_name, worst_res = sorted_results[-1]

    print(f"\nBest: {best_name} ({best_res['majority']*100:.1f}%)")
    print(f"Worst: {worst_name} ({worst_res['majority']*100:.1f}%)")
    print(f"Spread: {(best_res['majority'] - worst_res['majority'])*100:.1f}pp")

    # Agreement analysis
    print("\n" + "=" * 80)
    print("MODEL AGREEMENT ANALYSIS")
    print("=" * 80)

    model_names = list(all_preds.keys())
    n_models = len(model_names)

    correct_matrix = np.array([all_preds[name] == y_test for name in model_names])
    correct_counts = correct_matrix.sum(axis=0)

    all_correct = (correct_counts == n_models).sum()
    all_wrong = (correct_counts == 0).sum()

    print(f"\nAll {n_models} models correct: {all_correct} ({all_correct/len(y_test)*100:.1f}%)")
    print(f"All {n_models} models wrong: {all_wrong} ({all_wrong/len(y_test)*100:.1f}%)")

    # Oracle accuracy
    oracle_correct = (correct_counts >= 1).sum()
    oracle_acc = oracle_correct / len(y_test)
    print(f"Oracle (at least 1 correct): {oracle_acc*100:.1f}%")

    # Analysis by ratio
    print("\n" + "=" * 80)
    print("BEST MODEL ACCURACY BY RATIO")
    print("=" * 80)

    ratios = np.array([p['ratio'] for p in test_pairs])
    best_preds = all_preds[best_name]
    best_correct = (best_preds == y_test)

    print(f"\nUsing best model: {best_name}")
    for low, high in [(1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 100)]:
        mask = (ratios >= low) & (ratios < high)
        n = mask.sum()
        if n > 0:
            acc = best_correct[mask].mean()
            print(f"  {low:.1f}x - {high:.1f}x: {acc*100:.1f}% (n={n})")

    # Random baseline comparison
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nRandom baseline: 50.0%")
    print(f"Best model: {best_name} at {best_res['majority']*100:.1f}%")
    print(f"Improvement over random: +{(best_res['majority'] - 0.5)*100:.1f}pp")
    print(f"Oracle ceiling: {oracle_acc*100:.1f}%")


if __name__ == "__main__":
    run()
