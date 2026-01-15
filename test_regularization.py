#!/usr/bin/env python3
"""
Test Ridge vs Lasso vs Elastic Net on embeddings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

EMBEDDINGS_FILE = "data/gemma_1b_embeddings.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    """Load embeddings."""
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    data = np.load(EMBEDDINGS_FILE)

    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32, device=DEVICE)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=DEVICE)

    # Normalize
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(f"Data: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, y_train, X_test, y_test


def pca_reduce(X_train, X_test, n_components):
    """Apply PCA dimensionality reduction."""
    mean = X_train.mean(dim=0)
    X_train_centered = X_train - mean
    X_test_centered = X_test - mean

    U, S, Vh = torch.linalg.svd(X_train_centered, full_matrices=False)
    components = Vh[:n_components]

    X_train_pca = X_train_centered @ components.T
    X_test_pca = X_test_centered @ components.T

    return X_train_pca, X_test_pca


def random_project_relu(X, output_dim, seed=42):
    """Project X through random matrix with ReLU."""
    torch.manual_seed(seed)
    input_dim = X.shape[1]
    W = torch.randn(input_dim, output_dim, device=DEVICE) / np.sqrt(input_dim)
    return torch.relu(X @ W)


def train_logreg(X_train, y_train, X_test, y_test, epochs=300,
                 l1_lambda=0.0, l2_lambda=0.0, reg_type='none'):
    """Train logistic regression with different regularization."""
    input_dim = X_train.shape[1]
    model = nn.Linear(input_dim, 2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)

        # Add regularization
        if reg_type == 'ridge' or reg_type == 'elastic':
            l2_loss = l2_lambda * (model.weight ** 2).sum()
            loss = loss + l2_loss

        if reg_type == 'lasso' or reg_type == 'elastic':
            l1_loss = l1_lambda * model.weight.abs().sum()
            loss = loss + l1_loss

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
        sparsity = (model.weight.abs() < 1e-4).float().mean().item()

    return test_acc, train_acc, sparsity


def run():
    print("=" * 70)
    print("Ridge vs Lasso vs Elastic Net")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    X_train, y_train, X_test, y_test = load_data()

    # Apply PCA 256 -> ReLU 16x (our best config)
    print("\nApplying PCA 256 -> ReLU 16x...")
    X_train_pca, X_test_pca = pca_reduce(X_train, X_test, 256)
    X_train_proj = random_project_relu(X_train_pca, 256 * 16)
    X_test_proj = random_project_relu(X_test_pca, 256 * 16)
    print(f"Final shape: {X_train_proj.shape}")

    results = {}

    # No regularization
    print("\n" + "=" * 60)
    print("No Regularization")
    print("=" * 60)
    test_acc, train_acc, _ = train_logreg(X_train_proj, y_train, X_test_proj, y_test, reg_type='none')
    print(f"  Train: {train_acc*100:.1f}%, Test: {test_acc*100:.1f}%")
    results['none'] = test_acc

    # Ridge (L2)
    print("\n" + "=" * 60)
    print("Ridge (L2)")
    print("=" * 60)
    for l2 in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        test_acc, train_acc, _ = train_logreg(
            X_train_proj, y_train, X_test_proj, y_test,
            l2_lambda=l2, reg_type='ridge'
        )
        print(f"  L2={l2:.0e}: Train {train_acc*100:.1f}%, Test {test_acc*100:.1f}%")
        results[f'ridge_{l2:.0e}'] = test_acc

    # Lasso (L1)
    print("\n" + "=" * 60)
    print("Lasso (L1)")
    print("=" * 60)
    for l1 in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        test_acc, train_acc, sparsity = train_logreg(
            X_train_proj, y_train, X_test_proj, y_test,
            l1_lambda=l1, reg_type='lasso'
        )
        print(f"  L1={l1:.0e}: Train {train_acc*100:.1f}%, Test {test_acc*100:.1f}%, Sparsity {sparsity*100:.1f}%")
        results[f'lasso_{l1:.0e}'] = test_acc

    # Elastic Net (L1 + L2)
    print("\n" + "=" * 60)
    print("Elastic Net (L1 + L2)")
    print("=" * 60)
    for l1 in [1e-4, 1e-3, 1e-2]:
        for l2 in [1e-4, 1e-3, 1e-2]:
            test_acc, train_acc, sparsity = train_logreg(
                X_train_proj, y_train, X_test_proj, y_test,
                l1_lambda=l1, l2_lambda=l2, reg_type='elastic'
            )
            print(f"  L1={l1:.0e}, L2={l2:.0e}: Train {train_acc*100:.1f}%, Test {test_acc*100:.1f}%, Sparsity {sparsity*100:.1f}%")
            results[f'elastic_{l1:.0e}_{l2:.0e}'] = test_acc

    # Summary - top 10
    print("\n" + "=" * 70)
    print("TOP 10 RESULTS")
    print("=" * 70)
    for i, (name, acc) in enumerate(sorted(results.items(), key=lambda x: -x[1])[:10]):
        bar = "█" * int(acc * 50)
        print(f"  {name:25s}: {acc*100:.1f}% {bar}")

    print(f"\n  Random baseline: 50.0%")


if __name__ == "__main__":
    run()
