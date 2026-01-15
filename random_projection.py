#!/usr/bin/env python3
"""
Test logistic regression on embeddings with PCA and random projection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

EMBEDDINGS_FILE = "data/gemma_1b_embeddings.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pca_reduce(X_train, X_test, n_components):
    """Apply PCA dimensionality reduction."""
    # Center the data
    mean = X_train.mean(dim=0)
    X_train_centered = X_train - mean
    X_test_centered = X_test - mean

    # Compute covariance matrix and eigendecomposition
    # Use SVD for numerical stability
    U, S, Vh = torch.linalg.svd(X_train_centered, full_matrices=False)

    # Take top n_components
    components = Vh[:n_components]  # (n_components, input_dim)

    # Project
    X_train_pca = X_train_centered @ components.T
    X_test_pca = X_test_centered @ components.T

    # Variance explained
    var_explained = (S[:n_components] ** 2).sum() / (S ** 2).sum()

    return X_train_pca, X_test_pca, var_explained.item()


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


def train_logreg(X_train, y_train, X_test, y_test, epochs=200,
                  l1_lambda=0.0, l2_lambda=1e-4, reg_type='ridge'):
    """Train logistic regression with different regularization."""
    input_dim = X_train.shape[1]
    model = nn.Linear(input_dim, 2).to(DEVICE)

    # Set weight decay based on reg type
    if reg_type == 'ridge':
        weight_decay = l2_lambda
    else:
        weight_decay = 0  # Handle L1/elastic manually

    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)

        # Add L1 penalty for lasso/elastic
        if reg_type in ['lasso', 'elastic']:
            l1_loss = l1_lambda * model.weight.abs().sum()
            loss = loss + l1_loss

        # Add L2 penalty for elastic (since we disabled weight_decay)
        if reg_type == 'elastic':
            l2_loss = l2_lambda * (model.weight ** 2).sum()
            loss = loss + l2_loss

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
        # Count zero weights for lasso
        sparsity = (model.weight.abs() < 1e-6).float().mean().item()

    return test_acc, train_acc, sparsity


def random_project(X, output_dim, seed=42):
    """Project X through a random matrix to output_dim dimensions."""
    torch.manual_seed(seed)
    input_dim = X.shape[1]
    # Random matrix with proper scaling (like Xavier init)
    W = torch.randn(input_dim, output_dim, device=DEVICE) / np.sqrt(input_dim)
    return X @ W


def run():
    print("=" * 70)
    print("Random Projection + Logistic Regression")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    X_train, y_train, X_test, y_test = load_data()
    input_dim = X_train.shape[1]

    results = {}

    # Baseline: no projection
    print("\n" + "-" * 60)
    print(f"Baseline (no projection): {input_dim} dims")
    print("-" * 60)
    test_acc, train_acc = train_logreg(X_train, y_train, X_test, y_test)
    print(f"  Train: {train_acc*100:.1f}%, Test: {test_acc*100:.1f}%")
    results[f"baseline_{input_dim}"] = test_acc

    # Test various expansion sizes
    multipliers = [2, 4, 8, 16]

    for mult in multipliers:
        output_dim = input_dim * mult
        print("\n" + "-" * 60)
        print(f"Random projection: {input_dim} -> {output_dim} dims ({mult}x)")
        print("-" * 60)

        # Project
        X_train_proj = random_project(X_train, output_dim)
        X_test_proj = random_project(X_test, output_dim)

        # Train
        test_acc, train_acc = train_logreg(X_train_proj, y_train, X_test_proj, y_test)
        print(f"  Train: {train_acc*100:.1f}%, Test: {test_acc*100:.1f}%")
        results[f"random_{mult}x"] = test_acc

    # Also try with ReLU nonlinearity (random features)
    print("\n" + "=" * 60)
    print("With ReLU nonlinearity (Random Features)")
    print("=" * 60)

    for mult in [2, 4, 8]:
        output_dim = input_dim * mult
        print("\n" + "-" * 60)
        print(f"Random features (ReLU): {input_dim} -> {output_dim} dims ({mult}x)")
        print("-" * 60)

        # Project with ReLU
        X_train_proj = torch.relu(random_project(X_train, output_dim))
        X_test_proj = torch.relu(random_project(X_test, output_dim))

        # Train
        test_acc, train_acc = train_logreg(X_train_proj, y_train, X_test_proj, y_test)
        print(f"  Train: {train_acc*100:.1f}%, Test: {test_acc*100:.1f}%")
        results[f"relu_{mult}x"] = test_acc

    # PCA experiments
    print("\n" + "=" * 60)
    print("PCA Dimensionality Reduction")
    print("=" * 60)

    for n_comp in [32, 64, 128, 256, 512, 1024]:
        print("\n" + "-" * 60)
        X_train_pca, X_test_pca, var_exp = pca_reduce(X_train, X_test, n_comp)
        print(f"PCA: {input_dim} -> {n_comp} dims (var explained: {var_exp*100:.1f}%)")
        print("-" * 60)

        test_acc, train_acc = train_logreg(X_train_pca, y_train, X_test_pca, y_test)
        print(f"  Train: {train_acc*100:.1f}%, Test: {test_acc*100:.1f}%")
        results[f"pca_{n_comp}"] = test_acc

    # PCA + ReLU random features
    print("\n" + "=" * 60)
    print("PCA + ReLU Random Features")
    print("=" * 60)

    for n_comp in [64, 128, 256, 512]:
        X_train_pca, X_test_pca, var_exp = pca_reduce(X_train, X_test, n_comp)

        for mult in [4, 8, 16]:
            output_dim = n_comp * mult
            print("\n" + "-" * 60)
            print(f"PCA {n_comp} -> ReLU {mult}x -> {output_dim} dims")
            print("-" * 60)

            X_train_proj = torch.relu(random_project(X_train_pca, output_dim))
            X_test_proj = torch.relu(random_project(X_test_pca, output_dim))

            test_acc, train_acc = train_logreg(X_train_proj, y_train, X_test_proj, y_test)
            print(f"  Train: {train_acc*100:.1f}%, Test: {test_acc*100:.1f}%")
            results[f"pca{n_comp}_relu{mult}x"] = test_acc

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 50)
        print(f"  {name:20s}: {acc*100:.1f}% {bar}")

    print(f"\n  Random baseline: 50.0%")


if __name__ == "__main__":
    run()
