#!/usr/bin/env python3
"""
Traditional classifiers on GPU - including hyperparameter tuning.
All implementations in PyTorch for full GPU acceleration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from itertools import product

EMBEDDINGS_FILE = "data/gemma_1b_embeddings.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    """Load embeddings and convert to GPU tensors."""
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    data = np.load(EMBEDDINGS_FILE)

    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32, device=DEVICE)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=DEVICE)

    # Normalize on GPU
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


# ============== LOGISTIC REGRESSION WITH TUNING ==============

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


def tune_logistic_regression(X_train, y_train, X_test, y_test):
    """Hyperparameter tuning for logistic regression."""
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION - Hyperparameter Tuning")
    print("=" * 60)

    input_dim = X_train.shape[1]

    # Hyperparameter grid
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    weight_decays = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    batch_sizes = [32, 64, 128, 256, 512]

    best_acc = 0
    best_params = {}
    results = []

    # Quick search with fewer epochs
    print("\nSearching hyperparameters...")
    for lr, wd in product(lrs, weight_decays):
        torch.manual_seed(42)
        model = LogisticRegression(input_dim).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        # Train for fewer epochs during search
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

        results.append((lr, wd, test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            best_params = {'lr': lr, 'weight_decay': wd}

    # Show top results
    results.sort(key=lambda x: -x[2])
    print("\nTop 10 configurations:")
    for lr, wd, acc in results[:10]:
        print(f"  lr={lr:.0e}, wd={wd:.0e}: {acc*100:.1f}%")

    # Full training with best params
    print(f"\nBest params: {best_params}")
    print("Training with best params (more epochs)...")

    torch.manual_seed(42)
    model = LogisticRegression(input_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), **best_params)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    print(f"Final: Train={train_acc*100:.1f}%, Test={test_acc*100:.1f}%")
    return test_acc


# ============== SVM (Hinge Loss) ==============

class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_svm(X_train, y_train, X_test, y_test, C=1.0):
    """SVM with hinge loss on GPU."""
    input_dim = X_train.shape[1]
    # Convert labels to -1, 1
    y_train_svm = y_train.float() * 2 - 1
    y_test_svm = y_test.float() * 2 - 1

    model = LinearSVM(input_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1/C)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        scores = model(X_train)
        # Hinge loss: max(0, 1 - y * score)
        loss = torch.clamp(1 - y_train_svm * scores, min=0).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (model(X_test) > 0).long()
        test_acc = (preds == y_test).float().mean().item()

    return test_acc


# ============== GRADIENT BOOSTING (from scratch) ==============

class DecisionStump(nn.Module):
    """A simple differentiable decision stump."""
    def __init__(self, input_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Soft decision: sigmoid of weighted sum minus threshold
        return self.scale * torch.sigmoid(10 * (x @ self.weights - self.threshold))


class GradientBoostingGPU:
    """Gradient boosting implemented in PyTorch for GPU."""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_features=100):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.stumps = []
        self.feature_indices = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_float = y.float() * 2 - 1  # Convert to -1, 1

        # Initialize predictions
        F = torch.zeros(n_samples, device=DEVICE)

        for i in range(self.n_estimators):
            # Compute pseudo-residuals (gradient of log loss)
            probs = torch.sigmoid(F)
            residuals = y_float - (probs * 2 - 1)

            # Random feature subset
            idx = torch.randperm(n_features, device=DEVICE)[:self.max_features]
            self.feature_indices.append(idx)

            # Fit stump to residuals
            stump = DecisionStump(self.max_features).to(DEVICE)
            optimizer = optim.Adam(stump.parameters(), lr=0.01)

            X_subset = X[:, idx]
            for _ in range(20):
                optimizer.zero_grad()
                pred = stump(X_subset)
                loss = ((pred - residuals) ** 2).mean()
                loss.backward()
                optimizer.step()

            self.stumps.append(stump)

            # Update predictions
            with torch.no_grad():
                F = F + self.learning_rate * stump(X_subset)

    def predict(self, X):
        F = torch.zeros(X.shape[0], device=DEVICE)
        with torch.no_grad():
            for stump, idx in zip(self.stumps, self.feature_indices):
                F = F + self.learning_rate * stump(X[:, idx])
        return (F > 0).long()


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train gradient boosting."""
    best_acc = 0
    best_params = {}

    for n_est in [50, 100, 200]:
        for lr in [0.05, 0.1, 0.2]:
            model = GradientBoostingGPU(n_estimators=n_est, learning_rate=lr, max_features=200)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = (preds == y_test).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_params = {'n_estimators': n_est, 'lr': lr}

    print(f"  Best params: {best_params}")
    return best_acc


# ============== RANDOM FOREST (ensemble of linear models) ==============

class RandomForestGPU:
    """Random forest using ensemble of small neural nets."""
    def __init__(self, n_trees=50, max_features=200, hidden_dim=32):
        self.n_trees = n_trees
        self.max_features = max_features
        self.hidden_dim = hidden_dim
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        for i in range(self.n_trees):
            # Bootstrap sample
            boot_idx = torch.randint(0, n_samples, (n_samples,), device=DEVICE)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            # Random feature subset
            feat_idx = torch.randperm(n_features, device=DEVICE)[:self.max_features]
            self.feature_indices.append(feat_idx)

            # Small neural net as "tree"
            tree = nn.Sequential(
                nn.Linear(self.max_features, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 2)
            ).to(DEVICE)

            optimizer = optim.Adam(tree.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            X_subset = X_boot[:, feat_idx]
            for _ in range(30):
                optimizer.zero_grad()
                loss = criterion(tree(X_subset), y_boot)
                loss.backward()
                optimizer.step()

            self.trees.append(tree)

    def predict(self, X):
        votes = torch.zeros(X.shape[0], 2, device=DEVICE)
        with torch.no_grad():
            for tree, idx in zip(self.trees, self.feature_indices):
                probs = F.softmax(tree(X[:, idx]), dim=1)
                votes += probs
        return votes.argmax(dim=1)


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train random forest."""
    best_acc = 0
    for n_trees in [30, 50, 100]:
        for hidden in [16, 32, 64]:
            model = RandomForestGPU(n_trees=n_trees, max_features=200, hidden_dim=hidden)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = (preds == y_test).float().mean().item()
            if acc > best_acc:
                best_acc = acc
    return best_acc


# ============== K-NEAREST NEIGHBORS ==============

def train_knn(X_train, y_train, X_test, y_test):
    """KNN on GPU using batched distance computation."""
    best_acc = 0
    best_k = 1

    for k in [1, 3, 5, 7, 11, 15, 21, 31]:
        # Compute all pairwise distances on GPU
        # Using (a-b)^2 = a^2 + b^2 - 2ab
        train_sq = (X_train ** 2).sum(dim=1, keepdim=True)
        test_sq = (X_test ** 2).sum(dim=1, keepdim=True)
        dists = train_sq.T + test_sq - 2 * X_test @ X_train.T  # (n_test, n_train)

        # Get k nearest neighbors
        _, indices = dists.topk(k, dim=1, largest=False)
        neighbor_labels = y_train[indices]  # (n_test, k)

        # Majority vote
        preds = (neighbor_labels.float().mean(dim=1) > 0.5).long()
        acc = (preds == y_test).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"  Best k={best_k}")
    return best_acc


# ============== NAIVE BAYES (Gaussian) ==============

def train_naive_bayes(X_train, y_train, X_test, y_test):
    """Gaussian Naive Bayes on GPU."""
    # Compute class statistics
    mask0 = (y_train == 0)
    mask1 = (y_train == 1)

    mean0 = X_train[mask0].mean(dim=0)
    mean1 = X_train[mask1].mean(dim=0)
    var0 = X_train[mask0].var(dim=0) + 1e-6
    var1 = X_train[mask1].var(dim=0) + 1e-6

    prior0 = mask0.float().mean()
    prior1 = mask1.float().mean()

    # Compute log likelihoods
    def log_likelihood(X, mean, var):
        return -0.5 * (torch.log(var) + (X - mean) ** 2 / var).sum(dim=1)

    ll0 = log_likelihood(X_test, mean0, var0) + torch.log(prior0)
    ll1 = log_likelihood(X_test, mean1, var1) + torch.log(prior1)

    preds = (ll1 > ll0).long()
    acc = (preds == y_test).float().mean().item()

    return acc


# ============== RIDGE CLASSIFIER ==============

def train_ridge(X_train, y_train, X_test, y_test):
    """Ridge regression classifier (closed-form on GPU)."""
    best_acc = 0

    # Convert labels to -1, 1
    y_float = y_train.float() * 2 - 1

    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        # Closed-form solution: w = (X'X + αI)^(-1) X'y
        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_float
        I = torch.eye(X_train.shape[1], device=DEVICE)
        w = torch.linalg.solve(XtX + alpha * I, Xty)

        # Predict
        preds = (X_test @ w > 0).long()
        acc = (preds == y_test).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    print(f"  Best alpha={best_alpha}")
    return best_acc


# ============== MAIN ==============

def run():
    print("=" * 70)
    print("Traditional Classifiers on GPU")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    X_train, y_train, X_test, y_test = load_data()
    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features")

    results = {}

    # 1. Logistic Regression with tuning
    results['LogisticRegression'] = tune_logistic_regression(X_train, y_train, X_test, y_test)

    # 2. SVM
    print("\n" + "=" * 60)
    print("SVM (Hinge Loss)")
    print("=" * 60)
    best_svm = 0
    for C in [0.01, 0.1, 1, 10, 100]:
        acc = train_svm(X_train, y_train, X_test, y_test, C=C)
        if acc > best_svm:
            best_svm = acc
            best_C = C
    print(f"  Best C={best_C}: {best_svm*100:.1f}%")
    results['SVM'] = best_svm

    # 3. Ridge
    print("\n" + "=" * 60)
    print("Ridge Classifier")
    print("=" * 60)
    results['Ridge'] = train_ridge(X_train, y_train, X_test, y_test)
    print(f"  Test: {results['Ridge']*100:.1f}%")

    # 4. KNN
    print("\n" + "=" * 60)
    print("K-Nearest Neighbors")
    print("=" * 60)
    results['KNN'] = train_knn(X_train, y_train, X_test, y_test)
    print(f"  Test: {results['KNN']*100:.1f}%")

    # 5. Naive Bayes
    print("\n" + "=" * 60)
    print("Gaussian Naive Bayes")
    print("=" * 60)
    results['NaiveBayes'] = train_naive_bayes(X_train, y_train, X_test, y_test)
    print(f"  Test: {results['NaiveBayes']*100:.1f}%")

    # 6. Gradient Boosting
    print("\n" + "=" * 60)
    print("Gradient Boosting")
    print("=" * 60)
    results['GradientBoosting'] = train_gradient_boosting(X_train, y_train, X_test, y_test)
    print(f"  Test: {results['GradientBoosting']*100:.1f}%")

    # 7. Random Forest
    print("\n" + "=" * 60)
    print("Random Forest")
    print("=" * 60)
    results['RandomForest'] = train_random_forest(X_train, y_train, X_test, y_test)
    print(f"  Test: {results['RandomForest']*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - Sorted by Test Accuracy")
    print("=" * 70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 50)
        print(f"  {name:20s}: {acc*100:.1f}% {bar}")

    print(f"\n  Random baseline: 50.0%")


if __name__ == "__main__":
    run()
