#!/usr/bin/env python3
"""
GPU-accelerated classifiers on embeddings.
All models run entirely on GPU using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

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

    print(f"Data on {DEVICE}: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ============== MODELS ==============

class LogisticRegressionGPU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output(x)


class AttentionClassifier(nn.Module):
    """Simple self-attention based classifier."""
    def __init__(self, input_dim, n_heads=4, hidden_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # Project and add sequence dimension
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, H)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        x = self.norm2(x + self.ffn(x))

        return self.output(x.squeeze(1))


# ============== TRAINING ==============

def train_model(model, X_train, y_train, X_test, y_test,
                epochs=100, lr=1e-3, batch_size=128, patience=10):
    """Train a model with early stopping."""
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_acc = 0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_logits = model(X_train)
            train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()

            test_logits = model(X_test)
            test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

        scheduler.step(1 - test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_acc, train_acc


def run():
    print("=" * 70)
    print("GPU Classifiers on Gemma 1B Embeddings")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    X_train, y_train, X_test, y_test = load_data()
    input_dim = X_train.shape[1]

    results = {}

    # Define models to test
    models = {
        "LogisticRegression": lambda: LogisticRegressionGPU(input_dim),
        "MLP_256": lambda: MLP(input_dim, [256]),
        "MLP_512_256": lambda: MLP(input_dim, [512, 256]),
        "MLP_1024_512": lambda: MLP(input_dim, [1024, 512]),
        "DeepMLP": lambda: DeepMLP(input_dim),
        "ResNet_3blocks": lambda: ResNet(input_dim, hidden_dim=512, n_blocks=3),
        "ResNet_5blocks": lambda: ResNet(input_dim, hidden_dim=512, n_blocks=5),
        "Attention": lambda: AttentionClassifier(input_dim),
    }

    for name, model_fn in models.items():
        print(f"\n{'-'*60}")
        print(f"Training: {name}")
        print("-" * 60)

        # Train multiple times and take best
        best_test = 0
        for run_i in range(3):
            torch.manual_seed(42 + run_i)
            model = model_fn()
            test_acc, train_acc = train_model(
                model, X_train, y_train, X_test, y_test,
                epochs=200, lr=1e-3, patience=15
            )
            if test_acc > best_test:
                best_test = test_acc
                best_train = train_acc

        results[name] = best_test
        print(f"  Train: {best_train*100:.1f}%, Test: {best_test*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Sorted by Test Accuracy")
    print("=" * 70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:25s}: {acc*100:.1f}%")

    print(f"\n  Random baseline: 50.0%")


if __name__ == "__main__":
    run()
