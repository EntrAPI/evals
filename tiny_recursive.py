#!/usr/bin/env python3
"""
Tiny Recursive Model for classification on embeddings.
Inspired by TinyRecursiveModels - iteratively refines predictions.
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


class TinyRecursiveClassifier(nn.Module):
    """
    Tiny recursive model that iteratively refines predictions.

    Each iteration:
    1. Updates latent state z given input x and current z
    2. Produces prediction from z

    Key insight: same small network applied K times can solve
    problems that would require a much larger single-pass network.
    """
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()

        # Project input to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Recurrent update block (applied K times)
        # Takes [z, x_proj] and outputs updated z
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

        # Output head
        self.output = nn.Linear(hidden_dim, num_classes)

        self.hidden_dim = hidden_dim

    def forward(self, x, K=4):
        """
        Forward pass with K recursive iterations.

        Args:
            x: Input embeddings (B, input_dim)
            K: Number of refinement iterations

        Returns:
            logits: Final predictions (B, num_classes)
            all_logits: Predictions at each step (K, B, num_classes)
        """
        B = x.shape[0]

        # Project input
        x_proj = self.input_proj(x)  # (B, hidden_dim)

        # Initialize latent state
        z = torch.zeros(B, self.hidden_dim, device=x.device)

        all_logits = []

        for k in range(K):
            # Concatenate current state with input
            z_input = torch.cat([z, x_proj], dim=-1)  # (B, hidden_dim * 2)

            # Update state (with residual connection)
            z = z + self.update(z_input)
            z = self.norm(z)

            # Get prediction at this step
            logits = self.output(z)
            all_logits.append(logits)

        return logits, torch.stack(all_logits)


class TinyRecursiveClassifierV2(nn.Module):
    """
    Version 2: Two-level hierarchy like TRM.
    - L-level: fast inner loop
    - H-level: slow outer loop maintaining high-level state
    """
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # L-level update (inner loop)
        self.L_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # H-level update (outer loop)
        self.H_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm_L = nn.LayerNorm(hidden_dim)
        self.norm_H = nn.LayerNorm(hidden_dim)

        self.output = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim

    def forward(self, x, H_cycles=2, L_cycles=3):
        """
        Forward with hierarchical recursion.

        Args:
            x: Input (B, input_dim)
            H_cycles: Outer loop iterations
            L_cycles: Inner loop iterations per H cycle
        """
        B = x.shape[0]
        x_proj = self.input_proj(x)

        z_H = torch.zeros(B, self.hidden_dim, device=x.device)
        z_L = torch.zeros(B, self.hidden_dim, device=x.device)

        all_logits = []

        for h in range(H_cycles):
            # L-level inner loop
            for l in range(L_cycles):
                z_L_input = torch.cat([z_L, x_proj + z_H], dim=-1)
                z_L = z_L + self.L_update(z_L_input)
                z_L = self.norm_L(z_L)

            # H-level update
            z_H_input = torch.cat([z_H, z_L], dim=-1)
            z_H = z_H + self.H_update(z_H_input)
            z_H = self.norm_H(z_H)

            logits = self.output(z_H)
            all_logits.append(logits)

        return logits, torch.stack(all_logits)


def train_recursive(model, X_train, y_train, X_test, y_test,
                    epochs=300, lr=1e-3, K=4):
    """Train recursive model."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward with K iterations
        if hasattr(model, 'H_update'):
            # V2 model
            logits, all_logits = model(X_train, H_cycles=K, L_cycles=3)
        else:
            logits, all_logits = model(X_train, K=K)

        # Loss on final prediction
        loss = criterion(logits, y_train)

        # Optional: auxiliary loss on intermediate predictions
        # (helps train earlier iterations)
        for k in range(all_logits.shape[0] - 1):
            loss += 0.1 * criterion(all_logits[k], y_train)

        loss.backward()
        optimizer.step()

        # Evaluate
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'H_update'):
                    test_logits, _ = model(X_test, H_cycles=K, L_cycles=3)
                    train_logits, _ = model(X_train, H_cycles=K, L_cycles=3)
                else:
                    test_logits, _ = model(X_test, K=K)
                    train_logits, _ = model(X_train, K=K)

                train_acc = (train_logits.argmax(1) == y_train).float().mean().item()
                test_acc = (test_logits.argmax(1) == y_test).float().mean().item()

                if test_acc > best_test_acc:
                    best_test_acc = test_acc

                print(f"  Epoch {epoch+1}: Train {train_acc*100:.1f}%, Test {test_acc*100:.1f}%")

    return best_test_acc


def analyze_iterations(model, X_test, y_test, max_K=8):
    """Analyze accuracy at each iteration."""
    model.eval()
    print("\nAccuracy by iteration:")

    with torch.no_grad():
        if hasattr(model, 'H_update'):
            _, all_logits = model(X_test, H_cycles=max_K, L_cycles=3)
        else:
            _, all_logits = model(X_test, K=max_K)

        for k in range(all_logits.shape[0]):
            acc = (all_logits[k].argmax(1) == y_test).float().mean().item()
            bar = "█" * int(acc * 40)
            print(f"  K={k+1}: {acc*100:.1f}% {bar}")


def run():
    print("=" * 70)
    print("Tiny Recursive Model for Classification")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    X_train, y_train, X_test, y_test = load_data()
    input_dim = X_train.shape[1]

    results = {}

    # Test different configurations - even smaller models, higher K
    configs = [
        # Extremely tiny models
        ("V1_h2_K32", TinyRecursiveClassifier, {"hidden_dim": 2}, {"K": 32}),
        ("V1_h2_K64", TinyRecursiveClassifier, {"hidden_dim": 2}, {"K": 64}),
        ("V1_h2_K128", TinyRecursiveClassifier, {"hidden_dim": 2}, {"K": 128}),
        ("V1_h2_K256", TinyRecursiveClassifier, {"hidden_dim": 2}, {"K": 256}),
        ("V1_h4_K32", TinyRecursiveClassifier, {"hidden_dim": 4}, {"K": 32}),
        ("V1_h4_K64", TinyRecursiveClassifier, {"hidden_dim": 4}, {"K": 64}),
        ("V1_h4_K128", TinyRecursiveClassifier, {"hidden_dim": 4}, {"K": 128}),
        ("V1_h4_K256", TinyRecursiveClassifier, {"hidden_dim": 4}, {"K": 256}),
        ("V1_h8_K64", TinyRecursiveClassifier, {"hidden_dim": 8}, {"K": 64}),
        ("V1_h8_K128", TinyRecursiveClassifier, {"hidden_dim": 8}, {"K": 128}),
        ("V1_h8_K256", TinyRecursiveClassifier, {"hidden_dim": 8}, {"K": 256}),
    ]

    for name, model_cls, model_kwargs, train_kwargs in configs:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print("=" * 60)

        # Count parameters
        model = model_cls(input_dim, **model_kwargs).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        best_acc = train_recursive(
            model, X_train, y_train, X_test, y_test,
            epochs=300, lr=1e-3, **train_kwargs
        )
        results[name] = best_acc

        # Analyze iterations for this model (show subset for high K)
        analyze_iterations(model, X_test, y_test, max_K=min(train_kwargs["K"], 16))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}: {acc*100:.1f}%")

    print(f"\n  Previous best (Ridge): 75.0%")
    print(f"  Random baseline: 50.0%")


if __name__ == "__main__":
    run()
