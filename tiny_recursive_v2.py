#!/usr/bin/env python3
"""
Proper TRM-style Tiny Recursive Model for classification.
Following the actual TRM architecture more closely.
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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation as used in TRM."""
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden = int(dim * expansion)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))


class TRMBlock(nn.Module):
    """Single TRM block with SwiGLU and RMSNorm."""
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.mlp = SwiGLU(dim, expansion)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        # Post-norm residual
        return self.norm(x + self.mlp(x))


class TRMLevel(nn.Module):
    """A reasoning level (stack of blocks)."""
    def __init__(self, dim, n_layers=2, expansion=4):
        super().__init__()
        self.layers = nn.ModuleList([TRMBlock(dim, expansion) for _ in range(n_layers)])

    def forward(self, hidden, injection):
        """
        Args:
            hidden: Current hidden state
            injection: Input to add before processing
        """
        x = hidden + injection
        for layer in self.layers:
            x = layer(x)
        return x


class ProperTRM(nn.Module):
    """
    Proper TRM following the paper:
    - Two-level hierarchy (H and L)
    - Gradient only on last H-cycle
    - Weight sharing (same L_level for z_L and z_H updates)
    - Learned initial states
    """
    def __init__(self, input_dim, hidden_dim=32, n_layers=1, expansion=2, num_classes=2):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Single L_level used for both z_L and z_H updates (weight sharing!)
        self.L_level = TRMLevel(hidden_dim, n_layers=n_layers, expansion=expansion)

        # Learned initial states (like TRM)
        self.H_init = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.L_init = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # Output head
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, H_cycles=4, L_cycles=3, grad_last_only=True):
        """
        Forward with proper TRM-style iteration.

        Args:
            x: Input (B, input_dim)
            H_cycles: Number of outer loop iterations
            L_cycles: Number of inner loop iterations per H-cycle
            grad_last_only: If True, only compute gradients on last H-cycle
        """
        B = x.shape[0]

        # Project input
        x_proj = self.input_proj(x)  # (B, hidden_dim)

        # Initialize states from learned inits
        z_H = self.H_init.unsqueeze(0).expand(B, -1).clone()
        z_L = self.L_init.unsqueeze(0).expand(B, -1).clone()

        all_logits = []

        if grad_last_only and H_cycles > 1:
            # H_cycles - 1 without gradient
            with torch.no_grad():
                for h in range(H_cycles - 1):
                    # L-level inner loop: update z_L
                    for l in range(L_cycles):
                        z_L = self.L_level(z_L, z_H + x_proj)
                    # H-level: update z_H using z_L
                    z_H = self.L_level(z_H, z_L)

                    logits = self.output(z_H)
                    all_logits.append(logits.detach())

            # Detach for gradient boundary
            z_H = z_H.detach().requires_grad_(True)
            z_L = z_L.detach().requires_grad_(True)

        # Last H-cycle (or all cycles if not grad_last_only) with gradient
        remaining = 1 if (grad_last_only and H_cycles > 1) else H_cycles
        for h in range(remaining):
            for l in range(L_cycles):
                z_L = self.L_level(z_L, z_H + x_proj)
            z_H = self.L_level(z_H, z_L)

            logits = self.output(z_H)
            all_logits.append(logits)

        return logits, torch.stack(all_logits)


def train_model(model, X_train, y_train, X_test, y_test,
                epochs=300, lr=1e-3, H_cycles=4, L_cycles=3):
    """Train TRM model."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits, all_logits = model(X_train, H_cycles=H_cycles, L_cycles=L_cycles)

        # Loss only on final prediction (gradient only flows through last H-cycle)
        loss = criterion(logits, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_logits, _ = model(X_test, H_cycles=H_cycles, L_cycles=L_cycles)
                train_logits, _ = model(X_train, H_cycles=H_cycles, L_cycles=L_cycles)

                train_acc = (train_logits.argmax(1) == y_train).float().mean().item()
                test_acc = (test_logits.argmax(1) == y_test).float().mean().item()

                if test_acc > best_test_acc:
                    best_test_acc = test_acc

                print(f"  Epoch {epoch+1}: Train {train_acc*100:.1f}%, Test {test_acc*100:.1f}%")

    return best_test_acc


def analyze_iterations(model, X_test, y_test, H_cycles=8, L_cycles=3):
    """Analyze accuracy at each H-cycle."""
    model.eval()
    print(f"\nAccuracy by H-cycle (L_cycles={L_cycles}):")

    with torch.no_grad():
        _, all_logits = model(X_test, H_cycles=H_cycles, L_cycles=L_cycles, grad_last_only=False)

        for h in range(all_logits.shape[0]):
            acc = (all_logits[h].argmax(1) == y_test).float().mean().item()
            bar = "█" * int(acc * 40)
            total_iters = (h + 1) * (L_cycles + 1)  # L_cycles for z_L + 1 for z_H
            print(f"  H={h+1} (iters={total_iters:3d}): {acc*100:.1f}% {bar}")


def run():
    print("=" * 70)
    print("Proper TRM for Classification")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    X_train, y_train, X_test, y_test = load_data()
    input_dim = X_train.shape[1]

    results = {}

    # Test configurations - small models, high iterations
    configs = [
        # (name, hidden_dim, n_layers, expansion, H_cycles, L_cycles)
        ("h4_L1_H16_l3", 4, 1, 2, 16, 3),
        ("h4_L1_H32_l3", 4, 1, 2, 32, 3),
        ("h4_L1_H64_l3", 4, 1, 2, 64, 3),
        ("h4_L1_H128_l3", 4, 1, 2, 128, 3),
        ("h8_L1_H16_l3", 8, 1, 2, 16, 3),
        ("h8_L1_H32_l3", 8, 1, 2, 32, 3),
        ("h8_L1_H64_l3", 8, 1, 2, 64, 3),
        ("h8_L1_H128_l3", 8, 1, 2, 128, 3),
        ("h4_L2_H32_l3", 4, 2, 2, 32, 3),
        ("h4_L2_H64_l3", 4, 2, 2, 64, 3),
    ]

    for name, hidden_dim, n_layers, expansion, H_cycles, L_cycles in configs:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print("=" * 60)

        model = ProperTRM(
            input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            expansion=expansion
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        total_iters = H_cycles * (L_cycles + 1)
        print(f"Parameters: {n_params:,}, Total iterations: {total_iters}")

        best_acc = train_model(
            model, X_train, y_train, X_test, y_test,
            epochs=300, lr=1e-3, H_cycles=H_cycles, L_cycles=L_cycles
        )
        results[name] = best_acc

        # Analyze iterations
        analyze_iterations(model, X_test, y_test, H_cycles=min(H_cycles, 16), L_cycles=L_cycles)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:25s}: {acc*100:.1f}%")

    print(f"\n  Previous best (Ridge): 75.0%")
    print(f"  Random baseline: 50.0%")


if __name__ == "__main__":
    run()
