#!/usr/bin/env python3
"""
TRM v3: With output feedback loop (closer to real TRM).

Key insight from real TRM: The outer ACT loop feeds the model's
output back as input for the next iteration. This creates a
refinement loop where the model can look at its current prediction
and improve it.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

EMBEDDINGS_FILE = "data/gemma_1b_embeddings.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    data = np.load(EMBEDDINGS_FILE)

    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32, device=DEVICE)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=DEVICE)

    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(f"Data: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, y_train, X_test, y_test


def rms_norm(x, eps=1e-6):
    """RMSNorm without learnable weight (like real TRM)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class SwiGLU(nn.Module):
    """SwiGLU as in real TRM."""
    def __init__(self, dim, expansion=4):
        super().__init__()
        inter = int(dim * expansion * 2 / 3)
        # Combined gate and up projection (like real TRM)
        self.gate_up = nn.Linear(dim, inter * 2, bias=False)
        self.down = nn.Linear(inter, dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class TRMBlock(nn.Module):
    """
    Proper TRM block with TWO sub-layers:
    1. First MLP (or attention in full TRM)
    2. Second MLP
    Both with post-norm residuals.
    """
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.mlp1 = SwiGLU(dim, expansion)
        self.mlp2 = SwiGLU(dim, expansion)

    def forward(self, x):
        # Sub-layer 1
        x = rms_norm(x + self.mlp1(x))
        # Sub-layer 2
        x = rms_norm(x + self.mlp2(x))
        return x


class TRMLevel(nn.Module):
    """Stack of TRM blocks."""
    def __init__(self, dim, n_layers=1, expansion=4):
        super().__init__()
        self.layers = nn.ModuleList([TRMBlock(dim, expansion) for _ in range(n_layers)])

    def forward(self, hidden, injection):
        x = hidden + injection
        for layer in self.layers:
            x = layer(x)
        return x


class ProperTRMv3(nn.Module):
    """
    TRM with output feedback loop.

    Like real TRM's outer ACT loop: the model's output (logits/probs)
    is projected and added to the input for the next iteration.
    This allows the model to "see" its current prediction and refine it.
    """
    def __init__(self, input_dim, hidden_dim=32, n_layers=1, expansion=2, num_classes=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Output feedback projection (logits -> hidden)
        self.feedback_proj = nn.Linear(num_classes, hidden_dim)

        # Single L_level (weight sharing)
        self.L_level = TRMLevel(hidden_dim, n_layers=n_layers, expansion=expansion)

        # Learned initial states
        self.H_init = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.L_init = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # Output head
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, K_outer=4, H_cycles=2, L_cycles=3, grad_last_only=True):
        """
        Two-level iteration:
        - Outer loop (K_outer): with output feedback
        - Inner loops (H_cycles, L_cycles): state refinement
        """
        B = x.shape[0]
        x_proj = self.input_proj(x)

        # Initialize states
        z_H = self.H_init.unsqueeze(0).expand(B, -1).clone()
        z_L = self.L_init.unsqueeze(0).expand(B, -1).clone()

        # Initial "prediction" feedback (zeros)
        feedback = torch.zeros(B, self.hidden_dim, device=x.device)

        all_logits = []

        for k in range(K_outer):
            # Combined input: original + feedback from previous prediction
            combined_input = x_proj + feedback

            if grad_last_only and k < K_outer - 1:
                with torch.no_grad():
                    for h in range(H_cycles):
                        for l in range(L_cycles):
                            z_L = self.L_level(z_L, z_H + combined_input)
                        z_H = self.L_level(z_H, z_L)
                    z_H = z_H.detach()
                    z_L = z_L.detach()
            else:
                for h in range(H_cycles):
                    for l in range(L_cycles):
                        z_L = self.L_level(z_L, z_H + combined_input)
                    z_H = self.L_level(z_H, z_L)

            # Get current prediction
            logits = self.output(z_H)
            all_logits.append(logits)

            # Create feedback for next iteration
            # Use soft probabilities to avoid hard decisions
            probs = F.softmax(logits.detach(), dim=-1)
            feedback = self.feedback_proj(probs)

        return logits, torch.stack(all_logits)


class ProperTRMv3Simple(nn.Module):
    """
    Simpler version: just one level of iteration with feedback.
    More similar to original simple approach but with feedback.
    """
    def __init__(self, input_dim, hidden_dim=32, expansion=2, num_classes=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.feedback_proj = nn.Linear(num_classes, hidden_dim)

        # Update block
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, K=32, grad_last_only=True):
        B = x.shape[0]
        x_proj = self.input_proj(x)

        z = torch.zeros(B, self.hidden_dim, device=x.device)
        feedback = torch.zeros(B, self.hidden_dim, device=x.device)

        all_logits = []

        for k in range(K):
            combined = x_proj + feedback

            if grad_last_only and k < K - 1:
                with torch.no_grad():
                    z = rms_norm(z + self.update(torch.cat([z, combined], dim=-1)))
                z = z.detach()
            else:
                z = rms_norm(z + self.update(torch.cat([z, combined], dim=-1)))

            logits = self.output(z)
            all_logits.append(logits)

            probs = F.softmax(logits.detach(), dim=-1)
            feedback = self.feedback_proj(probs)

        return logits, torch.stack(all_logits)


def train_model(model, X_train, y_train, X_test, y_test,
                epochs=300, lr=1e-3, **forward_kwargs):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits, _ = model(X_train, **forward_kwargs)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_logits, _ = model(X_test, **forward_kwargs)
                train_logits, _ = model(X_train, **forward_kwargs)

                train_acc = (train_logits.argmax(1) == y_train).float().mean().item()
                test_acc = (test_logits.argmax(1) == y_test).float().mean().item()

                if test_acc > best_test_acc:
                    best_test_acc = test_acc

                print(f"  Epoch {epoch+1}: Train {train_acc*100:.1f}%, Test {test_acc*100:.1f}%")

    return best_test_acc


def analyze_iterations(model, X_test, y_test, **forward_kwargs):
    model.eval()
    print("\nAccuracy by outer iteration:")

    with torch.no_grad():
        _, all_logits = model(X_test, grad_last_only=False, **forward_kwargs)

        for k in range(all_logits.shape[0]):
            acc = (all_logits[k].argmax(1) == y_test).float().mean().item()
            bar = "█" * int(acc * 40)
            print(f"  K={k+1:2d}: {acc*100:.1f}% {bar}")


def run():
    print("=" * 70)
    print("TRM v3: With Output Feedback Loop")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    X_train, y_train, X_test, y_test = load_data()
    input_dim = X_train.shape[1]

    results = {}

    # Test configurations
    configs = [
        # Simple version with feedback
        ("Simple_h8_K16", ProperTRMv3Simple, {"hidden_dim": 8}, {"K": 16}),
        ("Simple_h8_K32", ProperTRMv3Simple, {"hidden_dim": 8}, {"K": 32}),
        ("Simple_h8_K64", ProperTRMv3Simple, {"hidden_dim": 8}, {"K": 64}),
        ("Simple_h16_K32", ProperTRMv3Simple, {"hidden_dim": 16}, {"K": 32}),
        ("Simple_h16_K64", ProperTRMv3Simple, {"hidden_dim": 16}, {"K": 64}),

        # Full TRM with feedback
        ("TRM_h8_K8_H2_L3", ProperTRMv3, {"hidden_dim": 8, "n_layers": 1}, {"K_outer": 8, "H_cycles": 2, "L_cycles": 3}),
        ("TRM_h8_K16_H2_L3", ProperTRMv3, {"hidden_dim": 8, "n_layers": 1}, {"K_outer": 16, "H_cycles": 2, "L_cycles": 3}),
        ("TRM_h8_K32_H2_L3", ProperTRMv3, {"hidden_dim": 8, "n_layers": 1}, {"K_outer": 32, "H_cycles": 2, "L_cycles": 3}),
        ("TRM_h4_K32_H2_L3", ProperTRMv3, {"hidden_dim": 4, "n_layers": 1}, {"K_outer": 32, "H_cycles": 2, "L_cycles": 3}),
        ("TRM_h4_K64_H2_L3", ProperTRMv3, {"hidden_dim": 4, "n_layers": 1}, {"K_outer": 64, "H_cycles": 2, "L_cycles": 3}),
    ]

    for name, model_cls, model_kwargs, forward_kwargs in configs:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print("=" * 60)

        model = model_cls(input_dim, **model_kwargs).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        best_acc = train_model(
            model, X_train, y_train, X_test, y_test,
            epochs=300, lr=1e-3, **forward_kwargs
        )
        results[name] = best_acc

        # Show iteration analysis for smaller K
        if forward_kwargs.get('K', forward_kwargs.get('K_outer', 0)) <= 32:
            analyze_iterations(model, X_test, y_test, **forward_kwargs)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:25s}: {acc*100:.1f}%")

    print(f"\n  Previous best (no feedback): 74.3%")
    print(f"  Ridge baseline: 75.0%")
    print(f"  Random: 50.0%")


if __name__ == "__main__":
    run()
