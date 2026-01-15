#!/usr/bin/env python3
"""
Directional Persona Sampling for LinkedIn MLP.

Instead of random noise, use gradient-based perturbation to move embeddings
toward higher persona scores. The key insight is that gradient ascent on
persona score moves inputs toward regions where the MLP is more "confident"
in a way that correlates with correctness.

Mathematical framework:
- x = input embedding
- h(x) = MLP hidden state
- v = persona vector (mean_correct - mean_incorrect in hidden space)
- s(x) = h(x) · v = persona score

We compute ∇_x s(x) and perturb: x' = x + ε * ∇_x s(x)
This moves x in the direction that maximally increases persona score.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "linkedin_pairs.json"
EMBEDDINGS_FILE = Path(__file__).parent.parent.parent / "data" / "gemma_1b_embeddings.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPWithHidden(nn.Module):
    """MLP that can return hidden states."""
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            prev = h
        self.final = nn.Linear(prev, 2)
        self.hidden_dim = hidden_dims[-1]

    def forward(self, x, return_hidden=False):
        for layer in self.layers:
            x = layer(x)
        hidden = x
        logits = self.final(x)
        if return_hidden:
            return logits, hidden
        return logits


def train_mlp(X, y, hidden_dims=[256, 128], epochs=300):
    """Train MLP."""
    model = MLPWithHidden(X.shape[1], hidden_dims).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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


def extract_persona_vector(model, X, y, normalize=True):
    """Extract persona vector from hidden states."""
    model.eval()
    with torch.no_grad():
        logits, hidden = model(X, return_hidden=True)
        preds = logits.argmax(dim=1)
        correct_mask = (preds == y)

    hidden_correct = hidden[correct_mask]
    hidden_incorrect = hidden[~correct_mask]

    n_correct = correct_mask.sum().item()
    n_incorrect = (~correct_mask).sum().item()
    print(f"  Correct: {n_correct}, Incorrect: {n_incorrect}")

    if n_incorrect == 0:
        print("  Warning: No incorrect predictions for persona vector")
        return torch.zeros(hidden.shape[1], device=DEVICE)

    persona_vector = hidden_correct.mean(dim=0) - hidden_incorrect.mean(dim=0)

    if normalize:
        persona_vector = persona_vector / (persona_vector.norm() + 1e-8)

    return persona_vector


def gradient_ascent_perturbation(model, persona_vector, x, n_steps=10, step_size=0.1, max_delta=0.5):
    """
    Perturb embedding to maximize persona score using gradient ascent.

    Args:
        model: MLP
        persona_vector: direction of "correctness"
        x: input embedding [1, dim]
        n_steps: gradient ascent steps
        step_size: learning rate
        max_delta: maximum L2 norm of perturbation

    Returns:
        final prediction, final persona score, perturbation norm
    """
    model.eval()
    x_orig = x.clone().detach()
    x_pert = x.clone().detach().requires_grad_(True)

    for step in range(n_steps):
        if x_pert.grad is not None:
            x_pert.grad.zero_()

        logits, hidden = model(x_pert, return_hidden=True)
        score = (hidden * persona_vector).sum()

        # Gradient ascent: maximize score
        score.backward()

        with torch.no_grad():
            # Update in gradient direction (ascent, not descent)
            x_pert = x_pert + step_size * x_pert.grad

            # Project to constrain perturbation magnitude
            delta = x_pert - x_orig
            delta_norm = delta.norm()
            if delta_norm > max_delta:
                delta = delta * (max_delta / delta_norm)
                x_pert = x_orig + delta

            x_pert = x_pert.detach().requires_grad_(True)

    # Final prediction
    with torch.no_grad():
        logits, hidden = model(x_pert, return_hidden=True)
        pred = logits.argmax(dim=1).item()
        final_score = (hidden * persona_vector).sum().item()
        delta_norm = (x_pert - x_orig).norm().item()

    return pred, final_score, delta_norm


def multi_start_perturbation(model, persona_vector, x, n_starts=5, n_steps=10,
                             step_size=0.1, max_delta=0.5, noise_init=0.1):
    """
    Run gradient ascent from multiple starting points, select best.

    Each start begins with small random noise added to x.
    """
    model.eval()
    x_orig = x.clone().detach()

    best_pred = None
    best_score = float('-inf')

    for _ in range(n_starts):
        # Initialize with small noise
        x_init = x_orig + torch.randn_like(x_orig) * noise_init
        x_pert = x_init.clone().requires_grad_(True)

        for step in range(n_steps):
            if x_pert.grad is not None:
                x_pert.grad.zero_()

            logits, hidden = model(x_pert, return_hidden=True)
            score = (hidden * persona_vector).sum()
            score.backward()

            with torch.no_grad():
                x_pert = x_pert + step_size * x_pert.grad

                # Constrain total perturbation from original
                delta = x_pert - x_orig
                delta_norm = delta.norm()
                if delta_norm > max_delta:
                    delta = delta * (max_delta / delta_norm)
                    x_pert = x_orig + delta

                x_pert = x_pert.detach().requires_grad_(True)

        # Evaluate this trajectory
        with torch.no_grad():
            logits, hidden = model(x_pert, return_hidden=True)
            pred = logits.argmax(dim=1).item()
            final_score = (hidden * persona_vector).sum().item()

        if final_score > best_score:
            best_score = final_score
            best_pred = pred

    return best_pred, best_score


def langevin_sampling(model, persona_vector, x, n_steps=20, step_size=0.1,
                      noise_scale=0.05, temperature=1.0):
    """
    Langevin dynamics sampling: gradient ascent + noise for exploration.

    This combines deterministic gradient ascent with stochastic exploration,
    similar to MCMC sampling from the persona-weighted distribution.

    x_{t+1} = x_t + ε * ∇_x s(x) + √(2ε/T) * noise
    """
    model.eval()
    x_pert = x.clone().detach().requires_grad_(True)

    best_pred = None
    best_score = float('-inf')

    for step in range(n_steps):
        if x_pert.grad is not None:
            x_pert.grad.zero_()

        logits, hidden = model(x_pert, return_hidden=True)
        score = (hidden * persona_vector).sum()
        score.backward()

        with torch.no_grad():
            # Gradient step + noise (Langevin dynamics)
            noise = torch.randn_like(x_pert) * noise_scale * np.sqrt(2 * step_size / temperature)
            x_pert = x_pert + step_size * x_pert.grad + noise
            x_pert = x_pert.detach().requires_grad_(True)

            # Track best
            logits_check, hidden_check = model(x_pert, return_hidden=True)
            current_score = (hidden_check * persona_vector).sum().item()
            current_pred = logits_check.argmax(dim=1).item()

            if current_score > best_score:
                best_score = current_score
                best_pred = current_pred

    return best_pred, best_score


def run():
    print("=" * 70)
    print("DIRECTIONAL PERSONA SAMPLING")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load data
    print("\nLoading data...")
    emb_data = np.load(EMBEDDINGS_FILE)
    X_train_emb, y_train = emb_data['X_train'], emb_data['y_train']
    X_test_emb, y_test = emb_data['X_test'], emb_data['y_test']

    print(f"Train: {len(X_train_emb)}, Test: {len(X_test_emb)}")

    # Split for persona extraction
    n_train = len(X_train_emb)
    n_val = int(n_train * 0.2)
    np.random.seed(42)
    indices = np.random.permutation(n_train)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_train_train = X_train_emb[train_idx]
    y_train_train = y_train[train_idx]
    X_train_val = X_train_emb[val_idx]
    y_train_val = y_train[val_idx]

    print(f"Train-train: {len(X_train_train)}, Train-val: {len(X_train_val)}")

    # Prepare tensors
    X_train_train_t = torch.tensor(X_train_train, dtype=torch.float32, device=DEVICE)
    X_train_val_t = torch.tensor(X_train_val, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test_emb, dtype=torch.float32, device=DEVICE)
    y_train_train_t = torch.tensor(y_train_train, dtype=torch.long, device=DEVICE)
    y_train_val_t = torch.tensor(y_train_val, dtype=torch.long, device=DEVICE)

    # Normalize
    mean, std = X_train_train_t.mean(0), X_train_train_t.std(0) + 1e-8
    X_train_train_norm = (X_train_train_t - mean) / std
    X_train_val_norm = (X_train_val_t - mean) / std
    X_test_norm = (X_test_t - mean) / std

    # Train MLP
    print("\nTraining MLP...")
    torch.manual_seed(42)
    model = train_mlp(X_train_train_norm, y_train_train_t)

    # Baseline
    model.eval()
    with torch.no_grad():
        logits = model(X_test_norm)
        baseline_preds = logits.argmax(dim=1).cpu().numpy()
    baseline_acc = (baseline_preds == y_test).mean()
    print(f"Baseline MLP accuracy: {baseline_acc*100:.1f}%")

    # Extract persona vector from validation set
    print("\nExtracting persona vector...")
    persona_vector = extract_persona_vector(model, X_train_val_norm, y_train_val_t)
    print(f"Persona vector norm: {persona_vector.norm().item():.4f}")

    # Analyze persona signal
    with torch.no_grad():
        _, test_hidden = model(X_test_norm, return_hidden=True)
        test_scores = (test_hidden * persona_vector).sum(dim=1)

    correct_mask = (baseline_preds == y_test)
    print(f"\nPersona score analysis (test set):")
    print(f"  Correct predictions: {test_scores[torch.tensor(correct_mask)].mean():.3f}")
    print(f"  Incorrect predictions: {test_scores[torch.tensor(~correct_mask)].mean():.3f}")

    # ========================================
    # APPROACH 1: Simple Gradient Ascent
    # ========================================
    print("\n" + "=" * 70)
    print("APPROACH 1: GRADIENT ASCENT PERTURBATION")
    print("=" * 70)

    for n_steps in [5, 10, 20]:
        for step_size in [0.05, 0.1, 0.2]:
            for max_delta in [0.3, 0.5, 1.0]:
                correct = 0
                for i in range(len(X_test_norm)):
                    x = X_test_norm[i:i+1]
                    pred, _, _ = gradient_ascent_perturbation(
                        model, persona_vector, x,
                        n_steps=n_steps, step_size=step_size, max_delta=max_delta
                    )
                    if pred == y_test[i]:
                        correct += 1

                acc = correct / len(X_test_norm)
                improvement = acc - baseline_acc
                marker = " <--" if improvement > 0.01 else ""
                print(f"  steps={n_steps}, lr={step_size}, max_δ={max_delta}: {acc*100:.1f}% ({improvement*100:+.1f}pp){marker}")

    # ========================================
    # APPROACH 2: Multi-Start Gradient Ascent
    # ========================================
    print("\n" + "=" * 70)
    print("APPROACH 2: MULTI-START GRADIENT ASCENT")
    print("=" * 70)

    for n_starts in [3, 5, 10]:
        for noise_init in [0.05, 0.1, 0.2]:
            correct = 0
            for i in range(len(X_test_norm)):
                x = X_test_norm[i:i+1]
                pred, _ = multi_start_perturbation(
                    model, persona_vector, x,
                    n_starts=n_starts, n_steps=10, step_size=0.1,
                    max_delta=0.5, noise_init=noise_init
                )
                if pred == y_test[i]:
                    correct += 1

            acc = correct / len(X_test_norm)
            improvement = acc - baseline_acc
            marker = " <--" if improvement > 0.01 else ""
            print(f"  starts={n_starts}, noise={noise_init}: {acc*100:.1f}% ({improvement*100:+.1f}pp){marker}")

    # ========================================
    # APPROACH 3: Langevin Dynamics
    # ========================================
    print("\n" + "=" * 70)
    print("APPROACH 3: LANGEVIN DYNAMICS")
    print("=" * 70)

    for n_steps in [10, 20, 50]:
        for temp in [0.5, 1.0, 2.0]:
            correct = 0
            for i in range(len(X_test_norm)):
                x = X_test_norm[i:i+1]
                pred, _ = langevin_sampling(
                    model, persona_vector, x,
                    n_steps=n_steps, step_size=0.1,
                    noise_scale=0.05, temperature=temp
                )
                if pred == y_test[i]:
                    correct += 1

            acc = correct / len(X_test_norm)
            improvement = acc - baseline_acc
            marker = " <--" if improvement > 0.01 else ""
            print(f"  steps={n_steps}, temp={temp}: {acc*100:.1f}% ({improvement*100:+.1f}pp){marker}")

    # ========================================
    # ANALYSIS: What happens during perturbation?
    # ========================================
    print("\n" + "=" * 70)
    print("ANALYSIS: PERTURBATION DYNAMICS")
    print("=" * 70)

    # Analyze a few examples
    n_examples = 10
    print(f"\nAnalyzing {n_examples} examples...")

    flips_to_correct = 0
    flips_to_incorrect = 0
    stayed_correct = 0
    stayed_incorrect = 0

    for i in range(min(n_examples * 10, len(X_test_norm))):
        x = X_test_norm[i:i+1]

        # Original prediction
        with torch.no_grad():
            orig_logits, orig_hidden = model(x, return_hidden=True)
            orig_pred = orig_logits.argmax(dim=1).item()
            orig_score = (orig_hidden * persona_vector).sum().item()

        # Perturbed prediction
        pert_pred, pert_score, delta_norm = gradient_ascent_perturbation(
            model, persona_vector, x, n_steps=20, step_size=0.1, max_delta=0.5
        )

        label = y_test[i]
        orig_correct = (orig_pred == label)
        pert_correct = (pert_pred == label)

        if orig_correct and pert_correct:
            stayed_correct += 1
        elif not orig_correct and not pert_correct:
            stayed_incorrect += 1
        elif not orig_correct and pert_correct:
            flips_to_correct += 1
        else:
            flips_to_incorrect += 1

    total = flips_to_correct + flips_to_incorrect + stayed_correct + stayed_incorrect
    print(f"\n  Stayed correct: {stayed_correct}/{total}")
    print(f"  Stayed incorrect: {stayed_incorrect}/{total}")
    print(f"  Flipped to correct: {flips_to_correct}/{total}")
    print(f"  Flipped to incorrect: {flips_to_incorrect}/{total}")

    net_improvement = flips_to_correct - flips_to_incorrect
    print(f"\n  Net flips: {net_improvement:+d}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline MLP: {baseline_acc*100:.1f}%")
    print(f"\nThe directional perturbation approaches above show whether")
    print(f"moving embeddings toward higher persona scores improves accuracy.")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    run()
