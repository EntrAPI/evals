#!/usr/bin/env python3
"""
Persona sampling for LinkedIn MLP.

Two approaches:
1. Quick test: Add noise to embeddings, score with persona vector
2. Full test: Generate paraphrases with Gemini, re-embed with Gemma

Persona vector: mean(hidden|correct) - mean(hidden|incorrect)
"""

import json
import time
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "linkedin_pairs.json"
EMBEDDINGS_FILE = Path(__file__).parent.parent.parent / "data" / "gemma_1b_embeddings.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gemini API
API_KEY = "AIzaSyCWrrgydQ-_kaS92B3vvXrstJcZnPgiE20"
GEMINI_MODEL = "gemini-2.0-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"


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
        hidden = x  # Last hidden layer before final projection
        logits = self.final(x)
        if return_hidden:
            return logits, hidden
        return logits


def train_mlp(X, y, hidden_dims=[256, 128], epochs=300):
    """Train MLP and return model."""
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
    """Extract persona vector from MLP hidden states on TRAINING data."""
    model.eval()
    with torch.no_grad():
        logits, hidden = model(X, return_hidden=True)
        preds = logits.argmax(dim=1)
        correct_mask = (preds == y)

    hidden_correct = hidden[correct_mask]
    hidden_incorrect = hidden[~correct_mask]

    print(f"  Correct: {correct_mask.sum().item()}, Incorrect: {(~correct_mask).sum().item()}")

    if len(hidden_correct) == 0 or len(hidden_incorrect) == 0:
        print("  Warning: Not enough samples for persona vector")
        return torch.zeros(hidden.shape[1], device=DEVICE)

    persona_vector = hidden_correct.mean(dim=0) - hidden_incorrect.mean(dim=0)

    if normalize:
        persona_vector = persona_vector / (persona_vector.norm() + 1e-8)

    return persona_vector


def call_gemini(prompt, temperature=0.7, max_tokens=500):
    """Call Gemini API."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }

    response = requests.post(API_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def generate_paraphrases(text, k=3):
    """Generate k paraphrases of the post using Gemini."""
    prompt = f"""Rewrite the following LinkedIn post in {k} different ways.
Each version should preserve the core message but vary the wording, structure, and style.
Return ONLY the {k} versions, separated by "---".

Original post:
{text[:500]}

Rewritten versions:"""

    try:
        response = call_gemini(prompt, temperature=0.9, max_tokens=1500)
        versions = [v.strip() for v in response.split("---") if v.strip()]
        return [text] + versions[:k]
    except Exception as e:
        return [text]


def run():
    print("=" * 70)
    print("MLP PERSONA SAMPLING")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load data
    print("\nLoading data...")
    emb_data = np.load(EMBEDDINGS_FILE)
    X_train_emb, y_train = emb_data['X_train'], emb_data['y_train']
    X_test_emb, y_test = emb_data['X_test'], emb_data['y_test']

    with open(DATA_FILE) as f:
        pairs_data = json.load(f)
    train_pairs = pairs_data['train']
    test_pairs = pairs_data['test']

    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")
    print(f"Embedding dim: {X_train_emb.shape[1]}")

    # Split training into train_train and train_val for persona extraction
    n_train = len(X_train_emb)
    n_val = int(n_train * 0.2)  # 20% for validation
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

    # Normalize using train-train statistics
    mean, std = X_train_train_t.mean(0), X_train_train_t.std(0) + 1e-8
    X_train_train_norm = (X_train_train_t - mean) / std
    X_train_val_norm = (X_train_val_t - mean) / std
    X_test_norm = (X_test_t - mean) / std

    # Train MLP on train-train
    print("\nTraining MLP on train-train...")
    torch.manual_seed(42)
    model = train_mlp(X_train_train_norm, y_train_train_t)

    # Baseline accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X_test_norm)
        baseline_preds = logits.argmax(dim=1).cpu().numpy()
    baseline_acc = (baseline_preds == y_test).mean()
    print(f"Baseline MLP accuracy: {baseline_acc*100:.1f}%")

    # Extract persona vector from validation data (where model makes errors)
    print("\nExtracting persona vector from validation data...")
    persona_vector = extract_persona_vector(model, X_train_val_norm, y_train_val_t)
    print(f"Persona vector norm: {persona_vector.norm().item():.4f}")

    # Check persona signal on test set (without resampling)
    print("\nAnalyzing persona signal on test set...")
    with torch.no_grad():
        _, test_hidden = model(X_test_norm, return_hidden=True)
        persona_scores = (test_hidden * persona_vector).sum(dim=1)

    correct_mask = (baseline_preds == y_test)
    score_correct = persona_scores[torch.tensor(correct_mask)].mean().item()
    score_incorrect = persona_scores[torch.tensor(~correct_mask)].mean().item()
    print(f"  Mean persona score (correct preds): {score_correct:.4f}")
    print(f"  Mean persona score (incorrect preds): {score_incorrect:.4f}")
    print(f"  Separation: {score_correct - score_incorrect:.4f}")

    # ========================================
    # APPROACH 1: Noise-based sampling
    # ========================================
    print("\n" + "=" * 70)
    print("APPROACH 1: NOISE-BASED PERSONA SAMPLING")
    print("=" * 70)

    for noise_scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
        for k in [5, 10, 20]:
            correct = 0
            for i in range(len(X_test_norm)):
                emb = X_test_norm[i:i+1]  # [1, dim]

                # Generate k noisy variants
                noise = torch.randn(k, emb.shape[1], device=DEVICE) * noise_scale
                variants = emb + noise  # [k, dim]

                # Score all variants
                with torch.no_grad():
                    logits, hidden = model(variants, return_hidden=True)
                    preds = logits.argmax(dim=1)  # [k]
                    scores = (hidden * persona_vector).sum(dim=1)  # [k]

                # Pick variant with highest persona score
                best_idx = scores.argmax().item()
                best_pred = preds[best_idx].item()

                if best_pred == y_test[i]:
                    correct += 1

            acc = correct / len(X_test_norm)
            improvement = acc - baseline_acc
            marker = " <--" if improvement > 0.005 else ""
            print(f"  noise={noise_scale}, k={k}: {acc*100:.1f}% ({improvement*100:+.1f}pp){marker}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline MLP: {baseline_acc*100:.1f}%")
    print(f"Persona signal separation (test): {score_correct - score_incorrect:.4f}")
    print(f"\nNoise-based sampling: See results above")

    # Note: Paraphrase approach requires Gemma 2B which is gated on HuggingFace
    # Would need HF authentication to run that experiment


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    run()
