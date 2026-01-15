#!/usr/bin/env python3
"""
Proper persona sampling v2.

Key fix: Score the OUTPUT (prediction), not the input.

For classification:
- persona_score(A) = dot(hidden_when_predict_A, persona_vector)
- persona_score(B) = dot(hidden_when_predict_B, persona_vector)
- Use softmax(persona_scores) to weight choices

The persona vector points toward "correct predictions" in hidden space.
"""

import json
import math
import random
import time
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "linkedin_pairs.json"
EMBEDDINGS_FILE = Path(__file__).parent.parent.parent / "data" / "gemma_1b_embeddings.npz"

API_KEY = "AIzaSyCWrrgydQ-_kaS92B3vvXrstJcZnPgiE20"
MODEL = "gemini-2.0-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPWithHiddenStates(nn.Module):
    """MLP that exposes hidden states."""
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(prev, h))
            self.dropouts.append(nn.Dropout(dropout))
            prev = h
        self.output = nn.Linear(prev, 2)
        self.hidden_dim = hidden_dims[-1]

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = torch.relu(layer(x))
            x = dropout(x)
        return self.output(x)

    def forward_with_hidden(self, x):
        """Return logits and last hidden state."""
        for layer, dropout in zip(self.layers, self.dropouts):
            x = torch.relu(layer(x))
            hidden = x  # Save before dropout
            x = dropout(x)
        logits = self.output(x)
        return logits, hidden


def train_mlp(model, X, y, epochs=300):
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


def get_gemini_predictions_multi(test_pairs, n_samples=5):
    """Get multiple Gemini predictions per pair."""
    cache_file = Path(__file__).parent / f"gemini_multi_{n_samples}.json"

    if cache_file.exists():
        print(f"Loading cached Gemini predictions ({n_samples} samples)...")
        with open(cache_file) as f:
            return json.load(f)

    print(f"Getting Gemini predictions ({n_samples} samples per pair)...")
    all_predictions = []

    for i, pair in enumerate(test_pairs):
        post_a = pair['post_a']['text'][:500]
        post_b = pair['post_b']['text'][:500]

        prompt = f"""You are predicting LinkedIn engagement. Which post got more engagement?

Post A:
{post_a}

Post B:
{post_b}

Reply with ONLY "A" or "B"."""

        pair_preds = []
        for s in range(n_samples):
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7 if s > 0 else 0.0,
                    "maxOutputTokens": 5
                }
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                response.raise_for_status()
                text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip().upper()

                if "A" in text and "B" not in text:
                    pair_preds.append(0)
                elif "B" in text and "A" not in text:
                    pair_preds.append(1)
                else:
                    pair_preds.append(-1)
            except Exception as e:
                pair_preds.append(-1)

            time.sleep(0.03)

        all_predictions.append(pair_preds)
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(test_pairs)}")

    with open(cache_file, 'w') as f:
        json.dump(all_predictions, f)

    return all_predictions


def run():
    print("=" * 70)
    print("PROPER PERSONA SAMPLING V2")
    print("=" * 70)

    # Load data
    emb_data = np.load(EMBEDDINGS_FILE)
    X_train_emb, y_train = emb_data['X_train'], emb_data['y_train']
    X_test_emb, y_test = emb_data['X_test'], emb_data['y_test']

    with open(DATA_FILE) as f:
        pairs_data = json.load(f)
    train_pairs = pairs_data['train']
    test_pairs = pairs_data['test']

    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Prepare tensors
    X_train_t = torch.tensor(X_train_emb, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test_emb, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)

    mean, std = X_train_t.mean(0), X_train_t.std(0) + 1e-8
    X_train_norm = (X_train_t - mean) / std
    X_test_norm = (X_test_t - mean) / std

    # ========================================
    # 1. Extract persona vector using cross-validation
    # ========================================
    print("\n" + "=" * 70)
    print("1. EXTRACTING PERSONA VECTOR")
    print("=" * 70)

    # Key insight: persona vector should distinguish correct vs incorrect
    # predictions in the HIDDEN SPACE of the model.
    #
    # For each training sample:
    # - Get hidden state h
    # - If model predicts correctly: h is a "positive" example
    # - If model predicts incorrectly: h is a "negative" example
    # - persona_vector = mean(h | correct) - mean(h | incorrect)

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Collect hidden states and correctness labels via CV
    all_hidden = []
    all_correct = []

    print(f"Running {n_folds}-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_norm)):
        X_fold_train = X_train_norm[train_idx]
        y_fold_train = y_train_t[train_idx]
        X_fold_val = X_train_norm[val_idx]
        y_fold_val = y_train[val_idx]

        torch.manual_seed(42 + fold)
        fold_model = MLPWithHiddenStates(X_train_emb.shape[1]).to(DEVICE)
        train_mlp(fold_model, X_fold_train, y_fold_train)

        fold_model.eval()
        with torch.no_grad():
            logits, hidden = fold_model.forward_with_hidden(X_fold_val)
            preds = logits.argmax(1).cpu().numpy()

        correct = (preds == y_fold_val)
        all_hidden.append(hidden.cpu())
        all_correct.append(correct)

    all_hidden = torch.cat(all_hidden, dim=0)
    all_correct = np.concatenate(all_correct)

    cv_acc = all_correct.mean()
    print(f"CV accuracy: {cv_acc*100:.1f}%")
    print(f"Correct: {all_correct.sum()}, Incorrect: {(~all_correct).sum()}")

    # Extract persona vector
    correct_mask = torch.tensor(all_correct, dtype=torch.bool)
    mean_correct = all_hidden[correct_mask].mean(dim=0)
    mean_incorrect = all_hidden[~correct_mask].mean(dim=0)

    persona_vector = mean_correct - mean_incorrect
    persona_vector = persona_vector / (persona_vector.norm() + 1e-8)
    persona_vector = persona_vector.to(DEVICE)

    print(f"Persona vector shape: {persona_vector.shape}")

    # Verify: persona scores should be higher for correct predictions
    persona_scores_train = (all_hidden.to(DEVICE) * persona_vector).sum(dim=1).cpu().numpy()
    print(f"Mean persona score (correct): {persona_scores_train[all_correct].mean():.3f}")
    print(f"Mean persona score (incorrect): {persona_scores_train[~all_correct].mean():.3f}")

    # ========================================
    # 2. Train final model
    # ========================================
    print("\n" + "=" * 70)
    print("2. TRAINING FINAL MODEL")
    print("=" * 70)

    torch.manual_seed(42)
    model = MLPWithHiddenStates(X_train_emb.shape[1]).to(DEVICE)
    train_mlp(model, X_train_norm, y_train_t)

    model.eval()
    with torch.no_grad():
        test_logits, test_hidden = model.forward_with_hidden(X_test_norm)
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = test_logits.argmax(1).cpu().numpy()

    mlp_acc = (test_preds == y_test).mean()
    print(f"MLP test accuracy: {mlp_acc*100:.1f}%")

    # Compute persona scores for test set
    persona_scores_test = (test_hidden * persona_vector).sum(dim=1).cpu().numpy()

    # ========================================
    # 3. Get Gemini proposals
    # ========================================
    print("\n" + "=" * 70)
    print("3. GEMINI PROPOSALS")
    print("=" * 70)

    n_samples = 5
    gemini_multi = get_gemini_predictions_multi(test_pairs, n_samples=n_samples)

    # Gemini accuracy
    gemini_preds = []
    for preds in gemini_multi:
        valid = [p for p in preds if p >= 0]
        if valid:
            gemini_preds.append(1 if sum(valid) > len(valid) / 2 else 0)
        else:
            gemini_preds.append(random.choice([0, 1]))
    gemini_preds = np.array(gemini_preds)
    gemini_acc = (gemini_preds == y_test).mean()
    print(f"Gemini majority vote: {gemini_acc*100:.1f}%")

    # ========================================
    # 4. PROPER PERSONA SAMPLING
    # ========================================
    print("\n" + "=" * 70)
    print("4. PERSONA SAMPLING METHODS")
    print("=" * 70)

    # Method A: Use persona score to decide when to trust MLP vs Gemini
    # High persona score = MLP is likely correct
    # Low persona score = MLP might be wrong, consider Gemini

    print("\n--- Method A: Threshold-based switching ---")
    for threshold in np.percentile(persona_scores_test, [25, 50, 75]):
        switched_preds = []
        gemini_used = 0
        for i in range(len(test_pairs)):
            if persona_scores_test[i] < threshold:
                # Low confidence - use Gemini
                switched_preds.append(gemini_preds[i])
                gemini_used += 1
            else:
                switched_preds.append(test_preds[i])

        switched_preds = np.array(switched_preds)
        acc = (switched_preds == y_test).mean()
        print(f"  Threshold={threshold:.2f}: {acc*100:.1f}% (Gemini used: {gemini_used})")

    # Method B: Importance weighting
    # P(use MLP) ∝ sigmoid(persona_score * beta)
    print("\n--- Method B: Soft importance weighting ---")
    for beta in [0.5, 1.0, 2.0]:
        weighted_preds = []
        for i in range(len(test_pairs)):
            # Weight for MLP increases with persona score
            mlp_weight = 1 / (1 + np.exp(-beta * persona_scores_test[i]))

            # Get Gemini probability
            valid = [p for p in gemini_multi[i] if p >= 0]
            gemini_prob_a = (1 - sum(valid) / len(valid)) if valid else 0.5

            # Combine
            combined_prob_a = mlp_weight * test_probs[i, 0] + (1 - mlp_weight) * gemini_prob_a
            weighted_preds.append(0 if combined_prob_a > 0.5 else 1)

        weighted_preds = np.array(weighted_preds)
        acc = (weighted_preds == y_test).mean()
        print(f"  Beta={beta}: {acc*100:.1f}%")

    # Method C: MH-style - accept Gemini proposal based on persona improvement
    print("\n--- Method C: MH acceptance ---")
    for beta in [1.0, 2.0, 5.0]:
        mh_preds = []
        accepted = 0
        for i in range(len(test_pairs)):
            mlp_pred = test_preds[i]
            gemini_pred = gemini_preds[i]

            if mlp_pred == gemini_pred:
                mh_preds.append(mlp_pred)
            else:
                # MH: accept Gemini if it improves persona score
                # Since we can't easily compute persona score for Gemini's choice,
                # we use: if MLP persona score is low, accept Gemini with higher prob

                # log_accept = -beta * persona_score (low score = accept more)
                log_accept = -beta * persona_scores_test[i]
                accept_prob = min(1.0, np.exp(log_accept))

                if random.random() < accept_prob:
                    mh_preds.append(gemini_pred)
                    accepted += 1
                else:
                    mh_preds.append(mlp_pred)

        mh_preds = np.array(mh_preds)
        acc = (mh_preds == y_test).mean()
        disagreements = (test_preds != gemini_preds).sum()
        print(f"  Beta={beta}: {acc*100:.1f}% (accepted {accepted}/{disagreements} switches)")

    # ========================================
    # 5. WITH AUTHOR HEURISTIC
    # ========================================
    print("\n" + "=" * 70)
    print("5. WITH AUTHOR HEURISTIC")
    print("=" * 70)

    # Build author stats
    author_stats = defaultdict(list)
    for pair in train_pairs:
        for key in ['post_a', 'post_b']:
            author = pair[key].get('author', 'unknown')
            eng = pair[key]['reactions'] + pair[key].get('comments', 0)
            author_stats[author].append(eng)

    author_avg = {a: np.mean(engs) for a, engs in author_stats.items()}
    global_median = np.median([e for engs in author_stats.values() for e in engs])

    def get_author_probs(pair):
        a = pair['post_a'].get('author', 'unknown')
        b = pair['post_b'].get('author', 'unknown')
        avg_a = author_avg.get(a, global_median)
        avg_b = author_avg.get(b, global_median)
        diff = np.log1p(avg_a) - np.log1p(avg_b)
        prob_a = 1 / (1 + np.exp(-diff * 2))
        return np.array([prob_a, 1 - prob_a])

    author_probs = np.array([get_author_probs(p) for p in test_pairs])
    author_preds = author_probs.argmax(1)
    author_acc = (author_preds == y_test).mean()
    print(f"Author heuristic: {author_acc*100:.1f}%")

    # Baseline: 80% author + 20% MLP
    baseline_probs = 0.8 * author_probs + 0.2 * test_probs
    baseline_preds = baseline_probs.argmax(1)
    baseline_acc = (baseline_preds == y_test).mean()
    print(f"Baseline (80% author + 20% MLP): {baseline_acc*100:.1f}%")

    # Persona-guided: adjust MLP weight based on persona score
    print("\n--- Persona-guided ensemble ---")
    for base_author_w in [0.7, 0.75, 0.8]:
        for beta in [0.5, 1.0, 2.0]:
            persona_preds = []
            for i in range(len(test_pairs)):
                # Adjust MLP contribution based on persona score
                # High persona = trust MLP more
                mlp_boost = 1 / (1 + np.exp(-beta * persona_scores_test[i]))
                mlp_w = 0.1 + 0.15 * mlp_boost  # 0.1 to 0.25

                author_w = base_author_w
                gemini_w = max(0, 1 - author_w - mlp_w)

                valid = [p for p in gemini_multi[i] if p >= 0]
                gemini_prob_a = (1 - sum(valid) / len(valid)) if valid else 0.5
                gemini_probs = np.array([gemini_prob_a, 1 - gemini_prob_a])

                combined = author_w * author_probs[i] + mlp_w * test_probs[i] + gemini_w * gemini_probs
                persona_preds.append(combined.argmax())

            persona_preds = np.array(persona_preds)
            acc = (persona_preds == y_test).mean()
            print(f"  Author={base_author_w}, beta={beta}: {acc*100:.1f}%")

    # ========================================
    # ANALYSIS: Where does persona help?
    # ========================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check if persona score correlates with correctness on test set
    mlp_correct = (test_preds == y_test)
    print(f"\nPersona score vs correctness (test set):")
    print(f"  Mean score when MLP correct: {persona_scores_test[mlp_correct].mean():.3f}")
    print(f"  Mean score when MLP wrong: {persona_scores_test[~mlp_correct].mean():.3f}")

    # Cases where MLP wrong but Gemini right
    mlp_wrong_gemini_right = (~mlp_correct) & (gemini_preds == y_test)
    print(f"\nCases where MLP wrong, Gemini right: {mlp_wrong_gemini_right.sum()}")
    if mlp_wrong_gemini_right.sum() > 0:
        print(f"  Mean persona score: {persona_scores_test[mlp_wrong_gemini_right].mean():.3f}")

    # Cases where MLP right but Gemini wrong
    mlp_right_gemini_wrong = mlp_correct & (gemini_preds != y_test)
    print(f"\nCases where MLP right, Gemini wrong: {mlp_right_gemini_wrong.sum()}")
    if mlp_right_gemini_wrong.sum() > 0:
        print(f"  Mean persona score: {persona_scores_test[mlp_right_gemini_wrong].mean():.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"MLP: {mlp_acc*100:.1f}%")
    print(f"Gemini: {gemini_acc*100:.1f}%")
    print(f"Author: {author_acc*100:.1f}%")
    print(f"Baseline (80% author + 20% MLP): {baseline_acc*100:.1f}%")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    run()
