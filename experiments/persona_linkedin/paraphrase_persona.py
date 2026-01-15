#!/usr/bin/env python3
"""
Paraphrase-based Persona Sampling for LinkedIn MLP.

The key insight: persona sampling works by generating semantically valid variations,
not by numerically perturbing embeddings. Here we:

1. Generate k paraphrases of each post using Gemini 2.0 Flash
2. Embed each paraphrase using sentence-transformers
3. Run all paraphrase combinations through the MLP
4. Score each combination with the persona vector
5. Pick the prediction with the highest persona score

This tests whether selecting among semantically equivalent inputs
based on the model's internal "confidence" signal improves accuracy.
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
from collections import defaultdict

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "linkedin_pairs.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gemini API for paraphrases
API_KEY = "AIzaSyCWrrgydQ-_kaS92B3vvXrstJcZnPgiE20"
GEMINI_MODEL = "gemini-2.0-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"


class MLPWithHidden(nn.Module):
    """MLP that returns hidden states for persona scoring."""
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

    def forward(self, x, return_hidden=False):
        for layer in self.layers:
            x = layer(x)
        hidden = x
        logits = self.final(x)
        if return_hidden:
            return logits, hidden
        return logits


def train_mlp(X, y, hidden_dims=[256, 128], epochs=300):
    """Train MLP with early stopping on validation."""
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


def extract_persona_vector(model, X, y):
    """Extract persona vector from validation predictions."""
    model.eval()
    with torch.no_grad():
        logits, hidden = model(X, return_hidden=True)
        preds = logits.argmax(dim=1)
        correct_mask = (preds == y)

    n_correct = correct_mask.sum().item()
    n_incorrect = (~correct_mask).sum().item()
    print(f"  Validation: {n_correct} correct, {n_incorrect} incorrect")

    if n_incorrect == 0:
        print("  WARNING: No incorrect predictions - persona vector will be weak")
        # Use difference between high and low confidence correct predictions
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values
        median_conf = confidence.median()
        high_conf = hidden[confidence > median_conf].mean(dim=0)
        low_conf = hidden[confidence <= median_conf].mean(dim=0)
        persona_vector = high_conf - low_conf
    else:
        hidden_correct = hidden[correct_mask]
        hidden_incorrect = hidden[~correct_mask]
        persona_vector = hidden_correct.mean(dim=0) - hidden_incorrect.mean(dim=0)

    # Normalize
    persona_vector = persona_vector / (persona_vector.norm() + 1e-8)
    return persona_vector


def call_gemini(prompt, temperature=0.7, max_tokens=1000):
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
    """Generate k paraphrases using Gemini."""
    # Truncate very long posts
    text = text[:800]

    prompt = f"""Rewrite this LinkedIn post in {k} different ways.
Keep the same meaning but vary the wording, tone, and structure.
Separate each version with "---" on its own line.
Do not number them or add any other text.

Original:
{text}

Rewritten versions:"""

    try:
        response = call_gemini(prompt, temperature=0.9)
        versions = [v.strip() for v in response.split("---") if v.strip() and len(v.strip()) > 20]
        # Always include original
        return [text] + versions[:k]
    except Exception as e:
        print(f"    Paraphrase error: {e}")
        return [text]


def run():
    print("=" * 70)
    print("PARAPHRASE-BASED PERSONA SAMPLING")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # ========================================
    # PHASE 1: Load data and create embeddings
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 1: EMBEDDING WITH SENTENCE-TRANSFORMERS")
    print("=" * 70)

    # Load sentence-transformers model
    print("\nLoading embedding model...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_dim = 384  # all-MiniLM-L6-v2 dimension
    print(f"Embedding model: all-MiniLM-L6-v2 (dim={embed_dim})")

    # Load LinkedIn data
    print("\nLoading LinkedIn data...")
    with open(DATA_FILE) as f:
        data = json.load(f)

    train_pairs = data['train']
    test_pairs = data['test']
    print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    # Create embeddings for all pairs
    print("\nGenerating embeddings for train pairs...")
    X_train = []
    y_train = []
    for pair in tqdm(train_pairs):
        text_a = pair['post_a']['text']
        text_b = pair['post_b']['text']
        emb_a = embed_model.encode(text_a, convert_to_numpy=True)
        emb_b = embed_model.encode(text_b, convert_to_numpy=True)
        X_train.append(np.concatenate([emb_a, emb_b]))

        eng_a = pair['post_a']['reactions'] + pair['post_a'].get('comments', 0)
        eng_b = pair['post_b']['reactions'] + pair['post_b'].get('comments', 0)
        y_train.append(0 if eng_a > eng_b else 1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("\nGenerating embeddings for test pairs...")
    X_test = []
    y_test = []
    for pair in tqdm(test_pairs):
        text_a = pair['post_a']['text']
        text_b = pair['post_b']['text']
        emb_a = embed_model.encode(text_a, convert_to_numpy=True)
        emb_b = embed_model.encode(text_b, convert_to_numpy=True)
        X_test.append(np.concatenate([emb_a, emb_b]))

        eng_a = pair['post_a']['reactions'] + pair['post_a'].get('comments', 0)
        eng_b = pair['post_b']['reactions'] + pair['post_b'].get('comments', 0)
        y_test.append(0 if eng_a > eng_b else 1)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"\nEmbedding shapes: Train {X_train.shape}, Test {X_test.shape}")

    # ========================================
    # PHASE 2: Train MLP and extract persona vector
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 2: TRAIN MLP AND EXTRACT PERSONA VECTOR")
    print("=" * 70)

    # Split training into train/val
    n_train = len(X_train)
    n_val = int(n_train * 0.2)
    np.random.seed(42)
    indices = np.random.permutation(n_train)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_train_train = X_train[train_idx]
    y_train_train = y_train[train_idx]
    X_train_val = X_train[val_idx]
    y_train_val = y_train[val_idx]

    print(f"Train-train: {len(X_train_train)}, Train-val: {len(X_train_val)}")

    # Convert to tensors
    X_train_train_t = torch.tensor(X_train_train, dtype=torch.float32, device=DEVICE)
    X_train_val_t = torch.tensor(X_train_val, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_train_train_t = torch.tensor(y_train_train, dtype=torch.long, device=DEVICE)
    y_train_val_t = torch.tensor(y_train_val, dtype=torch.long, device=DEVICE)

    # Normalize
    mean = X_train_train_t.mean(0)
    std = X_train_train_t.std(0) + 1e-8
    X_train_train_norm = (X_train_train_t - mean) / std
    X_train_val_norm = (X_train_val_t - mean) / std
    X_test_norm = (X_test_t - mean) / std

    # Train MLP
    print("\nTraining MLP...")
    torch.manual_seed(42)
    model = train_mlp(X_train_train_norm, y_train_train_t)

    # Baseline accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X_test_norm)
        baseline_preds = logits.argmax(dim=1).cpu().numpy()
    baseline_acc = (baseline_preds == y_test).mean()
    print(f"Baseline MLP accuracy: {baseline_acc*100:.1f}%")

    # Validation accuracy
    with torch.no_grad():
        val_logits = model(X_train_val_norm)
        val_preds = val_logits.argmax(dim=1).cpu().numpy()
    val_acc = (val_preds == y_train_val).mean()
    print(f"Validation accuracy: {val_acc*100:.1f}%")

    # Extract persona vector
    print("\nExtracting persona vector from validation set...")
    persona_vector = extract_persona_vector(model, X_train_val_norm, y_train_val_t)
    print(f"Persona vector norm: {persona_vector.norm().item():.4f}")

    # Analyze persona signal on test set
    with torch.no_grad():
        _, test_hidden = model(X_test_norm, return_hidden=True)
        test_scores = (test_hidden * persona_vector).sum(dim=1)

    correct_mask = (baseline_preds == y_test)
    score_correct = test_scores[torch.tensor(correct_mask)].mean().item()
    score_incorrect = test_scores[torch.tensor(~correct_mask)].mean().item()
    print(f"\nPersona score analysis (test set):")
    print(f"  Correct predictions: {score_correct:.3f}")
    print(f"  Incorrect predictions: {score_incorrect:.3f}")
    print(f"  Separation: {score_correct - score_incorrect:.3f}")

    # ========================================
    # PHASE 3: Paraphrase-based persona sampling
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 3: PARAPHRASE-BASED PERSONA SAMPLING")
    print("=" * 70)

    K = 3  # paraphrases per post
    N_TEST = len(test_pairs)  # full test set

    print(f"\nRunning on {N_TEST} test pairs with k={K} paraphrases per post...")
    print(f"This will generate {N_TEST * 2} paraphrase requests to Gemini...")

    # Cache paraphrases to avoid redundant API calls
    paraphrase_cache = {}

    correct_persona = 0
    correct_baseline = 0

    # Track detailed statistics
    flipped_to_correct = 0
    flipped_to_incorrect = 0
    stayed_correct = 0
    stayed_incorrect = 0

    score_improvements = []

    for i in tqdm(range(N_TEST)):
        pair = test_pairs[i]
        text_a = pair['post_a']['text']
        text_b = pair['post_b']['text']
        label = y_test[i]

        # Generate paraphrases (with caching)
        if text_a not in paraphrase_cache:
            paraphrase_cache[text_a] = generate_paraphrases(text_a, k=K)
            time.sleep(0.05)
        if text_b not in paraphrase_cache:
            paraphrase_cache[text_b] = generate_paraphrases(text_b, k=K)
            time.sleep(0.05)

        paras_a = paraphrase_cache[text_a]
        paras_b = paraphrase_cache[text_b]

        # Evaluate all combinations
        best_score = float('-inf')
        best_pred = None
        orig_score = None
        orig_pred = None

        for j, pa in enumerate(paras_a):
            for k_idx, pb in enumerate(paras_b):
                # Embed
                emb_a = embed_model.encode(pa, convert_to_numpy=True)
                emb_b = embed_model.encode(pb, convert_to_numpy=True)
                emb = np.concatenate([emb_a, emb_b])

                # Normalize and predict
                emb_t = torch.tensor(emb, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                emb_norm = (emb_t - mean) / std

                with torch.no_grad():
                    logits, hidden = model(emb_norm, return_hidden=True)
                    pred = logits.argmax(dim=1).item()
                    score = (hidden * persona_vector).sum().item()

                # Track original (j=0, k_idx=0)
                if j == 0 and k_idx == 0:
                    orig_score = score
                    orig_pred = pred

                if score > best_score:
                    best_score = score
                    best_pred = pred

        # Record results
        if best_pred == label:
            correct_persona += 1
        if baseline_preds[i] == label:
            correct_baseline += 1

        # Track flips
        baseline_correct = (baseline_preds[i] == label)
        persona_correct = (best_pred == label)

        if baseline_correct and persona_correct:
            stayed_correct += 1
        elif not baseline_correct and not persona_correct:
            stayed_incorrect += 1
        elif not baseline_correct and persona_correct:
            flipped_to_correct += 1
        else:
            flipped_to_incorrect += 1

        # Track score improvement
        if orig_score is not None:
            score_improvements.append(best_score - orig_score)

    # ========================================
    # RESULTS
    # ========================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    persona_acc = correct_persona / N_TEST
    baseline_acc_check = correct_baseline / N_TEST

    print(f"\nAccuracy:")
    print(f"  Baseline MLP: {baseline_acc_check*100:.1f}%")
    print(f"  Persona sampling (k={K}): {persona_acc*100:.1f}%")
    print(f"  Improvement: {(persona_acc - baseline_acc_check)*100:+.1f}pp")

    print(f"\nPrediction changes:")
    print(f"  Stayed correct: {stayed_correct}")
    print(f"  Stayed incorrect: {stayed_incorrect}")
    print(f"  Flipped to correct: {flipped_to_correct}")
    print(f"  Flipped to incorrect: {flipped_to_incorrect}")
    print(f"  Net flips: {flipped_to_correct - flipped_to_incorrect:+d}")

    if score_improvements:
        print(f"\nPersona score changes:")
        print(f"  Mean improvement: {np.mean(score_improvements):.3f}")
        print(f"  Max improvement: {np.max(score_improvements):.3f}")
        print(f"  % improved: {100 * np.mean(np.array(score_improvements) > 0):.1f}%")

    # ========================================
    # ORACLE ANALYSIS
    # ========================================
    print("\n" + "=" * 70)
    print("ORACLE ANALYSIS: What if we could always pick correctly?")
    print("=" * 70)

    # Re-run to check if correct answer exists among paraphrases
    oracle_correct = 0
    has_correct_option = 0

    for i in range(min(100, N_TEST)):  # Sample for speed
        pair = test_pairs[i]
        text_a = pair['post_a']['text']
        text_b = pair['post_b']['text']
        label = y_test[i]

        paras_a = paraphrase_cache.get(text_a, [text_a])
        paras_b = paraphrase_cache.get(text_b, [text_b])

        found_correct = False
        for pa in paras_a:
            for pb in paras_b:
                emb_a = embed_model.encode(pa, convert_to_numpy=True)
                emb_b = embed_model.encode(pb, convert_to_numpy=True)
                emb = np.concatenate([emb_a, emb_b])
                emb_t = torch.tensor(emb, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                emb_norm = (emb_t - mean) / std

                with torch.no_grad():
                    pred = model(emb_norm).argmax(dim=1).item()

                if pred == label:
                    found_correct = True
                    break
            if found_correct:
                break

        if found_correct:
            has_correct_option += 1
            oracle_correct += 1
        elif baseline_preds[i] == label:
            oracle_correct += 1

    print(f"\nOn first 100 test samples:")
    print(f"  Baseline correct: {(baseline_preds[:100] == y_test[:100]).sum()}")
    print(f"  Has correct option among paraphrases: {has_correct_option}")
    print(f"  Oracle (always pick correct if available): {oracle_correct}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Baseline MLP accuracy: {baseline_acc_check*100:.1f}%
Persona sampling accuracy: {persona_acc*100:.1f}%
Change: {(persona_acc - baseline_acc_check)*100:+.1f}pp

Net prediction flips: {flipped_to_correct - flipped_to_incorrect:+d}
  - Flipped to correct: {flipped_to_correct}
  - Flipped to incorrect: {flipped_to_incorrect}

Interpretation:
- If positive improvement: persona sampling helps select better paraphrases
- If negative: persona score doesn't predict correctness for this task
- If near zero with many flips: persona sampling changes predictions but randomly
""")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    run()
