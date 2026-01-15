#!/usr/bin/env python3
"""
Embedding extraction + logistic regression using local Gemma 3 1B.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from linkedin_data import get_pairs
import random

MODEL_ID = "google/gemma-3-1b-it"

# Global model and tokenizer
model = None
tokenizer = None


def load_model():
    global model, tokenizer
    print(f"Loading {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def get_embedding(text: str, max_length: int = 512) -> np.ndarray:
    """Extract embedding from model's last hidden state (mean pooling)."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get last hidden state
        hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        # Mean pool over sequence
        embedding = hidden.mean(dim=1).squeeze().float().cpu().numpy()

    return embedding


def extract_pair_features(pairs, desc: str = ""):
    """Extract embeddings for all pairs."""
    features = []
    labels = []

    for i, pair in enumerate(pairs):
        emb_a = get_embedding(pair.post_a.text)
        emb_b = get_embedding(pair.post_b.text)

        # Concatenate embeddings
        feature = np.concatenate([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b])
        features.append(feature)
        labels.append(pair.label)

        if (i + 1) % 50 == 0:
            print(f"  {desc} [{i+1}/{len(pairs)}]")

    return np.array(features), np.array(labels)


EMBEDDINGS_FILE = "data/gemma_1b_embeddings.npz"


def run():
    print("=" * 70)
    print("Gemma 3 1B - Embedding + Logistic Regression")
    print("=" * 70)

    train_pairs, test_pairs = get_pairs()
    print(f"\nTrain: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Check if embeddings already exist
    import os
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"\nLoading cached embeddings from {EMBEDDINGS_FILE}...")
        data = np.load(EMBEDDINGS_FILE)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
    else:
        load_model()

        # Extract embeddings
        print("\nExtracting train embeddings...")
        X_train, y_train = extract_pair_features(train_pairs, "Train")

        print("\nExtracting test embeddings...")
        X_test, y_test = extract_pair_features(test_pairs, "Test")

        # Save embeddings
        print(f"\nSaving embeddings to {EMBEDDINGS_FILE}...")
        np.savez(EMBEDDINGS_FILE, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print(f"\nFeature shape: {X_train.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    print("\nTraining logistic regression...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)

    print(f"\n{'='*60}")
    print("RESULTS")
    print("=" * 60)
    print(f"  Train accuracy: {train_acc*100:.1f}%")
    print(f"  Test accuracy:  {test_acc*100:.1f}%")
    print(f"\n  Random baseline: 50.0%")

    # Try different C values
    print("\n" + "-" * 60)
    print("Trying different regularization strengths...")
    print("-" * 60)

    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(max_iter=1000, C=C)
        clf.fit(X_train_scaled, y_train)
        test_acc = clf.score(X_test_scaled, y_test)
        print(f"  C={C:5.2f}: {test_acc*100:.1f}%")


if __name__ == "__main__":
    random.seed(42)
    run()
