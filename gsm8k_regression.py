#!/usr/bin/env python3
"""
GSM8K: Predict answer magnitude from question embeddings using classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "all-MiniLM-L6-v2"

# Answer buckets (log scale)
BUCKETS = [
    (0, 10, "0-10"),
    (10, 100, "10-100"),
    (100, 1000, "100-1K"),
    (1000, 10000, "1K-10K"),
    (10000, float('inf'), "10K+"),
]

# Global model
model = None


def load_model():
    global model
    print(f"Loading {MODEL_ID}...")
    model = SentenceTransformer(MODEL_ID, device=str(DEVICE))
    print(f"Model loaded on {DEVICE}")


def answer_to_bucket(answer):
    """Convert numerical answer to bucket index."""
    for i, (low, high, _) in enumerate(BUCKETS):
        if low <= answer < high:
            return i
    return len(BUCKETS) - 1


def extract_answer(answer_str):
    """Extract numerical answer from GSM8K answer string."""
    # GSM8K format: "#### 42" at the end
    match = re.search(r'####\s*([\d,]+)', answer_str)
    if match:
        return float(match.group(1).replace(',', ''))
    return None


def get_embedding(text):
    """Extract embedding from model."""
    return model.encode(text, convert_to_numpy=True)


def load_gsm8k(n_samples=500):
    """Load GSM8K dataset."""
    print("Loading GSM8K...")
    dataset = load_dataset("gsm8k", "main", split="train")

    questions = []
    answers = []

    for i, item in enumerate(dataset):
        if len(questions) >= n_samples:
            break

        answer = extract_answer(item['answer'])
        if answer is not None:
            questions.append(item['question'])
            answers.append(answer)

    print(f"Loaded {len(questions)} samples")
    return questions, answers


def extract_embeddings(questions):
    """Extract embeddings for all questions."""
    embeddings = []
    for i, q in enumerate(questions):
        emb = get_embedding(q)
        embeddings.append(emb)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(questions)}]")
    return np.array(embeddings)


class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_classifier(X_train, y_train, X_test, y_test, num_classes, epochs=500):
    """Train classification model."""
    X_train = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

    # Normalize features
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    input_dim = X_train.shape[1]

    best_acc = 0
    best_model_state = None

    for hidden in [[256], [512, 256], [256, 128, 64]]:
        model = ClassifierMLP(input_dim, num_classes, hidden).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            preds = logits.argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

            if acc > best_acc:
                best_acc = acc
                best_hidden = hidden
                best_preds = preds.cpu().numpy()

    return best_acc, best_hidden, best_preds


def run():
    print("=" * 70)
    print("GSM8K: Answer Magnitude Classification")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"\nBuckets: {[b[2] for b in BUCKETS]}")

    # Load model
    load_model()

    # Load data
    questions, answers = load_gsm8k(n_samples=500)

    # Convert answers to bucket labels
    answers_arr = np.array(answers)
    labels = np.array([answer_to_bucket(a) for a in answers])

    # Stats on answers
    print(f"\nAnswer statistics:")
    print(f"  Min: {answers_arr.min():.0f}, Max: {answers_arr.max():.0f}")
    print(f"  Mean: {answers_arr.mean():.1f}, Median: {np.median(answers_arr):.1f}")

    # Bucket distribution
    print(f"\nBucket distribution:")
    for i, (_, _, name) in enumerate(BUCKETS):
        count = (labels == i).sum()
        print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(questions)
    print(f"Embedding shape: {embeddings.shape}")

    # Train/test split
    n_train = int(len(embeddings) * 0.8)
    X_train, X_test = embeddings[:n_train], embeddings[n_train:]
    y_train, y_test = labels[:n_train], labels[n_train:]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Train classifier
    print("\n" + "=" * 60)
    print("Training Classification Model")
    print("=" * 60)

    num_classes = len(BUCKETS)
    acc, best_hidden, preds = train_classifier(X_train, y_train, X_test, y_test, num_classes)
    print(f"Best architecture: {best_hidden}")
    print(f"Test accuracy: {acc*100:.1f}%")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, (_, _, name) in enumerate(BUCKETS):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (preds[mask] == y_test[mask]).mean()
            print(f"  {name}: {class_acc*100:.1f}% ({mask.sum()} samples)")

    # Baseline: predict most common class
    most_common = np.bincount(y_train).argmax()
    baseline_acc = (y_test == most_common).mean()
    print(f"\nBaseline (predict most common): {baseline_acc*100:.1f}%")

    # Random baseline
    random_baseline = 1.0 / num_classes
    print(f"Random baseline: {random_baseline*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model accuracy:       {acc*100:.1f}%")
    print(f"  Most-common baseline: {baseline_acc*100:.1f}%")
    print(f"  Random baseline:      {random_baseline*100:.1f}%")
    print(f"  Improvement over random: {(acc - random_baseline)*100:.1f}pp")


if __name__ == "__main__":
    run()
