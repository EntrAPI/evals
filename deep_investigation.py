#!/usr/bin/env python3
"""
Deep investigation of model predictions:
1. Look at actual content of hard samples
2. Train meta-model to predict which model to trust
3. Analyze feature differences
4. Check for potential mislabeling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from collections import defaultdict, Counter

EMBEDDINGS_FILE = "data/gemma_1b_embeddings.npz"
PAIRS_FILE = "data/linkedin_pairs.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    print("Loading embeddings...")
    data = np.load(EMBEDDINGS_FILE)

    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=DEVICE)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32, device=DEVICE)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=DEVICE)

    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, y_train, X_test_norm, y_test


def load_pairs():
    print("Loading LinkedIn pairs...")
    with open(PAIRS_FILE, 'r') as f:
        data = json.load(f)
    train_pairs = data['train']
    test_pairs = data['test']
    print(f"Loaded {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")
    return train_pairs, test_pairs


# ============== MODELS ==============

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class TinyRecursive(nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        self.hidden_dim = hidden_dim
    def forward(self, x, K=32):
        B = x.shape[0]
        x_proj = self.input_proj(x)
        z = torch.zeros(B, self.hidden_dim, device=x.device)
        for _ in range(K):
            z = z + self.update(torch.cat([z, x_proj], dim=-1))
            z = self.norm(z)
        return self.output(z)


def train_model(model, X_train, y_train, epochs=200, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
    return model


def get_predictions(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)
    return preds.cpu().numpy(), probs.cpu().numpy()


def run():
    print("=" * 70)
    print("DEEP INVESTIGATION")
    print("=" * 70)

    X_train, y_train, X_test, y_test = load_data()
    train_pairs, test_pairs = load_pairs()
    input_dim = X_train.shape[1]
    y_test_np = y_test.cpu().numpy()
    y_train_np = y_train.cpu().numpy()

    n_train = len(y_train)
    n_test = len(y_test)
    print(f"Train: {n_train}, Test: {n_test}")

    # Train models
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    models_config = [
        ("Ridge", lambda: LogisticRegression(input_dim)),
        ("MLP_256", lambda: MLP(input_dim, [256])),
        ("MLP_512_256", lambda: MLP(input_dim, [512, 256])),
        ("TRM_h8", lambda: TinyRecursive(input_dim, hidden_dim=8)),
        ("TRM_h4", lambda: TinyRecursive(input_dim, hidden_dim=4)),
    ]

    all_preds = {}
    all_probs = {}
    all_correct = {}

    for name, model_fn in models_config:
        print(f"Training {name}...")
        preds_runs = []
        probs_runs = []

        for run_i in range(3):
            torch.manual_seed(42 + run_i)
            model = model_fn().to(DEVICE)
            train_model(model, X_train, y_train)
            preds, probs = get_predictions(model, X_test)
            preds_runs.append(preds)
            probs_runs.append(probs)

        # Majority vote
        preds_stack = np.stack(preds_runs)
        majority_preds = (preds_stack.sum(axis=0) >= 2).astype(int)
        avg_probs = np.mean(probs_runs, axis=0)

        all_preds[name] = majority_preds
        all_probs[name] = avg_probs
        all_correct[name] = (majority_preds == y_test_np)

        acc = all_correct[name].mean()
        print(f"  {name}: {acc*100:.1f}%")

    model_names = list(all_preds.keys())

    # Categorize samples
    correct_counts = sum(all_correct[name].astype(int) for name in model_names)
    all_right_mask = (correct_counts == len(model_names))
    all_wrong_mask = (correct_counts == 0)
    contested_mask = ~all_right_mask & ~all_wrong_mask

    all_wrong_indices = np.where(all_wrong_mask)[0]
    contested_indices = np.where(contested_mask)[0]

    # ================================================================
    # PART 1: Examine "all wrong" samples in detail
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 1: EXAMINING 'ALL WRONG' SAMPLES")
    print("=" * 70)

    print(f"\n{len(all_wrong_indices)} samples ALL models got wrong.\n")

    # Analyze engagement metrics
    engagement_analysis = {"true_0_pred_1": [], "true_1_pred_0": []}

    for idx in all_wrong_indices:
        pair = test_pairs[idx]
        true_label = y_test_np[idx]

        # Handle nested structure
        if isinstance(pair.get('post_a'), dict):
            post_a = pair['post_a'].get('text', '')
            post_b = pair['post_b'].get('text', '')
            eng_a = pair['post_a'].get('reactions', 0) + pair['post_a'].get('comments', 0)
            eng_b = pair['post_b'].get('reactions', 0) + pair['post_b'].get('comments', 0)
        else:
            post_a = pair.get('post_a', pair.get('text_a', ''))
            post_b = pair.get('post_b', pair.get('text_b', ''))
            eng_a = pair.get('engagement_a', pair.get('likes_a', 0))
            eng_b = pair.get('engagement_b', pair.get('likes_b', 0))

        # Truncate for display
        post_a_short = post_a[:100].replace('\n', ' ') + "..." if len(post_a) > 100 else post_a.replace('\n', ' ')
        post_b_short = post_b[:100].replace('\n', ' ') + "..." if len(post_b) > 100 else post_b.replace('\n', ' ')

        if true_label == 0:
            engagement_analysis["true_0_pred_1"].append({
                "idx": idx, "eng_a": eng_a, "eng_b": eng_b,
                "post_a": post_a_short, "post_b": post_b_short,
                "ratio": eng_a / (eng_b + 1)
            })
        else:
            engagement_analysis["true_1_pred_0"].append({
                "idx": idx, "eng_a": eng_a, "eng_b": eng_b,
                "post_a": post_a_short, "post_b": post_b_short,
                "ratio": eng_b / (eng_a + 1)
            })

    print(f"True=0 but predicted 1 (models think B is better, but A won): {len(engagement_analysis['true_0_pred_1'])}")
    print(f"True=1 but predicted 0 (models think A is better, but B won): {len(engagement_analysis['true_1_pred_0'])}")

    # Check engagement ratios for "all wrong"
    print("\n--- Engagement ratio analysis for 'all wrong' ---")
    for category, items in engagement_analysis.items():
        if items:
            ratios = [x['ratio'] for x in items]
            print(f"\n{category}:")
            print(f"  Count: {len(items)}")
            print(f"  Ratio (winner/loser) - Mean: {np.mean(ratios):.2f}, Median: {np.median(ratios):.2f}")
            print(f"  Min ratio: {np.min(ratios):.2f}, Max ratio: {np.max(ratios):.2f}")

            # Show cases with close engagement (potential label noise)
            close_cases = [x for x in items if x['ratio'] < 1.5]
            print(f"  Close calls (ratio < 1.5): {len(close_cases)}")

    # Show some examples
    print("\n--- Example 'all wrong' samples with close engagement ---")
    all_items = engagement_analysis["true_0_pred_1"] + engagement_analysis["true_1_pred_0"]
    close_items = sorted([x for x in all_items if x['ratio'] < 2.0], key=lambda x: x['ratio'])

    for item in close_items[:5]:
        idx = item['idx']
        print(f"\nSample {idx}: True={y_test_np[idx]}, Ratio={item['ratio']:.2f}")
        print(f"  Engagement A: {item['eng_a']}, Engagement B: {item['eng_b']}")
        print(f"  Post A: {item['post_a']}")
        print(f"  Post B: {item['post_b']}")
        print(f"  Model confidences:")
        for name in model_names:
            prob = all_probs[name][idx]
            print(f"    {name}: [{prob[0]:.3f}, {prob[1]:.3f}]")

    # ================================================================
    # PART 2: Train meta-model to predict which model to trust
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 2: META-MODEL TO SELECT BEST MODEL")
    print("=" * 70)

    # Create features for meta-model:
    # - Model confidences
    # - Agreement between models
    # - Embedding statistics

    X_test_np = X_test.cpu().numpy()

    meta_features = []
    meta_targets = []

    for idx in range(n_test):
        # Which models got this right?
        correct_models = [name for name in model_names if all_correct[name][idx]]

        if len(correct_models) == 0 or len(correct_models) == len(model_names):
            continue  # Skip all-right and all-wrong (nothing to learn)

        # Features
        features = []

        # Model confidences
        for name in model_names:
            features.extend(all_probs[name][idx])  # [p0, p1] for each model

        # Agreement features
        preds_here = [all_preds[name][idx] for name in model_names]
        features.append(np.mean(preds_here))  # Average prediction
        features.append(np.std(preds_here))   # Disagreement

        # Max confidence across models
        max_confs = [max(all_probs[name][idx]) for name in model_names]
        features.append(np.max(max_confs))
        features.append(np.mean(max_confs))

        # Embedding stats
        emb = X_test_np[idx]
        features.append(np.linalg.norm(emb))
        features.append(np.mean(emb))
        features.append(np.std(emb))
        features.append(np.max(emb))
        features.append(np.min(emb))

        meta_features.append(features)

        # Target: index of a correct model (or -1 if none)
        if correct_models:
            target = model_names.index(correct_models[0])
        else:
            target = -1
        meta_targets.append(target)

    meta_features = np.array(meta_features, dtype=np.float32)
    meta_targets = np.array(meta_targets)

    print(f"Meta-model training data: {len(meta_features)} contested samples")
    print(f"Feature dim: {meta_features.shape[1]}")

    # Train a simple meta-classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    # Filter out -1 targets
    valid_mask = meta_targets >= 0
    X_meta = meta_features[valid_mask]
    y_meta = meta_targets[valid_mask]

    if len(X_meta) > 20:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(clf, X_meta, y_meta, cv=5)
        print(f"Meta-model CV accuracy: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")

        # Feature importance
        clf.fit(X_meta, y_meta)
        feature_names = []
        for name in model_names:
            feature_names.extend([f"{name}_p0", f"{name}_p1"])
        feature_names.extend(["avg_pred", "std_pred", "max_conf", "mean_conf",
                              "emb_norm", "emb_mean", "emb_std", "emb_max", "emb_min"])

        importances = clf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("\nTop feature importances:")
        for i in sorted_idx[:10]:
            print(f"  {feature_names[i]}: {importances[i]:.3f}")

    # ================================================================
    # PART 3: Analyze what distinguishes models' correct predictions
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 3: WHAT MAKES EACH MODEL SUCCEED?")
    print("=" * 70)

    for name in model_names:
        correct_idx = np.where(all_correct[name])[0]
        wrong_idx = np.where(~all_correct[name])[0]

        correct_emb = X_test_np[correct_idx]
        wrong_emb = X_test_np[wrong_idx]

        correct_conf = np.array([max(all_probs[name][i]) for i in correct_idx])
        wrong_conf = np.array([max(all_probs[name][i]) for i in wrong_idx])

        print(f"\n{name}:")
        print(f"  Correct: {len(correct_idx)}, Wrong: {len(wrong_idx)}")
        print(f"  Confidence when correct: {correct_conf.mean():.3f}")
        print(f"  Confidence when wrong: {wrong_conf.mean():.3f}")

        # Embedding norms
        print(f"  Embedding norm (correct): {np.linalg.norm(correct_emb, axis=1).mean():.2f}")
        print(f"  Embedding norm (wrong): {np.linalg.norm(wrong_emb, axis=1).mean():.2f}")

    # ================================================================
    # PART 4: Check for potential mislabeling
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 4: POTENTIAL MISLABELING ANALYSIS")
    print("=" * 70)

    # Samples where ALL models are very confident but wrong
    suspicious = []
    for idx in all_wrong_indices:
        min_conf = min(max(all_probs[name][idx]) for name in model_names)
        avg_conf = np.mean([max(all_probs[name][idx]) for name in model_names])
        if avg_conf > 0.9:  # Very high average confidence
            pair = test_pairs[idx]
            if isinstance(pair.get('post_a'), dict):
                eng_a = pair['post_a'].get('reactions', 0) + pair['post_a'].get('comments', 0)
                eng_b = pair['post_b'].get('reactions', 0) + pair['post_b'].get('comments', 0)
            else:
                eng_a = pair.get('engagement_a', pair.get('likes_a', 0))
                eng_b = pair.get('engagement_b', pair.get('likes_b', 0))
            ratio = eng_a / (eng_b + 1) if y_test_np[idx] == 0 else eng_b / (eng_a + 1)
            suspicious.append({
                "idx": idx,
                "true_label": y_test_np[idx],
                "avg_conf": avg_conf,
                "eng_a": eng_a,
                "eng_b": eng_b,
                "ratio": ratio
            })

    print(f"\nHighly suspicious samples (all models >90% confident but wrong): {len(suspicious)}")

    for item in sorted(suspicious, key=lambda x: -x['avg_conf'])[:10]:
        idx = item['idx']
        print(f"\n  Sample {idx}: True={item['true_label']}, AvgConf={item['avg_conf']:.3f}")
        print(f"    Engagement: A={item['eng_a']}, B={item['eng_b']}, Ratio={item['ratio']:.2f}")

        pair = test_pairs[idx]
        if isinstance(pair.get('post_a'), dict):
            post_a = pair['post_a'].get('text', '')[:150].replace('\n', ' ')
            post_b = pair['post_b'].get('text', '')[:150].replace('\n', ' ')
        else:
            post_a = pair.get('post_a', pair.get('text_a', ''))[:150].replace('\n', ' ')
            post_b = pair.get('post_b', pair.get('text_b', ''))[:150].replace('\n', ' ')
        print(f"    Post A: {post_a}")
        print(f"    Post B: {post_b}")

    # ================================================================
    # PART 5: Ensemble with learned confidence threshold
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 5: SMART ENSEMBLE STRATEGIES")
    print("=" * 70)

    # Strategy 1: Only trust high-confidence predictions
    for conf_threshold in [0.6, 0.7, 0.8, 0.9]:
        correct = 0
        total = 0
        abstain = 0

        for idx in range(n_test):
            # Get best model's prediction (highest confidence)
            best_model = None
            best_conf = 0
            for name in model_names:
                conf = max(all_probs[name][idx])
                if conf > best_conf:
                    best_conf = conf
                    best_model = name

            if best_conf >= conf_threshold:
                pred = all_preds[best_model][idx]
                if pred == y_test_np[idx]:
                    correct += 1
                total += 1
            else:
                abstain += 1

        if total > 0:
            print(f"Confidence threshold {conf_threshold}: {correct}/{total} = {correct/total*100:.1f}% (abstain: {abstain})")

    # Strategy 2: Use agreement as confidence
    print("\n--- Agreement-based ensemble ---")
    for agree_threshold in [3, 4, 5]:
        correct = 0
        total = 0

        for idx in range(n_test):
            preds_here = [all_preds[name][idx] for name in model_names]
            vote = sum(preds_here)

            if vote >= agree_threshold or vote <= (len(model_names) - agree_threshold):
                # Strong agreement
                pred = 1 if vote > len(model_names) / 2 else 0
                if pred == y_test_np[idx]:
                    correct += 1
                total += 1

        if total > 0:
            print(f"Agreement >= {agree_threshold}: {correct}/{total} = {correct/total*100:.1f}% ({total} samples)")

    # Strategy 3: Weighted by past accuracy (simulated)
    print("\n--- Accuracy-weighted ensemble ---")
    # Use train set to estimate model weights
    train_preds = {}
    for name, model_fn in models_config:
        torch.manual_seed(42)
        model = model_fn().to(DEVICE)
        train_model(model, X_train, y_train)
        preds, _ = get_predictions(model, X_train)
        train_preds[name] = preds

    train_accs = {name: (train_preds[name] == y_train_np).mean() for name in model_names}
    print("Train accuracies (for weighting):")
    for name, acc in train_accs.items():
        print(f"  {name}: {acc*100:.1f}%")

    # Weighted vote
    weights = np.array([train_accs[name] for name in model_names])
    weights = weights / weights.sum()

    weighted_preds = np.zeros(n_test)
    for i, name in enumerate(model_names):
        weighted_preds += weights[i] * all_probs[name][:, 1]

    final_preds = (weighted_preds > 0.5).astype(int)
    acc = (final_preds == y_test_np).mean()
    print(f"Weighted ensemble accuracy: {acc*100:.1f}%")

    # ================================================================
    # PART 6: Error correlation analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 6: ERROR CORRELATION MATRIX")
    print("=" * 70)

    # Build error matrix: which models make errors on same samples?
    error_matrix = np.zeros((len(model_names), len(model_names)))
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            # Count samples where both are wrong
            both_wrong = (~all_correct[name1]) & (~all_correct[name2])
            error_matrix[i, j] = both_wrong.sum()

    print("\nError overlap matrix (# samples both wrong):")
    print(f"{'':12s}", end="")
    for name in model_names:
        print(f"{name[:8]:>9s}", end="")
    print()
    for i, name1 in enumerate(model_names):
        print(f"{name1:12s}", end="")
        for j, name2 in enumerate(model_names):
            print(f"{int(error_matrix[i,j]):9d}", end="")
        print()

    # Find most complementary pair
    print("\nMost complementary pairs (least error overlap):")
    pairs_overlap = []
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i < j:
                pairs_overlap.append((name1, name2, error_matrix[i, j]))

    for name1, name2, overlap in sorted(pairs_overlap, key=lambda x: x[2])[:3]:
        print(f"  {name1} + {name2}: {int(overlap)} shared errors")


if __name__ == "__main__":
    run()
