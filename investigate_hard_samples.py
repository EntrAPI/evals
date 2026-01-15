#!/usr/bin/env python3
"""
Deep investigation of the 10.8% of samples where ALL models fail.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import re
from collections import Counter

EMBEDDINGS_FILE = "data/twitter_qwen_embeddings.npz"
PAIRS_FILE = "data/twitter_same_author_small.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def extract_features(text):
    """Extract content features from a tweet."""
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'has_url': 'http' in text.lower() or 't.co' in text,
        'has_hashtag': '#' in text,
        'has_mention': '@' in text,
        'has_emoji': bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text)),
        'has_media_link': 'pic.twitter' in text or 'photo' in text.lower() or 'video' in text.lower(),
        'is_reply': text.startswith('@'),
        'has_numbers': bool(re.search(r'\d', text)),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'question': '?' in text,
        'exclamation': '!' in text,
    }


def detect_language(text):
    """Simple language detection based on character patterns."""
    # Check for CJK characters
    if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text):
        return 'CJK'
    # Check for Cyrillic
    if re.search(r'[\u0400-\u04ff]', text):
        return 'Cyrillic'
    # Check for Arabic
    if re.search(r'[\u0600-\u06ff]', text):
        return 'Arabic'
    # Check for accented characters (European languages)
    if re.search(r'[àáâãäåèéêëìíîïòóôõöùúûüñçß]', text.lower()):
        return 'European'
    return 'English'


def run():
    print("=" * 80)
    print("DEEP INVESTIGATION: HARD SAMPLES (ALL MODELS WRONG)")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    data = np.load(EMBEDDINGS_FILE)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    with open(PAIRS_FILE, 'r') as f:
        pairs_data = json.load(f)
    test_pairs = pairs_data['test']

    # Quick model training to identify hard samples
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    mean = X_train_t.mean(dim=0)
    std = X_train_t.std(dim=0) + 1e-8
    X_train_norm = (X_train_t - mean) / std
    X_test_norm = (X_test_t - mean) / std
    input_dim = X_train_norm.shape[1]

    # Train multiple models to find consensus
    print("Training models to identify hard samples...")
    all_preds = []

    for seed in range(5):
        torch.manual_seed(42 + seed)
        model = MLP(input_dim, [256, 128]).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for _ in range(300):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_norm), y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test_norm).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)

    # Find samples all models get wrong
    pred_matrix = np.stack(all_preds)
    correct_matrix = pred_matrix == y_test
    correct_counts = correct_matrix.sum(axis=0)

    hard_indices = np.where(correct_counts == 0)[0]
    easy_indices = np.where(correct_counts == 5)[0]

    print(f"\nFound {len(hard_indices)} hard samples (0/5 models correct)")
    print(f"Found {len(easy_indices)} easy samples (5/5 models correct)")

    # ========================================================================
    # DETAILED ANALYSIS OF HARD SAMPLES
    # ========================================================================
    print("\n" + "=" * 80)
    print("DETAILED HARD SAMPLE ANALYSIS")
    print("=" * 80)

    hard_samples = []
    for idx in hard_indices:
        pair = test_pairs[idx]
        true_label = y_test[idx]

        tweet_a = pair['tweet_a']['text']
        tweet_b = pair['tweet_b']['text']
        eng_a = pair['engagement_a']
        eng_b = pair['engagement_b']
        ratio = pair['ratio']

        winner_text = tweet_a if true_label == 0 else tweet_b
        loser_text = tweet_b if true_label == 0 else tweet_a
        winner_eng = eng_a if true_label == 0 else eng_b
        loser_eng = eng_b if true_label == 0 else eng_a

        hard_samples.append({
            'idx': idx,
            'label': true_label,
            'ratio': ratio,
            'winner_eng': winner_eng,
            'loser_eng': loser_eng,
            'winner_text': winner_text,
            'loser_text': loser_text,
            'winner_features': extract_features(winner_text),
            'loser_features': extract_features(loser_text),
            'winner_lang': detect_language(winner_text),
            'loser_lang': detect_language(loser_text),
            'winner_len': len(winner_text),
            'loser_len': len(loser_text),
        })

    # Print each hard sample in detail
    for i, s in enumerate(hard_samples):
        print(f"\n{'='*80}")
        print(f"HARD SAMPLE #{i+1} (idx={s['idx']})")
        print(f"{'='*80}")
        print(f"Engagement ratio: {s['ratio']:.2f}x")
        print(f"Winner engagement: {s['winner_eng']}, Loser engagement: {s['loser_eng']}")
        print(f"Winner language: {s['winner_lang']}, Loser language: {s['loser_lang']}")
        print(f"Winner length: {s['winner_len']} chars, Loser length: {s['loser_len']} chars")

        print(f"\n--- WINNER (got more engagement) ---")
        print(f"{s['winner_text']}")
        wf = s['winner_features']
        print(f"  [URL:{wf['has_url']}, #:{wf['has_hashtag']}, @:{wf['has_mention']}, emoji:{wf['has_emoji']}, reply:{wf['is_reply']}]")

        print(f"\n--- LOSER (got less engagement) ---")
        print(f"{s['loser_text']}")
        lf = s['loser_features']
        print(f"  [URL:{lf['has_url']}, #:{lf['has_hashtag']}, @:{lf['has_mention']}, emoji:{lf['has_emoji']}, reply:{lf['is_reply']}]")

        # Why might models have failed?
        print(f"\n--- POSSIBLE FAILURE REASONS ---")
        reasons = []

        if s['ratio'] < 2.0:
            reasons.append(f"Low ratio ({s['ratio']:.1f}x) - weak signal")

        if s['winner_lang'] != 'English' or s['loser_lang'] != 'English':
            reasons.append(f"Non-English content (winner={s['winner_lang']}, loser={s['loser_lang']})")

        if s['winner_features']['is_reply'] or s['loser_features']['is_reply']:
            reasons.append("Reply tweet (context-dependent)")

        if s['loser_len'] > s['winner_len'] * 1.5:
            reasons.append(f"Loser is longer ({s['loser_len']} vs {s['winner_len']} chars) - models may prefer length")

        if s['loser_features']['has_hashtag'] and not s['winner_features']['has_hashtag']:
            reasons.append("Loser has hashtags, winner doesn't")

        if s['loser_features']['has_url'] and not s['winner_features']['has_url']:
            reasons.append("Loser has URL, winner doesn't")

        if s['winner_eng'] <= 3:
            reasons.append(f"Very low absolute engagement ({s['winner_eng']}) - noisy signal")

        if not reasons:
            reasons.append("No obvious pattern - genuinely unpredictable?")

        for r in reasons:
            print(f"  • {r}")

    # ========================================================================
    # AGGREGATE STATISTICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS: HARD vs EASY SAMPLES")
    print("=" * 80)

    # Collect stats for easy samples too
    easy_samples = []
    for idx in easy_indices[:50]:  # Sample 50 easy ones
        pair = test_pairs[idx]
        true_label = y_test[idx]
        winner_text = pair['tweet_a']['text'] if true_label == 0 else pair['tweet_b']['text']
        loser_text = pair['tweet_b']['text'] if true_label == 0 else pair['tweet_a']['text']

        easy_samples.append({
            'ratio': pair['ratio'],
            'winner_eng': pair['engagement_a'] if true_label == 0 else pair['engagement_b'],
            'winner_len': len(winner_text),
            'loser_len': len(loser_text),
            'winner_lang': detect_language(winner_text),
            'winner_features': extract_features(winner_text),
            'loser_features': extract_features(loser_text),
        })

    # Compare distributions
    print("\n{:<30} {:>15} {:>15}".format("Metric", "Hard (n={})".format(len(hard_samples)), "Easy (n={})".format(len(easy_samples))))
    print("-" * 60)

    hard_ratios = [s['ratio'] for s in hard_samples]
    easy_ratios = [s['ratio'] for s in easy_samples]
    print("{:<30} {:>15.2f} {:>15.2f}".format("Mean engagement ratio", np.mean(hard_ratios), np.mean(easy_ratios)))
    print("{:<30} {:>15.2f} {:>15.2f}".format("Median engagement ratio", np.median(hard_ratios), np.median(easy_ratios)))

    hard_eng = [s['winner_eng'] for s in hard_samples]
    easy_eng = [s['winner_eng'] for s in easy_samples]
    print("{:<30} {:>15.1f} {:>15.1f}".format("Mean winner engagement", np.mean(hard_eng), np.mean(easy_eng)))

    hard_wlen = [s['winner_len'] for s in hard_samples]
    easy_wlen = [s['winner_len'] for s in easy_samples]
    print("{:<30} {:>15.1f} {:>15.1f}".format("Mean winner length (chars)", np.mean(hard_wlen), np.mean(easy_wlen)))

    # Language distribution
    hard_langs = Counter([s['winner_lang'] for s in hard_samples])
    easy_langs = Counter([s['winner_lang'] for s in easy_samples])
    print("\nWinner language distribution:")
    print(f"  Hard: {dict(hard_langs)}")
    print(f"  Easy: {dict(easy_langs)}")

    # Feature comparison
    print("\nContent features (% of samples with feature):")
    features_to_check = ['has_url', 'has_hashtag', 'has_mention', 'has_emoji', 'is_reply']

    for feat in features_to_check:
        hard_pct = sum(1 for s in hard_samples if s['winner_features'][feat]) / len(hard_samples) * 100
        easy_pct = sum(1 for s in easy_samples if s['winner_features'][feat]) / len(easy_samples) * 100
        print(f"  {feat:<20}: Hard {hard_pct:>5.1f}%  Easy {easy_pct:>5.1f}%")

    # Length comparison (winner vs loser)
    print("\nWinner vs Loser length:")
    hard_winner_longer = sum(1 for s in hard_samples if s['winner_len'] > s['loser_len']) / len(hard_samples) * 100
    easy_winner_longer = sum(1 for s in easy_samples if s['winner_len'] > s['loser_len']) / len(easy_samples) * 100
    print(f"  Winner longer: Hard {hard_winner_longer:.1f}%  Easy {easy_winner_longer:.1f}%")

    # ========================================================================
    # CATEGORIZE FAILURE MODES
    # ========================================================================
    print("\n" + "=" * 80)
    print("FAILURE MODE CATEGORIZATION")
    print("=" * 80)

    categories = {
        'low_ratio': [],      # ratio < 2.0
        'non_english': [],    # non-English content
        'reply': [],          # reply tweets
        'low_engagement': [], # winner has < 5 engagement
        'loser_longer': [],   # loser significantly longer
        'other': [],
    }

    for s in hard_samples:
        categorized = False

        if s['ratio'] < 2.0:
            categories['low_ratio'].append(s['idx'])
            categorized = True

        if s['winner_lang'] != 'English':
            categories['non_english'].append(s['idx'])
            categorized = True

        if s['winner_features']['is_reply']:
            categories['reply'].append(s['idx'])
            categorized = True

        if s['winner_eng'] < 5:
            categories['low_engagement'].append(s['idx'])
            categorized = True

        if s['loser_len'] > s['winner_len'] * 1.3:
            categories['loser_longer'].append(s['idx'])
            categorized = True

        if not categorized:
            categories['other'].append(s['idx'])

    print("\nFailure mode breakdown (samples can be in multiple categories):")
    for cat, indices in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<20}: {len(indices):>3} samples ({len(indices)/len(hard_samples)*100:>5.1f}%)")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
Key findings about the {len(hard_samples)} hard samples:

1. LOW ENGAGEMENT RATIO: {len(categories['low_ratio'])}/{len(hard_samples)} ({len(categories['low_ratio'])/len(hard_samples)*100:.0f}%)
   - When ratio < 2x, the signal is too weak to reliably predict

2. LOW ABSOLUTE ENGAGEMENT: {len(categories['low_engagement'])}/{len(hard_samples)} ({len(categories['low_engagement'])/len(hard_samples)*100:.0f}%)
   - Winner has < 5 total engagement - very noisy signal

3. NON-ENGLISH CONTENT: {len(categories['non_english'])}/{len(hard_samples)} ({len(categories['non_english'])/len(hard_samples)*100:.0f}%)
   - Model embeddings may be weaker for non-English text

4. LOSER IS LONGER: {len(categories['loser_longer'])}/{len(hard_samples)} ({len(categories['loser_longer'])/len(hard_samples)*100:.0f}%)
   - Models may have a bias toward longer content

5. REPLY TWEETS: {len(categories['reply'])}/{len(hard_samples)} ({len(categories['reply'])/len(hard_samples)*100:.0f}%)
   - Context-dependent; can't judge without seeing original tweet

CONCLUSION: Most failures are due to weak/noisy signals (low ratio + low engagement).
These may be fundamentally unpredictable from content alone.
""")


if __name__ == "__main__":
    run()
