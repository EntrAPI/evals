#!/usr/bin/env python3
"""
Create a larger sample for statistical testing.
Focus on test set size for significance testing.
"""

from datasets import load_dataset
from collections import defaultdict
import json
import random

def run():
    print("=" * 70)
    print("Creating Larger Twitter Sample for Statistical Testing")
    print("=" * 70)

    print("\nLoading dataset (streaming)...")
    ds = load_dataset("enryu43/twitter100m_tweets", split="train", streaming=True)

    user_tweets = defaultdict(list)
    total_collected = 0
    target_tweets = 100000  # More tweets

    print(f"\nCollecting {target_tweets:,} tweets...")

    for i, sample in enumerate(ds):
        if sample['likes'] == 0 and sample['retweets'] == 0:
            continue
        if len(sample['tweet']) < 20:
            continue

        user_tweets[sample['user']].append({
            'id': sample['id'],
            'text': sample['tweet'],
            'likes': sample['likes'],
            'retweets': sample['retweets'],
            'date': sample['date'],
        })

        total_collected += 1

        if total_collected % 20000 == 0:
            users_with_multi = sum(1 for u in user_tweets if len(user_tweets[u]) >= 2)
            print(f"  Collected {total_collected:,} tweets, {len(user_tweets):,} users, {users_with_multi:,} with 2+ tweets")

        if total_collected >= target_tweets:
            break

    print(f"\nTotal: {total_collected:,} tweets from {len(user_tweets):,} users")

    users_with_multiple = {u: tweets for u, tweets in user_tweets.items() if len(tweets) >= 2}
    print(f"Users with 2+ tweets: {len(users_with_multiple):,}")

    # Create pairs - no per-user limit, but cap total
    print("\nCreating pairs...")
    pairs = []
    min_gap = 1.5
    max_total_pairs = 20000  # Cap total pairs to keep file reasonable

    for user, tweets in users_with_multiple.items():
        for i, t1 in enumerate(tweets):
            for t2 in tweets[i+1:]:
                if len(pairs) >= max_total_pairs:
                    break

                eng1 = t1['likes'] + t1['retweets']
                eng2 = t2['likes'] + t2['retweets']

                if eng1 == 0 and eng2 == 0:
                    continue

                if eng1 > eng2 and eng1 >= eng2 * min_gap:
                    pairs.append({
                        'user': user,
                        'tweet_a': t1,
                        'tweet_b': t2,
                        'label': 0,
                        'engagement_a': eng1,
                        'engagement_b': eng2,
                        'ratio': eng1 / (eng2 + 1),
                    })
                elif eng2 > eng1 and eng2 >= eng1 * min_gap:
                    pairs.append({
                        'user': user,
                        'tweet_a': t1,
                        'tweet_b': t2,
                        'label': 1,
                        'engagement_a': eng1,
                        'engagement_b': eng2,
                        'ratio': eng2 / (eng1 + 1),
                    })
            if len(pairs) >= max_total_pairs:
                break
        if len(pairs) >= max_total_pairs:
            break

    print(f"Created {len(pairs):,} pairs")

    # Shuffle and split - larger test set
    random.seed(42)
    random.shuffle(pairs)

    # Use 2000 test, rest for train
    n_test = min(2000, len(pairs) // 3)
    n_train = len(pairs) - n_test

    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    print(f"\nTrain: {len(train_pairs):,}, Test: {len(test_pairs):,}")

    # Analyze test set composition
    ratios = [p['ratio'] for p in test_pairs]
    print(f"\nTest set ratio distribution:")
    for thresh in [1.5, 2.0, 2.5, 3.0, 5.0]:
        count = sum(1 for r in ratios if r >= thresh)
        print(f"  >= {thresh}x: {count} ({count/len(ratios)*100:.1f}%)")

    # Save
    output = {
        'train': train_pairs,
        'test': test_pairs,
        'seed': 42,
        'min_gap': min_gap,
        'source': 'enryu43/twitter100m_tweets',
    }

    output_file = 'data/twitter_large_sample.json'
    with open(output_file, 'w') as f:
        json.dump(output, f)

    import os
    size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nSaved to {output_file} ({size:.1f} MB)")


if __name__ == "__main__":
    run()
