#!/usr/bin/env python3
"""
Create a smaller sample from Twitter dataset for testing.
Samples directly from the HuggingFace stream to avoid memory issues.
"""

from datasets import load_dataset
from collections import defaultdict
import json
import random

def run():
    print("=" * 70)
    print("Creating Small Twitter Same-Author Sample")
    print("=" * 70)

    # Load dataset in streaming mode
    print("\nLoading dataset (streaming)...")
    ds = load_dataset("enryu43/twitter100m_tweets", split="train", streaming=True)

    # Collect tweets by user - target more users with fewer tweets each
    user_tweets = defaultdict(list)
    total_collected = 0
    target_tweets = 50000  # Larger sample for more pairs

    print(f"\nCollecting {target_tweets:,} tweets...")

    for i, sample in enumerate(ds):
        # Skip tweets with no engagement
        if sample['likes'] == 0 and sample['retweets'] == 0:
            continue

        # Skip very short tweets
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

        if total_collected % 5000 == 0:
            users_with_multi = sum(1 for u in user_tweets if len(user_tweets[u]) >= 2)
            print(f"  Collected {total_collected:,} tweets, {len(user_tweets):,} users, {users_with_multi:,} with 2+ tweets")

        if total_collected >= target_tweets:
            break

    print(f"\nTotal collected: {total_collected:,} tweets from {len(user_tweets):,} users")

    # Find users with multiple tweets
    users_with_multiple = {u: tweets for u, tweets in user_tweets.items() if len(tweets) >= 2}
    print(f"Users with 2+ tweets: {len(users_with_multiple):,}")

    # Create pairs with engagement gap - limit pairs per user
    print("\nCreating pairs with engagement gap...")
    pairs = []
    min_gap = 1.5
    max_pairs_per_user = 10  # Limit pairs per user

    for user, tweets in users_with_multiple.items():
        user_pairs = []
        for i, t1 in enumerate(tweets):
            for t2 in tweets[i+1:]:
                if len(user_pairs) >= max_pairs_per_user:
                    break

                eng1 = t1['likes'] + t1['retweets']
                eng2 = t2['likes'] + t2['retweets']

                if eng1 == 0 and eng2 == 0:
                    continue

                # Determine winner
                if eng1 > eng2 and eng1 >= eng2 * min_gap:
                    user_pairs.append({
                        'user': user,
                        'tweet_a': t1,
                        'tweet_b': t2,
                        'label': 0,  # A wins
                        'engagement_a': eng1,
                        'engagement_b': eng2,
                        'ratio': eng1 / (eng2 + 1),
                    })
                elif eng2 > eng1 and eng2 >= eng1 * min_gap:
                    user_pairs.append({
                        'user': user,
                        'tweet_a': t1,
                        'tweet_b': t2,
                        'label': 1,  # B wins
                        'engagement_a': eng1,
                        'engagement_b': eng2,
                        'ratio': eng2 / (eng1 + 1),
                    })
            if len(user_pairs) >= max_pairs_per_user:
                break
        pairs.extend(user_pairs)

    print(f"Created {len(pairs):,} pairs")

    if len(pairs) == 0:
        print("No pairs found!")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(pairs)

    # Target 1600 train, 400 test like LinkedIn
    n_train = min(1600, int(len(pairs) * 0.8))
    n_test = min(400, len(pairs) - n_train)
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:n_train+n_test]

    # If we have more, use them
    if len(pairs) > 2000:
        n_train = 1600
        n_test = 400
        train_pairs = pairs[:n_train]
        test_pairs = pairs[n_train:n_train+n_test]

    print(f"\nTrain: {len(train_pairs):,}, Test: {len(test_pairs):,}")

    # Save
    output = {
        'train': train_pairs,
        'test': test_pairs,
        'seed': 42,
        'min_gap': min_gap,
        'source': 'enryu43/twitter100m_tweets',
    }

    output_file = 'data/twitter_same_author_small.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Check file size
    import os
    size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nSaved to {output_file} ({size:.1f} MB)")

    # Show examples
    print("\n" + "=" * 70)
    print("EXAMPLE PAIRS")
    print("=" * 70)

    for i, p in enumerate(test_pairs[:3]):
        print(f"\n--- Pair {i+1} (User: {p['user']}) ---")
        print(f"Tweet A ({p['engagement_a']} eng): {p['tweet_a']['text'][:80]}...")
        print(f"Tweet B ({p['engagement_b']} eng): {p['tweet_b']['text'][:80]}...")
        print(f"Winner: {'A' if p['label'] == 0 else 'B'} (ratio: {p['ratio']:.1f}x)")


if __name__ == "__main__":
    run()
