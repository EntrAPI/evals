#!/usr/bin/env python3
"""
Download Twitter dataset and create same-author pairs for A/B testing.
"""

from datasets import load_dataset
from collections import defaultdict
import json
import random

def run():
    print("=" * 70)
    print("Downloading Twitter Dataset & Creating Same-Author Pairs")
    print("=" * 70)

    # Load dataset in streaming mode to get a sample
    print("\nLoading dataset (streaming)...")
    ds = load_dataset("enryu43/twitter100m_tweets", split="train", streaming=True)

    # Collect tweets by user
    user_tweets = defaultdict(list)
    total_collected = 0
    target_tweets = 100000  # Collect 100k tweets

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
            'replies': sample['replies'],
            'quotes': sample['quotes'],
            'date': sample['date'],
        })

        total_collected += 1

        if total_collected % 10000 == 0:
            users_with_multi = sum(1 for u in user_tweets if len(user_tweets[u]) >= 2)
            print(f"  Collected {total_collected:,} tweets, {len(user_tweets):,} users, {users_with_multi:,} with 2+ tweets")

        if total_collected >= target_tweets:
            break

    print(f"\nTotal collected: {total_collected:,} tweets from {len(user_tweets):,} users")

    # Find users with multiple tweets
    users_with_multiple = {u: tweets for u, tweets in user_tweets.items() if len(tweets) >= 2}
    print(f"Users with 2+ tweets: {len(users_with_multiple):,}")

    # Count potential pairs
    total_pairs = sum(len(tweets) * (len(tweets) - 1) // 2 for tweets in users_with_multiple.values())
    print(f"Potential same-author pairs: {total_pairs:,}")

    # Create pairs with engagement gap
    print("\nCreating pairs with engagement gap...")
    pairs = []
    min_gap = 1.5  # Winner should have at least 1.5x more engagement

    for user, tweets in users_with_multiple.items():
        # Sort by total engagement
        for i, t1 in enumerate(tweets):
            for t2 in tweets[i+1:]:
                eng1 = t1['likes'] + t1['retweets']
                eng2 = t2['likes'] + t2['retweets']

                if eng1 == 0 and eng2 == 0:
                    continue

                # Determine winner
                if eng1 > eng2 and eng1 >= eng2 * min_gap:
                    pairs.append({
                        'user': user,
                        'tweet_a': t1,
                        'tweet_b': t2,
                        'label': 0,  # A wins
                        'engagement_a': eng1,
                        'engagement_b': eng2,
                        'ratio': eng1 / (eng2 + 1),
                    })
                elif eng2 > eng1 and eng2 >= eng1 * min_gap:
                    pairs.append({
                        'user': user,
                        'tweet_a': t1,
                        'tweet_b': t2,
                        'label': 1,  # B wins
                        'engagement_a': eng1,
                        'engagement_b': eng2,
                        'ratio': eng2 / (eng1 + 1),
                    })

    print(f"Created {len(pairs):,} pairs with {min_gap}x+ engagement gap")

    if len(pairs) == 0:
        print("No pairs found! Try collecting more data.")
        return

    # Show engagement ratio distribution
    ratios = [p['ratio'] for p in pairs]
    print(f"\nEngagement ratio distribution:")
    print(f"  Min: {min(ratios):.1f}x, Max: {max(ratios):.1f}x")
    print(f"  Median: {sorted(ratios)[len(ratios)//2]:.1f}x")

    # Shuffle and split
    random.seed(42)
    random.shuffle(pairs)

    n_train = int(len(pairs) * 0.8)
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    print(f"\nTrain: {len(train_pairs):,}, Test: {len(test_pairs):,}")

    # Save
    output = {
        'train': train_pairs,
        'test': test_pairs,
        'seed': 42,
        'min_gap': min_gap,
        'source': 'enryu43/twitter100m_tweets',
    }

    output_file = 'data/twitter_same_author_pairs.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_file}")

    # Show examples
    print("\n" + "=" * 70)
    print("EXAMPLE PAIRS")
    print("=" * 70)

    for i, p in enumerate(test_pairs[:3]):
        print(f"\n--- Pair {i+1} (User: {p['user']}) ---")
        print(f"Tweet A ({p['engagement_a']} eng): {p['tweet_a']['text'][:100]}...")
        print(f"Tweet B ({p['engagement_b']} eng): {p['tweet_b']['text'][:100]}...")
        print(f"Winner: {'A' if p['label'] == 0 else 'B'} (ratio: {p['ratio']:.1f}x)")


if __name__ == "__main__":
    run()
