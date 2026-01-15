#!/usr/bin/env python3
"""
Create clean dataset:
- NO replies (exclude tweets starting with @)
- More users (collect more tweets)
- Split by user (no leakage)
"""

from datasets import load_dataset
from collections import defaultdict
import json
import random

def is_reply(text):
    """Check if tweet is a reply (starts with @mention)."""
    return text.strip().startswith('@')

def run():
    print("=" * 70)
    print("Creating Clean Dataset (No Replies, More Users)")
    print("=" * 70)

    print("\nLoading dataset (streaming)...")
    ds = load_dataset("enryu43/twitter100m_tweets", split="train", streaming=True)

    user_tweets = defaultdict(list)
    total_collected = 0
    replies_skipped = 0
    target_tweets = 500000  # Collect more to get more users

    print(f"\nCollecting up to {target_tweets:,} tweets (excluding replies)...")

    for i, sample in enumerate(ds):
        # Skip no engagement
        if sample['likes'] == 0 and sample['retweets'] == 0:
            continue

        # Skip short tweets
        if len(sample['tweet']) < 20:
            continue

        # Skip replies
        if is_reply(sample['tweet']):
            replies_skipped += 1
            continue

        user_tweets[sample['user']].append({
            'id': sample['id'],
            'text': sample['tweet'],
            'likes': sample['likes'],
            'retweets': sample['retweets'],
            'date': sample['date'],
        })

        total_collected += 1

        if total_collected % 100000 == 0:
            users_with_multi = sum(1 for u in user_tweets if len(user_tweets[u]) >= 2)
            print(f"  Collected {total_collected:,} tweets, {len(user_tweets):,} users, {users_with_multi:,} with 2+ tweets")
            print(f"  (Skipped {replies_skipped:,} replies)")

        if total_collected >= target_tweets:
            break

    print(f"\nTotal collected: {total_collected:,} tweets from {len(user_tweets):,} users")
    print(f"Replies skipped: {replies_skipped:,}")

    # Filter to users with 2+ tweets
    users_with_multiple = {u: tweets for u, tweets in user_tweets.items() if len(tweets) >= 2}
    print(f"Users with 2+ tweets: {len(users_with_multiple):,}")

    # Show tweet count distribution
    tweet_counts = [len(tweets) for tweets in users_with_multiple.values()]
    print(f"\nTweets per user: min={min(tweet_counts)}, max={max(tweet_counts)}, median={sorted(tweet_counts)[len(tweet_counts)//2]}")

    # SPLIT BY USER - 80/20
    all_users = list(users_with_multiple.keys())
    random.seed(42)
    random.shuffle(all_users)

    n_train_users = int(len(all_users) * 0.8)
    train_users = set(all_users[:n_train_users])
    test_users = set(all_users[n_train_users:])

    print(f"\nUser split:")
    print(f"  Train users: {len(train_users)}")
    print(f"  Test users: {len(test_users)}")

    # Create pairs
    def create_pairs(users_set, max_pairs_per_user=100):
        pairs = []
        min_gap = 1.5

        for user in users_set:
            tweets = users_with_multiple[user]
            user_pairs = []

            for i, t1 in enumerate(tweets):
                for t2 in tweets[i+1:]:
                    if len(user_pairs) >= max_pairs_per_user:
                        break

                    eng1 = t1['likes'] + t1['retweets']
                    eng2 = t2['likes'] + t2['retweets']

                    if eng1 == 0 and eng2 == 0:
                        continue

                    if eng1 > eng2 and eng1 >= eng2 * min_gap:
                        user_pairs.append({
                            'user': user,
                            'tweet_a': t1,
                            'tweet_b': t2,
                            'label': 0,
                            'engagement_a': eng1,
                            'engagement_b': eng2,
                            'ratio': eng1 / (eng2 + 1),
                        })
                    elif eng2 > eng1 and eng2 >= eng1 * min_gap:
                        user_pairs.append({
                            'user': user,
                            'tweet_a': t1,
                            'tweet_b': t2,
                            'label': 1,
                            'engagement_a': eng1,
                            'engagement_b': eng2,
                            'ratio': eng2 / (eng1 + 1),
                        })

                if len(user_pairs) >= max_pairs_per_user:
                    break

            pairs.extend(user_pairs)

        random.shuffle(pairs)
        return pairs

    print("\nCreating pairs (no replies)...")
    train_pairs = create_pairs(train_users)
    test_pairs = create_pairs(test_users)

    print(f"  Train pairs: {len(train_pairs):,}")
    print(f"  Test pairs: {len(test_pairs):,}")

    # Verify no leakage
    train_tweet_ids = set()
    for p in train_pairs:
        train_tweet_ids.add(p['tweet_a']['id'])
        train_tweet_ids.add(p['tweet_b']['id'])

    test_tweet_ids = set()
    for p in test_pairs:
        test_tweet_ids.add(p['tweet_a']['id'])
        test_tweet_ids.add(p['tweet_b']['id'])

    overlap = train_tweet_ids & test_tweet_ids
    print(f"\nLeakage check: {len(overlap)} overlapping tweets (should be 0)")

    # Verify no replies in data
    reply_count = 0
    for p in train_pairs + test_pairs:
        if is_reply(p['tweet_a']['text']) or is_reply(p['tweet_b']['text']):
            reply_count += 1
    print(f"Replies in pairs: {reply_count} (should be 0)")

    # Show ratio distribution
    print(f"\nTest set ratio distribution:")
    ratios = [p['ratio'] for p in test_pairs]
    for thresh in [1.5, 2.0, 2.5, 3.0, 5.0]:
        count = sum(1 for r in ratios if r >= thresh)
        print(f"  >= {thresh}x: {count} ({count/len(ratios)*100:.1f}%)")

    # Save
    output = {
        'train': train_pairs,
        'test': test_pairs,
        'seed': 42,
        'min_gap': 1.5,
        'source': 'enryu43/twitter100m_tweets',
        'split_method': 'by_user',
        'excluded_replies': True,
    }

    output_file = 'data/twitter_clean_no_replies.json'
    with open(output_file, 'w') as f:
        json.dump(output, f)

    import os
    size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nSaved to {output_file} ({size:.1f} MB)")


if __name__ == "__main__":
    run()
