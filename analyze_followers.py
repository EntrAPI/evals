#!/usr/bin/env python3
"""
Analyze high-ratio wrong predictions - check followers and post length.
"""

import json
import csv
import numpy as np
from collections import defaultdict

PAIRS_FILE = "data/linkedin_pairs.json"
INFLUENCERS_FILE = "data/influencers_data.csv"


def load_influencer_followers():
    """Load follower counts from influencers data."""
    followers = {}
    with open(INFLUENCERS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('name', '').strip()
            follower_count = row.get('followers', '0')
            try:
                follower_count = float(follower_count) if follower_count else 0
            except:
                follower_count = 0
            if name:
                followers[name] = follower_count
    return followers


def load_pairs():
    with open(PAIRS_FILE, 'r') as f:
        data = json.load(f)
    return data['train'], data['test']


def get_engagement(pair):
    if isinstance(pair.get('post_a'), dict):
        eng_a = pair['post_a'].get('reactions', 0) + pair['post_a'].get('comments', 0)
        eng_b = pair['post_b'].get('reactions', 0) + pair['post_b'].get('comments', 0)
    else:
        eng_a = pair.get('engagement_a', 0)
        eng_b = pair.get('engagement_b', 0)
    return eng_a, eng_b


def get_author(pair):
    if isinstance(pair.get('post_a'), dict):
        return pair['post_a'].get('author', ''), pair['post_b'].get('author', '')
    return '', ''


def get_text(pair):
    if isinstance(pair.get('post_a'), dict):
        return pair['post_a'].get('text', ''), pair['post_b'].get('text', '')
    return pair.get('post_a', ''), pair.get('post_b', '')


def run():
    print("=" * 80)
    print("FOLLOWER & POST LENGTH ANALYSIS")
    print("=" * 80)

    # Load data
    followers_data = load_influencer_followers()
    print(f"Loaded follower data for {len(followers_data)} influencers")

    # Show some sample follower counts
    sorted_by_followers = sorted(followers_data.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 by followers:")
    for name, count in sorted_by_followers:
        print(f"  {name}: {count:,.0f}")

    train_pairs, test_pairs = load_pairs()
    print(f"\nLoaded {len(test_pairs)} test pairs")

    # High-ratio wrong samples (from previous analysis)
    # These are samples where all models got it wrong despite clear engagement winner
    high_ratio_wrong_indices = [395, 33, 70, 67, 292, 118, 210, 284, 100, 91,
                                 316, 265, 17, 85, 253, 178, 358, 146, 94, 377]

    print("\n" + "=" * 80)
    print("ANALYSIS OF HIGH-RATIO WRONG PREDICTIONS")
    print("=" * 80)

    results = []

    for idx in high_ratio_wrong_indices:
        if idx >= len(test_pairs):
            continue

        pair = test_pairs[idx]
        eng_a, eng_b = get_engagement(pair)
        author_a, author_b = get_author(pair)
        text_a, text_b = get_text(pair)

        followers_a = followers_data.get(author_a, 0)
        followers_b = followers_data.get(author_b, 0)

        len_a = len(text_a)
        len_b = len(text_b)

        label = pair.get('label', 0)

        # Determine winner and loser
        if label == 0:  # A wins
            winner_eng, loser_eng = eng_a, eng_b
            winner_followers, loser_followers = followers_a, followers_b
            winner_len, loser_len = len_a, len_b
            winner_author, loser_author = author_a, author_b
            winner_text, loser_text = text_a, text_b
        else:  # B wins
            winner_eng, loser_eng = eng_b, eng_a
            winner_followers, loser_followers = followers_b, followers_a
            winner_len, loser_len = len_b, len_a
            winner_author, loser_author = author_b, author_a
            winner_text, loser_text = text_b, text_a

        ratio = winner_eng / (loser_eng + 1)
        follower_ratio = (winner_followers + 1) / (loser_followers + 1)

        results.append({
            'idx': idx,
            'label': label,
            'winner_eng': winner_eng,
            'loser_eng': loser_eng,
            'eng_ratio': ratio,
            'winner_followers': winner_followers,
            'loser_followers': loser_followers,
            'follower_ratio': follower_ratio,
            'winner_len': winner_len,
            'loser_len': loser_len,
            'winner_author': winner_author,
            'loser_author': loser_author,
            'winner_text': winner_text[:100],
            'loser_text': loser_text[:100],
        })

    # Print detailed results
    print(f"\n{'Idx':>5} | {'Eng Ratio':>10} | {'Winner Foll':>12} | {'Loser Foll':>12} | {'Foll Ratio':>10} | {'Win Len':>8} | {'Lose Len':>8}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: -x['eng_ratio']):
        print(f"{r['idx']:>5} | {r['eng_ratio']:>10.1f}x | {r['winner_followers']:>12,.0f} | {r['loser_followers']:>12,.0f} | {r['follower_ratio']:>10.1f}x | {r['winner_len']:>8} | {r['loser_len']:>8}")

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    winner_followers = [r['winner_followers'] for r in results]
    loser_followers = [r['loser_followers'] for r in results]
    follower_ratios = [r['follower_ratio'] for r in results]
    winner_lens = [r['winner_len'] for r in results]
    loser_lens = [r['loser_len'] for r in results]

    print(f"\nFollower counts:")
    print(f"  Winner mean: {np.mean(winner_followers):,.0f}")
    print(f"  Loser mean: {np.mean(loser_followers):,.0f}")
    print(f"  Follower ratio (winner/loser) mean: {np.mean(follower_ratios):.2f}x")
    print(f"  Follower ratio median: {np.median(follower_ratios):.2f}x")

    # How often does winner have more followers?
    winner_more_followers = sum(1 for r in results if r['winner_followers'] > r['loser_followers'])
    print(f"\n  Winner has more followers: {winner_more_followers}/{len(results)} ({winner_more_followers/len(results)*100:.1f}%)")

    print(f"\nPost length:")
    print(f"  Winner mean length: {np.mean(winner_lens):.0f} chars")
    print(f"  Loser mean length: {np.mean(loser_lens):.0f} chars")
    print(f"  Winner shorter: {sum(1 for r in results if r['winner_len'] < r['loser_len'])}/{len(results)}")

    # Show examples where winner has WAY more followers
    print("\n" + "=" * 80)
    print("EXAMPLES: WINNER HAS MORE FOLLOWERS")
    print("=" * 80)

    for r in sorted(results, key=lambda x: -x['follower_ratio'])[:5]:
        print(f"\nSample {r['idx']}: Engagement ratio {r['eng_ratio']:.1f}x")
        print(f"  Winner: {r['winner_author']} ({r['winner_followers']:,.0f} followers)")
        print(f"  Loser: {r['loser_author']} ({r['loser_followers']:,.0f} followers)")
        print(f"  Follower ratio: {r['follower_ratio']:.1f}x")
        print(f"  Winner post ({r['winner_len']} chars): {r['winner_text']}...")

    # Show examples where loser has more followers (surprising!)
    print("\n" + "=" * 80)
    print("EXAMPLES: LOSER HAS MORE FOLLOWERS (SURPRISING)")
    print("=" * 80)

    surprising = [r for r in results if r['loser_followers'] > r['winner_followers']]
    for r in sorted(surprising, key=lambda x: x['follower_ratio'])[:5]:
        print(f"\nSample {r['idx']}: Engagement ratio {r['eng_ratio']:.1f}x")
        print(f"  Winner: {r['winner_author']} ({r['winner_followers']:,.0f} followers)")
        print(f"  Loser: {r['loser_author']} ({r['loser_followers']:,.0f} followers)")
        print(f"  Winner got more engagement despite fewer followers!")
        print(f"  Winner post ({r['winner_len']} chars): {r['winner_text']}...")
        print(f"  Loser post ({r['loser_len']} chars): {r['loser_text']}...")

    # Analyze all test pairs for comparison
    print("\n" + "=" * 80)
    print("COMPARISON: ALL TEST PAIRS VS HIGH-RATIO WRONG")
    print("=" * 80)

    all_follower_ratios = []
    all_len_ratios = []

    for pair in test_pairs:
        eng_a, eng_b = get_engagement(pair)
        author_a, author_b = get_author(pair)
        text_a, text_b = get_text(pair)

        followers_a = followers_data.get(author_a, 0)
        followers_b = followers_data.get(author_b, 0)

        label = pair.get('label', 0)

        if label == 0:
            winner_f, loser_f = followers_a, followers_b
            winner_l, loser_l = len(text_a), len(text_b)
        else:
            winner_f, loser_f = followers_b, followers_a
            winner_l, loser_l = len(text_b), len(text_a)

        all_follower_ratios.append((winner_f + 1) / (loser_f + 1))
        all_len_ratios.append(winner_l / (loser_l + 1))

    print(f"\nAll test pairs (n={len(test_pairs)}):")
    print(f"  Mean follower ratio (winner/loser): {np.mean(all_follower_ratios):.2f}x")
    print(f"  Median follower ratio: {np.median(all_follower_ratios):.2f}x")
    print(f"  Winner has more followers: {sum(1 for r in all_follower_ratios if r > 1)}/{len(all_follower_ratios)} ({sum(1 for r in all_follower_ratios if r > 1)/len(all_follower_ratios)*100:.1f}%)")

    print(f"\nHigh-ratio wrong (n={len(results)}):")
    print(f"  Mean follower ratio: {np.mean(follower_ratios):.2f}x")
    print(f"  Median follower ratio: {np.median(follower_ratios):.2f}x")
    print(f"  Winner has more followers: {winner_more_followers}/{len(results)} ({winner_more_followers/len(results)*100:.1f}%)")


if __name__ == "__main__":
    run()
