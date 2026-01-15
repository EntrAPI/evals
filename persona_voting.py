#!/usr/bin/env python3
"""
Persona-based engagement prediction.

Approach (inspired by "AI Chatbots Mimic Human Collective Behaviour"):
1. Give the LLM a persona
2. Show it a post and ask "Would you like this?"
3. Do this multiple times (e.g., 4) for each post
4. The post with more "likes" wins
"""

import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from linkedin_data import get_pairs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


class PersonaVoter:
    """LLM with a persona that votes on posts."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        print(f"\nLoading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(DEVICE)
        self.model.eval()
        print("  Ready.")

    def would_like(self, post: str, persona: str, temperature: float = 0.7) -> bool:
        """Ask if persona would like the post. Returns True/False."""

        prompt = f"""You are {persona}

You're scrolling through LinkedIn and see this post:

"{post}"

Would you react to (like) this post? Answer only YES or NO."""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        last_part = response[-30:]

        # Parse YES/NO
        if "yes" in last_part and "no" not in last_part:
            return True
        elif "no" in last_part:
            return False
        else:
            # Ambiguous - treat as no
            return False

    def vote_on_pair(self, post_a: str, post_b: str, persona: str,
                     n_votes: int = 4, temperature: float = 0.7) -> tuple:
        """
        Have persona vote on both posts multiple times.
        Returns (votes_a, votes_b, prediction)
        prediction: 0 if A wins, 1 if B wins
        """
        votes_a = sum(self.would_like(post_a, persona, temperature) for _ in range(n_votes))
        votes_b = sum(self.would_like(post_b, persona, temperature) for _ in range(n_votes))

        if votes_a > votes_b:
            pred = 0  # A wins
        elif votes_b > votes_a:
            pred = 1  # B wins
        else:
            pred = random.choice([0, 1])  # Tie-break randomly

        return votes_a, votes_b, pred

    def cleanup(self):
        del self.model
        del self.tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()


# Different personas to try
PERSONAS = [
    "a typical social media user who enjoys engaging content",
    "a millennial professional who uses Twitter during work breaks",
    "someone who appreciates clever, witty posts",
    "a person who values authenticity and genuine expression",
    "an average internet user scrolling through their feed",
]


def get_pair_texts(pair):
    """Get text from pair (works for both Twitter and LinkedIn)."""
    if hasattr(pair, 'tweet_a'):
        return pair.tweet_a.text, pair.tweet_b.text
    else:
        return pair.post_a.text, pair.post_b.text


def evaluate_persona(voter, pairs, persona, n_votes=4, temperature=0.7):
    """Evaluate accuracy with a specific persona."""
    correct = 0
    total = 0

    for i, pair in enumerate(pairs):
        text_a, text_b = get_pair_texts(pair)
        votes_a, votes_b, pred = voter.vote_on_pair(
            text_a[:200],
            text_b[:200],
            persona,
            n_votes=n_votes,
            temperature=temperature
        )

        if pred == pair.label:
            correct += 1
        total += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}] {correct}/{total} = {correct/total*100:.1f}%")

    return correct / total


def run():
    print("=" * 70)
    print("Persona-Based Voting - LinkedIn (3B Model)")
    print("=" * 70)

    train_pairs, test_pairs = get_pairs()
    print(f"\nTest set: {len(test_pairs)} pairs")

    voter = PersonaVoter()

    results = {}

    # Test key personas with 4 votes
    print("\n" + "=" * 70)
    print("Testing Personas (4 votes each, temp=0.7)")
    print("=" * 70)

    test_personas = [
        "a professional who uses LinkedIn daily for networking",
        "a mid-level manager interested in business insights and leadership content",
        "a tech industry professional who engages with thought leadership posts",
    ]

    for persona in test_personas:
        print(f"\n--- Persona: '{persona[:50]}...' ---")
        acc = evaluate_persona(voter, test_pairs, persona, n_votes=4, temperature=0.7)
        results[persona[:25]] = acc
        print(f"Accuracy: {acc*100:.1f}%")
        torch.cuda.empty_cache()

    # Test with different vote counts using best persona
    print("\n" + "=" * 70)
    print("Testing Vote Counts")
    print("=" * 70)

    best_persona = "a professional who uses LinkedIn daily for networking"

    for n_votes in [2, 6, 10]:
        print(f"\n--- {n_votes} votes per post ---")
        acc = evaluate_persona(voter, test_pairs, best_persona, n_votes=n_votes, temperature=0.7)
        results[f"votes_{n_votes}"] = acc
        print(f"Accuracy: {acc*100:.1f}%")
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:30s}: {acc*100:.1f}%")

    print(f"\nRandom baseline: 50.0%")

    voter.cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    run()
