#!/usr/bin/env python3
"""
Persona-guided tweet virality prediction.

Approach:
1. Extract a "good predictor" persona vector from training data
   - Positive: hidden states when model correctly predicts winner
   - Negative: hidden states when model incorrectly predicts
2. At test time, generate multiple predictions and pick the one
   with highest persona score (dot product with persona vector)

Uses:
- Gemma 3 4B (HuggingFace) for hidden state extraction & scoring
- Gemma 3 4B (Ollama) for generation
"""

import json
import random
import requests
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Paths
DATA_FILE = Path(__file__).parent.parent.parent / "data" / "twitter_clean_no_replies.json"
PERSONA_VECTOR_FILE = Path(__file__).parent / "persona_vector.pt"

# Model configs
# Use Qwen for scoring (fits in VRAM), Gemma for generation (via Ollama)
HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # For hidden states / scoring
OLLAMA_MODEL = "gemma3:4b"  # For generation
OLLAMA_URL = "http://localhost:11434/api/generate"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TweetPair:
    tweet_a: str
    tweet_b: str
    label: int  # 0 = A wins, 1 = B wins
    ratio: float


def load_pairs(n_train: int = 1000, n_test: int = 500, seed: int = 42) -> tuple[list[TweetPair], list[TweetPair]]:
    """Load tweet pairs from JSON."""
    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE) as f:
        data = json.load(f)

    random.seed(seed)

    train_raw = data["train"]
    test_raw = data["test"]

    random.shuffle(train_raw)
    random.shuffle(test_raw)

    train_pairs = [
        TweetPair(
            tweet_a=p["tweet_a"]["text"],
            tweet_b=p["tweet_b"]["text"],
            label=p["label"],
            ratio=p["ratio"],
        )
        for p in train_raw[:n_train]
    ]

    test_pairs = [
        TweetPair(
            tweet_a=p["tweet_a"]["text"],
            tweet_b=p["tweet_b"]["text"],
            label=p["label"],
            ratio=p["ratio"],
        )
        for p in test_raw[:n_test]
    ]

    print(f"  Train: {len(train_pairs)}, Test: {len(test_pairs)}")
    return train_pairs, test_pairs


def make_prompt(tweet_a: str, tweet_b: str) -> str:
    """Create the prediction prompt."""
    return f"""You are predicting Twitter engagement. Both tweets are from the SAME user. Which tweet got more engagement (likes + retweets)?

Tweet A:
{tweet_a[:300]}

Tweet B:
{tweet_b[:300]}

Reply with ONLY "A" or "B" - nothing else."""


class OllamaGenerator:
    """Generate predictions using Ollama."""

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a single response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 10,
            }
        }

        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except Exception as e:
            print(f"  Ollama error: {e}")
            return ""

    def generate_multiple(self, prompt: str, n: int = 4, temperature: float = 0.7) -> list[str]:
        """Generate multiple responses."""
        return [self.generate(prompt, temperature) for _ in range(n)]


class HiddenStateScorer:
    """Extract hidden states and score using persona vector."""

    def __init__(self, model_name: str = HF_MODEL, target_layer: int = None, use_4bit: bool = True):
        print(f"\nLoading {model_name} for scoring...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if use_4bit:
            print("  Using 4-bit quantization...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self.model.eval()

        # Handle Gemma 3 multi-modal config
        config = self.model.config
        if hasattr(config, 'text_config'):
            n_layers = config.text_config.num_hidden_layers
        else:
            n_layers = config.num_hidden_layers

        self.target_layer = target_layer if target_layer else n_layers // 2
        print(f"  Layers: {n_layers}, Using layer: {self.target_layer}")

        self.persona_vector = None

    def get_pooled_hidden_state(self, prompt: str, response: str) -> torch.Tensor:
        """Get mean-pooled hidden state from target layer for the response tokens."""
        # Format as chat
        messages = [{"role": "user", "content": prompt}]

        # Tokenize prompt
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # Tokenize full (prompt + response)
        full_text = prompt_text + response
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Get hidden states
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get target layer hidden states
        hidden = outputs.hidden_states[self.target_layer][0]  # [seq_len, hidden_dim]

        # Pool over response tokens only
        response_start = len(prompt_ids)
        response_hidden = hidden[response_start:]

        if len(response_hidden) == 0:
            return hidden.mean(dim=0)

        return response_hidden.mean(dim=0).float()

    def extract_persona_vector(
        self,
        pairs: list[TweetPair],
        generator: OllamaGenerator,
        n_examples: int = 200,
    ) -> torch.Tensor:
        """
        Extract persona vector from training examples.

        Positive: hidden states when prediction is correct
        Negative: hidden states when prediction is incorrect
        """
        print(f"\nExtracting persona vector from {n_examples} examples...")

        pos_vectors = []
        neg_vectors = []

        for pair in tqdm(pairs[:n_examples], desc="Extracting"):
            prompt = make_prompt(pair.tweet_a, pair.tweet_b)
            response = generator.generate(prompt, temperature=0.0)  # Greedy for extraction

            # Parse prediction
            response_upper = response.upper()
            if "A" in response_upper and "B" not in response_upper:
                pred = 0
            elif "B" in response_upper and "A" not in response_upper:
                pred = 1
            else:
                continue  # Skip ambiguous

            # Get hidden state
            hidden = self.get_pooled_hidden_state(prompt, response)

            # Classify as positive (correct) or negative (incorrect)
            if pred == pair.label:
                pos_vectors.append(hidden)
            else:
                neg_vectors.append(hidden)

        print(f"  Positive (correct): {len(pos_vectors)}")
        print(f"  Negative (incorrect): {len(neg_vectors)}")

        if len(pos_vectors) == 0 or len(neg_vectors) == 0:
            raise ValueError("Need both positive and negative examples")

        # Compute mean difference
        pos_mean = torch.stack(pos_vectors).mean(dim=0)
        neg_mean = torch.stack(neg_vectors).mean(dim=0)

        persona_vec = pos_mean - neg_mean
        persona_vec = persona_vec / persona_vec.norm()  # Normalize

        self.persona_vector = persona_vec
        return persona_vec

    def score(self, prompt: str, response: str) -> float:
        """Score a response using persona vector."""
        if self.persona_vector is None:
            raise ValueError("Persona vector not set")

        hidden = self.get_pooled_hidden_state(prompt, response)

        # Ensure same device
        pv = self.persona_vector.to(hidden.device)

        return float((hidden * pv).sum().item())

    def save_persona_vector(self, path: Path = PERSONA_VECTOR_FILE):
        """Save persona vector to disk."""
        torch.save({
            "vector": self.persona_vector.cpu(),
            "layer": self.target_layer,
        }, path)
        print(f"Saved persona vector to {path}")

    def load_persona_vector(self, path: Path = PERSONA_VECTOR_FILE):
        """Load persona vector from disk."""
        data = torch.load(path, weights_only=True)
        self.persona_vector = data["vector"]
        self.target_layer = data["layer"]
        print(f"Loaded persona vector from {path}")


def parse_prediction(response: str) -> int | None:
    """Parse A/B prediction from response."""
    response_upper = response.upper()
    if "A" in response_upper and "B" not in response_upper:
        return 0
    elif "B" in response_upper and "A" not in response_upper:
        return 1
    return None


def evaluate_baseline(pairs: list[TweetPair], generator: OllamaGenerator) -> float:
    """Evaluate baseline (single greedy prediction)."""
    print("\n--- Baseline (single greedy prediction) ---")
    correct = 0
    total = 0

    for pair in tqdm(pairs, desc="Baseline"):
        prompt = make_prompt(pair.tweet_a, pair.tweet_b)
        response = generator.generate(prompt, temperature=0.0)
        pred = parse_prediction(response)

        if pred is not None:
            if pred == pair.label:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"Baseline accuracy: {acc*100:.1f}% ({correct}/{total})")
    return acc


def evaluate_persona_sampling(
    pairs: list[TweetPair],
    generator: OllamaGenerator,
    scorer: HiddenStateScorer,
    n_samples: int = 4,
    temperature: float = 0.7,
) -> float:
    """Evaluate with persona-guided sampling."""
    print(f"\n--- Persona Sampling ({n_samples} samples, temp={temperature}) ---")
    correct = 0
    total = 0

    for pair in tqdm(pairs, desc="Persona sampling"):
        prompt = make_prompt(pair.tweet_a, pair.tweet_b)

        # Generate multiple responses
        responses = generator.generate_multiple(prompt, n=n_samples, temperature=temperature)

        # Score each and pick best
        best_score = float("-inf")
        best_pred = None

        for response in responses:
            pred = parse_prediction(response)
            if pred is None:
                continue

            score = scorer.score(prompt, response)
            if score > best_score:
                best_score = score
                best_pred = pred

        if best_pred is not None:
            if best_pred == pair.label:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"Persona sampling accuracy: {acc*100:.1f}% ({correct}/{total})")
    return acc


def evaluate_majority_voting(
    pairs: list[TweetPair],
    generator: OllamaGenerator,
    n_samples: int = 4,
    temperature: float = 0.7,
) -> float:
    """Evaluate with majority voting (no persona, just vote count)."""
    print(f"\n--- Majority Voting ({n_samples} samples, temp={temperature}) ---")
    correct = 0
    total = 0

    for pair in tqdm(pairs, desc="Majority voting"):
        prompt = make_prompt(pair.tweet_a, pair.tweet_b)

        # Generate multiple responses
        responses = generator.generate_multiple(prompt, n=n_samples, temperature=temperature)

        # Count votes
        votes = {0: 0, 1: 0}
        for response in responses:
            pred = parse_prediction(response)
            if pred is not None:
                votes[pred] += 1

        # Pick majority
        if votes[0] > votes[1]:
            final_pred = 0
        elif votes[1] > votes[0]:
            final_pred = 1
        elif votes[0] == votes[1] and votes[0] > 0:
            final_pred = random.choice([0, 1])
        else:
            continue  # All unparseable

        if final_pred == pair.label:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"Majority voting accuracy: {acc*100:.1f}% ({correct}/{total})")
    return acc


def run():
    print("=" * 70)
    print("Persona-Guided Tweet Virality Prediction")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load data
    train_pairs, test_pairs = load_pairs(n_train=500, n_test=200)

    # Initialize components
    generator = OllamaGenerator()
    scorer = HiddenStateScorer()

    # Check if persona vector exists
    if PERSONA_VECTOR_FILE.exists():
        print(f"\nFound existing persona vector at {PERSONA_VECTOR_FILE}")
        scorer.load_persona_vector()
    else:
        # Extract persona vector from training data
        scorer.extract_persona_vector(train_pairs, generator, n_examples=300)
        scorer.save_persona_vector()

    # Evaluate
    results = {}

    # 1. Baseline
    results["baseline"] = evaluate_baseline(test_pairs, generator)

    # 2. Majority voting (for comparison)
    results["majority_4"] = evaluate_majority_voting(test_pairs, generator, n_samples=4, temperature=0.7)

    # 3. Persona sampling
    results["persona_4"] = evaluate_persona_sampling(test_pairs, generator, scorer, n_samples=4, temperature=0.7)

    # 4. More samples
    results["majority_8"] = evaluate_majority_voting(test_pairs, generator, n_samples=8, temperature=0.7)
    results["persona_8"] = evaluate_persona_sampling(test_pairs, generator, scorer, n_samples=8, temperature=0.7)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}: {acc*100:.1f}%")

    print(f"\nRandom baseline: 50.0%")

    # Save results
    results_file = Path(__file__).parent / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    run()
