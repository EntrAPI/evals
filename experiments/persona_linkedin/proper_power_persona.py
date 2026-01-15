#!/usr/bin/env python3
"""
Proper Power Persona Sampling for LinkedIn engagement prediction.

This implements the ACTUAL algorithm from power_persona_sampling:
1. Single model for generation AND persona scoring (Qwen 2.5 3B)
2. Persona vector = mean(hidden | correct) - mean(hidden | incorrect)
3. MCMC sampling with suffix resampling proposals
4. Stochastic acceptance based on: (α-1)*Δlogp + β*Δpersona

For classification, we generate chain-of-thought reasoning, then extract answer.
This allows meaningful suffix resampling (cut reasoning, resample rest).
"""

import json
import math
import random
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "linkedin_pairs.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Trajectory:
    prompt_ids: list[int]
    response_ids: list[int]


@dataclass
class ScoredTrajectory:
    traj: Trajectory
    logp_response: float
    persona_score: float
    prediction: int  # 0=A, 1=B, -1=unparseable


class PersonaSampler:
    """
    Proper MH sampler with persona scoring.
    """
    def __init__(
        self,
        model,
        tokenizer,
        persona_vector: torch.Tensor,
        persona_layer: int = -8,
        alpha: float = 1.0,  # logp weight (α-1 in acceptance ratio)
        beta: float = 1.0,   # persona weight
        temperature: float = 0.7,
        max_new_tokens: int = 150,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.persona_vector = persona_vector.to(model.device)
        self.persona_layer = persona_layer
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def sample(
        self,
        prompt: str,
        num_steps: int = 20,
        burn_in: int = 5,
        seed: Optional[int] = None,
    ) -> ScoredTrajectory:
        """Run MH sampling and return final trajectory."""
        if seed is not None:
            random.seed(seed)

        # Encode prompt
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # Initialize with a sample
        init_response_ids = self._sample_response(prompt_ids)
        current = self._score(Trajectory(prompt_ids, init_response_ids))

        # MH chain
        accepted = 0
        for step in range(num_steps):
            # Propose by suffix resampling
            proposed_traj, cut = self._propose(current.traj)
            proposed = self._score(proposed_traj)

            # Compute log acceptance ratio
            # For suffix resampling: need logp of suffix only
            logp_cur_suffix = self._logp_suffix(current.traj, cut)
            logp_prop_suffix = self._logp_suffix(proposed.traj, cut)

            delta_logp = logp_prop_suffix - logp_cur_suffix
            delta_persona = proposed.persona_score - current.persona_score

            log_r = (self.alpha - 1.0) * delta_logp + self.beta * delta_persona

            # Accept/reject
            if log_r >= 0 or math.log(random.random()) < log_r:
                current = proposed
                accepted += 1

        return current

    def _sample_response(self, prompt_ids: list[int]) -> list[int]:
        """Sample a response from the model."""
        input_ids = torch.tensor([prompt_ids], device=self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = outputs[0, len(prompt_ids):].tolist()
        return self._ensure_ended(response_ids)

    def _ensure_ended(self, response_ids: list[int]) -> list[int]:
        """Ensure response ends with EOS."""
        eos_id = self.tokenizer.eos_token_id
        if not response_ids or response_ids[-1] != eos_id:
            response_ids = response_ids + [eos_id]
        return response_ids[:self.max_new_tokens]

    def _propose(self, traj: Trajectory) -> tuple[Trajectory, int]:
        """Propose new trajectory by suffix resampling."""
        resp = traj.response_ids
        if len(resp) <= 1:
            cut = 0
        else:
            cut = random.randrange(0, len(resp))

        # Keep prefix, resample suffix
        prefix = resp[:cut]
        full_prefix = traj.prompt_ids + prefix

        input_ids = torch.tensor([full_prefix], device=self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max(1, self.max_new_tokens - len(prefix)),
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_suffix = outputs[0, len(full_prefix):].tolist()
        new_response = prefix + new_suffix
        new_response = self._ensure_ended(new_response)

        return Trajectory(traj.prompt_ids, new_response), cut

    def _score(self, traj: Trajectory) -> ScoredTrajectory:
        """Score trajectory: compute logp, persona score, and prediction."""
        full_ids = traj.prompt_ids + traj.response_ids
        input_ids = torch.tensor([full_ids], device=self.model.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        # Log probability of response
        logits = outputs.logits[0, len(traj.prompt_ids)-1:-1, :]  # shifted
        targets = torch.tensor(traj.response_ids, device=self.model.device)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        logp_response = token_log_probs.sum().item()

        # Persona score: mean pool response hidden states, dot with persona vector
        hidden = outputs.hidden_states[self.persona_layer][0]  # [seq_len, hidden_dim]
        response_hidden = hidden[len(traj.prompt_ids):]  # response portion
        pooled = response_hidden.mean(dim=0)  # mean pool
        persona_score = (pooled * self.persona_vector).sum().item()

        # Extract prediction from response
        response_text = self.tokenizer.decode(traj.response_ids, skip_special_tokens=True)
        prediction = self._parse_prediction(response_text)

        return ScoredTrajectory(
            traj=traj,
            logp_response=logp_response,
            persona_score=persona_score,
            prediction=prediction,
        )

    def _logp_suffix(self, traj: Trajectory, cut: int) -> float:
        """Compute log probability of suffix (response[cut:])."""
        if cut >= len(traj.response_ids):
            return 0.0

        full_ids = traj.prompt_ids + traj.response_ids
        input_ids = torch.tensor([full_ids], device=self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, return_dict=True)

        # Log prob of tokens from cut onwards
        start_idx = len(traj.prompt_ids) + cut - 1  # -1 for shifted logits
        logits = outputs.logits[0, start_idx:-1, :]
        targets = torch.tensor(traj.response_ids[cut:], device=self.model.device)

        if len(targets) == 0:
            return 0.0

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return token_log_probs.sum().item()

    def _parse_prediction(self, text: str) -> int:
        """Extract A/B prediction from response."""
        return parse_prediction(text)


def parse_prediction(text: str) -> int:
    """Extract A/B prediction from response text."""
    text_upper = text.upper()

    # Look for explicit "Answer: A/B" pattern
    if "ANSWER: A" in text_upper or "ANSWER:A" in text_upper:
        return 0
    if "ANSWER: B" in text_upper or "ANSWER:B" in text_upper:
        return 1

    # Look for "POST A" or "POST B" as final answer
    if "POST A" in text_upper and "POST B" not in text_upper[-50:]:
        return 0
    if "POST B" in text_upper and "POST A" not in text_upper[-50:]:
        return 1

    # Look at last 30 chars for standalone A or B
    last_part = text_upper[-30:] if len(text_upper) > 30 else text_upper

    # Count A's and B's in last part (excluding common words)
    # Remove common words that contain A or B
    clean_last = last_part
    for word in ["AND", "ABOUT", "BECAUSE", "BASED", "ABLE", "ABOVE"]:
        clean_last = clean_last.replace(word, "")

    a_count = clean_last.count("A")
    b_count = clean_last.count("B")

    if a_count > 0 and b_count == 0:
        return 0
    if b_count > 0 and a_count == 0:
        return 1

    # Look for patterns like "A." or "B." or "(A)" or "(B)"
    import re
    if re.search(r'\bA[.\):\s]', last_part):
        return 0
    if re.search(r'\bB[.\):\s]', last_part):
        return 1

    return -1


def load_model(model_name="Qwen/Qwen2.5-3B-Instruct"):
    """Load model for persona sampling."""
    print(f"Loading model: {model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model device: {next(model.parameters()).device}")
    return model, tokenizer


def build_author_stats(train_pairs):
    """Build author statistics from training data."""
    author_posts = defaultdict(list)
    for pair in train_pairs:
        for key in ['post_a', 'post_b']:
            author = pair[key].get('author', 'Unknown')
            eng = pair[key]['reactions'] + pair[key].get('comments', 0)
            author_posts[author].append(eng)
    return {a: {'avg_eng': np.mean(engs)} for a, engs in author_posts.items()}


def format_prompt(pair, author_stats, cot=True):
    """Format prompt for classification."""
    author_a = pair['post_a'].get('author', 'Unknown')
    author_b = pair['post_b'].get('author', 'Unknown')
    post_a = pair['post_a']['text'][:300]
    post_b = pair['post_b']['text'][:300]

    stats_a = author_stats.get(author_a, {})
    stats_b = author_stats.get(author_b, {})
    avg_a = stats_a.get('avg_eng', 50)
    avg_b = stats_b.get('avg_eng', 50)

    if cot:
        return f"""Which LinkedIn post got more engagement?

Post A by {author_a} (avg engagement: {avg_a:.0f}):
{post_a}

Post B by {author_b} (avg engagement: {avg_b:.0f}):
{post_b}

Brief reasoning, then answer with "Answer: A" or "Answer: B"."""
    else:
        # Simple prompt for direct A/B answer
        return f"""Which LinkedIn post got more engagement?

Post A by {author_a} (avg engagement: {avg_a:.0f}):
{post_a}

Post B by {author_b} (avg engagement: {avg_b:.0f}):
{post_b}

Reply with only A or B."""


def extract_persona_vector(
    model, tokenizer, train_pairs, author_stats, n_samples=200, layer_idx=-8
):
    """
    Extract persona vector from training data.
    persona_vector = mean(hidden | correct) - mean(hidden | incorrect)
    """
    print(f"\nExtracting persona vector from {n_samples} training samples...")

    # Sample training pairs
    indices = np.random.choice(len(train_pairs), min(n_samples, len(train_pairs)), replace=False)

    correct_hiddens = []
    incorrect_hiddens = []
    parse_failures = 0

    for i, idx in enumerate(tqdm(indices, desc="Extracting persona")):
        pair = train_pairs[idx]

        # Ground truth
        eng_a = pair['post_a']['reactions'] + pair['post_a'].get('comments', 0)
        eng_b = pair['post_b']['reactions'] + pair['post_b'].get('comments', 0)
        truth = 0 if eng_a > eng_b else 1

        # Get model response (use CoT for persona extraction)
        prompt = format_prompt(pair, author_stats, cot=True)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # Enough for brief reasoning + answer
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            response_ids = outputs[0, inputs['input_ids'].shape[1]:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            # Debug first few
            if i < 3:
                print(f"  Sample {i}: '{response_text[:60]}...'")

            # Get hidden states for the response
            full_outputs = model(outputs, output_hidden_states=True, return_dict=True)
            hidden = full_outputs.hidden_states[layer_idx][0]  # [seq_len, hidden_dim]
            response_hidden = hidden[inputs['input_ids'].shape[1]:]  # response portion
            pooled = response_hidden.mean(dim=0).cpu()  # mean pool

            # Parse prediction
            pred = parse_prediction(response_text)

            if pred >= 0:
                if pred == truth:
                    correct_hiddens.append(pooled)
                else:
                    incorrect_hiddens.append(pooled)
            else:
                parse_failures += 1
                if parse_failures > 10 and i < 20:
                    print(f"\nWARNING: High parse failure rate ({parse_failures}/{i+1})")
                    print(f"Last response: {response_text}")

    print(f"Correct predictions: {len(correct_hiddens)}")
    print(f"Incorrect predictions: {len(incorrect_hiddens)}")
    print(f"Parse failures: {parse_failures}")

    if len(correct_hiddens) == 0 or len(incorrect_hiddens) == 0:
        raise ValueError("Need both correct and incorrect predictions for persona vector")

    correct_mean = torch.stack(correct_hiddens).mean(dim=0)
    incorrect_mean = torch.stack(incorrect_hiddens).mean(dim=0)

    persona_vector = correct_mean - incorrect_mean
    persona_vector = persona_vector / (persona_vector.norm() + 1e-8)

    # Verify separation
    correct_scores = [(h * persona_vector).sum().item() for h in correct_hiddens]
    incorrect_scores = [(h * persona_vector).sum().item() for h in incorrect_hiddens]
    print(f"Mean persona score (correct): {np.mean(correct_scores):.4f}")
    print(f"Mean persona score (incorrect): {np.mean(incorrect_scores):.4f}")
    print(f"Separation: {np.mean(correct_scores) - np.mean(incorrect_scores):.4f}")

    return persona_vector


def run():
    print("=" * 70)
    print("PROPER POWER PERSONA SAMPLING")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    with open(DATA_FILE) as f:
        data = json.load(f)
    train_pairs = data['train']
    test_pairs = data['test']
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Build author stats
    author_stats = build_author_stats(train_pairs)

    # Load model
    model, tokenizer = load_model()

    # Extract persona vector (use more samples for better vector)
    persona_vector = extract_persona_vector(
        model, tokenizer, train_pairs, author_stats, n_samples=200
    )

    # ========================================
    # Baseline: Direct sampling (no persona)
    # ========================================
    print("\n" + "=" * 70)
    print("BASELINE: Direct sampling (no MH)")
    print("=" * 70)

    # Use full test set
    test_subset = test_pairs

    baseline_correct = 0
    baseline_preds = []
    baseline_parse_failures = 0

    for i, pair in enumerate(tqdm(test_subset, desc="Baseline")):
        eng_a = pair['post_a']['reactions'] + pair['post_a'].get('comments', 0)
        eng_b = pair['post_b']['reactions'] + pair['post_b'].get('comments', 0)
        truth = 0 if eng_a > eng_b else 1

        # Use simple prompt for baseline (no CoT)
        prompt = format_prompt(pair, author_stats, cot=False)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # Short for simple A/B answer
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = parse_prediction(response)

        # Debug: print first few responses to verify parsing
        if i < 3:
            print(f"  Sample {i}: '{response[:50]}...' -> pred={pred}, truth={truth}")

        # Check for parsing failures
        if pred == -1:
            baseline_parse_failures += 1
            if baseline_parse_failures > 5 and i < 10:
                print(f"\nERROR: Too many parsing failures ({baseline_parse_failures}/{i+1})")
                print(f"Last response: {response}")
                raise ValueError("Parsing is broken - fix before continuing")

        baseline_preds.append(pred)
        if pred == truth:
            baseline_correct += 1

    valid_baseline = sum(1 for p in baseline_preds if p >= 0)
    baseline_acc = baseline_correct / len(test_subset)
    print(f"Baseline accuracy: {baseline_acc*100:.1f}% ({baseline_correct}/{len(test_subset)})")
    print(f"Parse failures: {baseline_parse_failures}/{len(test_subset)}")

    # ========================================
    # Power Persona Sampling
    # ========================================
    print("\n" + "=" * 70)
    print("POWER PERSONA SAMPLING (MH)")
    print("=" * 70)

    # Test different beta values
    for beta in [1.0]:  # Start with just one value for speed
        print(f"\n--- Beta = {beta} ---")

        sampler = PersonaSampler(
            model=model,
            tokenizer=tokenizer,
            persona_vector=persona_vector,
            persona_layer=-8,
            alpha=1.0,
            beta=beta,
            temperature=0.7,
            max_new_tokens=150,
        )

        persona_correct = 0
        persona_preds = []

        for i, pair in enumerate(tqdm(test_subset, desc=f"Persona β={beta}")):
            eng_a = pair['post_a']['reactions'] + pair['post_a'].get('comments', 0)
            eng_b = pair['post_b']['reactions'] + pair['post_b'].get('comments', 0)
            truth = 0 if eng_a > eng_b else 1

            prompt = format_prompt(pair, author_stats)

            # Run MH sampling (reduced steps for speed)
            result = sampler.sample(prompt, num_steps=10, burn_in=3, seed=42+i)

            persona_preds.append(result.prediction)
            if result.prediction == truth:
                persona_correct += 1

        persona_acc = persona_correct / len(test_subset)
        print(f"Persona sampling accuracy: {persona_acc*100:.1f}%")

        # Compare predictions
        flipped_correct = sum(1 for i in range(len(test_subset))
                            if baseline_preds[i] != persona_preds[i]
                            and persona_preds[i] == (0 if test_subset[i]['post_a']['reactions'] > test_subset[i]['post_b']['reactions'] else 1))
        flipped_wrong = sum(1 for i in range(len(test_subset))
                          if baseline_preds[i] != persona_preds[i]
                          and baseline_preds[i] == (0 if test_subset[i]['post_a']['reactions'] > test_subset[i]['post_b']['reactions'] else 1))
        print(f"Flipped to correct: {flipped_correct}, Flipped to wrong: {flipped_wrong}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline (greedy): {baseline_acc*100:.1f}%")
    print("See above for persona sampling results at different β values")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    run()
