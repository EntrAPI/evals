#!/usr/bin/env python3
"""
Steering vector experiment - steer during generation.

Proper persona vector approach:
1. Compute steering vector from correct vs incorrect predictions
2. Add steering vector during generation
3. See if model's direct A/B predictions improve from ~50%
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

from tweet_data import get_pairs, TweetPair

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class SteeredPredictor:
    """Model that predicts A vs B with optional steering during generation."""

    def __init__(self, model_id: str = MODEL_ID):
        print(f"\nLoading {model_id}...")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(DEVICE)
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        print(f"  Layers: {self.n_layers}, Hidden: {self.hidden_size}")

        self.hooks = []
        self.captured = {}

        # Steering config
        self.steering_vector = None
        self.steering_layers = []
        self.steering_strength = 0.0

    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        raise ValueError("Unknown model architecture")

    def set_steering(self, vector: torch.Tensor, layers: list, strength: float):
        """Configure steering for generation."""
        self.steering_vector = vector
        self.steering_layers = layers
        self.steering_strength = strength
        self._register_steering_hooks()

    def clear_steering(self):
        """Remove steering."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.steering_vector = None
        self.steering_layers = []
        self.steering_strength = 0.0

    def _register_steering_hooks(self):
        """Register hooks that modify activations during forward pass."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

        if self.steering_vector is None:
            return

        layers = self._get_layers()

        for layer_idx in self.steering_layers:
            layer = layers[layer_idx]

            def make_hook(strength, vector):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        steered = hidden + strength * vector.to(hidden.device).to(hidden.dtype)
                        return (steered,) + output[1:]
                    else:
                        return output + strength * vector.to(output.device).to(output.dtype)
                return hook_fn

            handle = layer.register_forward_hook(
                make_hook(self.steering_strength, self.steering_vector)
            )
            self.hooks.append(handle)

    def _register_capture_hook(self, layer_idx: int):
        """Register hook to capture activations at a layer."""
        # Clear existing capture hooks
        for h in self.hooks:
            h.remove()
        self.hooks = []

        layers = self._get_layers()
        layer = layers[layer_idx]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.captured['hidden'] = hidden.detach()

        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def predict(self, tweet_a: str, tweet_b: str, with_reasoning: bool = True):
        """
        Predict which tweet gets more engagement.
        Returns: (prediction, response_text)
        prediction: 0 = A, 1 = B, -1 = unparseable
        """
        if with_reasoning:
            prompt = f"""You are an expert at predicting social media engagement.

Given two tweets, predict which one will get MORE engagement (likes, retweets, replies).

Tweet A: {tweet_a}

Tweet B: {tweet_b}

Think step by step about what makes content engaging, then give your final answer.
Your response MUST end with either "ANSWER: A" or "ANSWER: B" on its own line."""
        else:
            prompt = f"""Which tweet will get more engagement?

Tweet A: {tweet_a}

Tweet B: {tweet_b}

Reply with only "A" or "B"."""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200 if with_reasoning else 10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False,  # Disable KV cache so hooks apply to every token
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse prediction
        response_lower = response.lower()
        if with_reasoning:
            if "answer: a" in response_lower:
                pred = 0
            elif "answer: b" in response_lower:
                pred = 1
            else:
                pred = -1
        else:
            # Just look for A or B
            response_end = response_lower[-20:]
            if "a" in response_end and "b" not in response_end:
                pred = 0
            elif "b" in response_end and "a" not in response_end:
                pred = 1
            else:
                pred = -1

        return pred, response

    def predict_with_activations(self, tweet_a: str, tweet_b: str, layer_idx: int):
        """Predict and capture activations at specified layer."""
        self._register_capture_hook(layer_idx)

        prompt = f"""You are an expert at predicting social media engagement.

Given two tweets, predict which one will get MORE engagement (likes, retweets, replies).

Tweet A: {tweet_a}

Tweet B: {tweet_b}

Think step by step about what makes content engaging, then give your final answer.
Your response MUST end with either "ANSWER: A" or "ANSWER: B" on its own line."""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Run forward pass on full output to get activations
        with torch.no_grad():
            self.model(outputs)
            activations = self.captured['hidden'].float()

        # Parse prediction
        response_lower = response.lower()
        if "answer: a" in response_lower:
            pred = 0
        elif "answer: b" in response_lower:
            pred = 1
        else:
            pred = -1

        # Clear hooks
        for h in self.hooks:
            h.remove()
        self.hooks = []

        return pred, response, activations

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        del self.model
        del self.tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def collect_steering_data(predictor, pairs, layer_idx, n_correct=20, n_incorrect=20):
    """Collect activations for correct and incorrect predictions."""
    correct_acts = []
    incorrect_acts = []

    random.shuffle(pairs)

    for i, pair in enumerate(pairs):
        if len(correct_acts) >= n_correct and len(incorrect_acts) >= n_incorrect:
            break

        text_a = pair.tweet_a.text
        text_b = pair.tweet_b.text
        true_label = pair.label

        pred, response, activations = predictor.predict_with_activations(
            text_a, text_b, layer_idx
        )

        if pred == -1:
            print(f"  [{i}] Unparseable, skipping")
            continue

        is_correct = (pred == true_label)

        # Mean of last 10 tokens (answer region)
        act_mean = activations[0, -10:, :].mean(dim=0)

        if is_correct and len(correct_acts) < n_correct:
            correct_acts.append(act_mean)
            print(f"  [{i}] CORRECT ({len(correct_acts)}/{n_correct})")
        elif not is_correct and len(incorrect_acts) < n_incorrect:
            incorrect_acts.append(act_mean)
            print(f"  [{i}] INCORRECT ({len(incorrect_acts)}/{n_incorrect})")

        torch.cuda.empty_cache()

    return correct_acts, incorrect_acts


def compute_steering_vector(correct_acts, incorrect_acts):
    """Steering vector = mean(correct) - mean(incorrect)"""
    correct_mean = torch.stack(correct_acts).mean(dim=0)
    incorrect_mean = torch.stack(incorrect_acts).mean(dim=0)
    return correct_mean - incorrect_mean


def evaluate_predictions(predictor, pairs, desc=""):
    """Evaluate model's direct prediction accuracy."""
    correct = 0
    total = 0
    unparseable = 0

    for i, pair in enumerate(pairs):
        pred, _ = predictor.predict(pair.tweet_a.text, pair.tweet_b.text)

        if pred == -1:
            unparseable += 1
            continue

        if pred == pair.label:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pairs)}] {correct}/{total} = {correct/total*100:.1f}%")

    acc = correct / total if total > 0 else 0
    print(f"{desc}: {correct}/{total} = {acc*100:.1f}% (unparseable: {unparseable})")
    return acc


def run():
    print("=" * 70)
    print("Steering During Generation")
    print("=" * 70)

    train_pairs, test_pairs = get_pairs()
    print(f"\nTrain: {len(train_pairs)} pairs")
    print(f"Test: {len(test_pairs)} pairs")

    predictor = SteeredPredictor(MODEL_ID)

    # Step 1: Baseline accuracy (no steering)
    print("\n" + "=" * 70)
    print("STEP 1: Baseline (no steering)")
    print("=" * 70)

    baseline_acc = evaluate_predictions(predictor, test_pairs, "Baseline")

    # Step 2: Collect steering data from training set
    print("\n" + "=" * 70)
    print("STEP 2: Collecting steering data")
    print("=" * 70)

    steering_layer = predictor.n_layers // 2  # Layer 12
    print(f"Using layer {steering_layer} for steering vector")

    correct_acts, incorrect_acts = collect_steering_data(
        predictor, train_pairs, steering_layer, n_correct=20, n_incorrect=20
    )

    print(f"\nCollected: {len(correct_acts)} correct, {len(incorrect_acts)} incorrect")

    if len(correct_acts) < 5 or len(incorrect_acts) < 5:
        print("Not enough data. Exiting.")
        return

    # Step 3: Compute steering vector
    print("\n" + "=" * 70)
    print("STEP 3: Computing steering vector")
    print("=" * 70)

    steering_vector = compute_steering_vector(correct_acts, incorrect_acts)
    print(f"Steering vector norm: {steering_vector.norm().item():.4f}")

    # Step 4: Test with steering during generation
    print("\n" + "=" * 70)
    print("STEP 4: Testing steering during generation")
    print("=" * 70)

    results = {"baseline": baseline_acc}

    # Test configurations - need MUCH higher strengths!
    # Steering vector norm ~2, so per-component ~0.07
    # Hidden states have std ~4.6, need shift of similar magnitude
    # strength = 4.6 / 0.07 ≈ 65 for 1 std, need more for argmax flip
    configs = [
        # (layers, strength, name)
        ([12], 100.0, "L12_s100"),
        ([12], 500.0, "L12_s500"),
        ([12], 1000.0, "L12_s1000"),
        ([12], 2000.0, "L12_s2000"),
        # Multi-layer
        (list(range(8, 16)), 500.0, "L8-15_s500"),
        # Negative (should hurt)
        ([12], -500.0, "L12_s-500"),
    ]

    for layers, strength, name in configs:
        print(f"\n--- {name} ---")
        predictor.set_steering(steering_vector, layers, strength)
        acc = evaluate_predictions(predictor, test_pairs, name)
        results[name] = acc
        predictor.clear_steering()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if acc == max(results.values()) else ""
        baseline_marker = " (baseline)" if name == "baseline" else ""
        print(f"  {name:20s}: {acc*100:.1f}%{marker}{baseline_marker}")

    best = max(results, key=results.get)
    improvement = results[best] - results["baseline"]
    print(f"\nBest: {best} = {results[best]*100:.1f}%")
    print(f"Improvement over baseline: {improvement*100:+.1f}%")

    predictor.cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    run()
