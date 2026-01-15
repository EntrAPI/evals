#!/usr/bin/env python3
"""
Steering vector experiment for tweet engagement prediction.

Approach:
1. Ask model to predict which tweet gets more engagement (with reasoning)
2. Split into correct vs incorrect predictions
3. Compute steering vector = mean(correct) - mean(incorrect)
4. Apply steering during embedding extraction
5. Test if this improves classifier performance
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

from tweet_data import get_pairs, TweetPair

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class SteeringVectorExtractor:
    """Extract steering vectors from correct vs incorrect predictions."""

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

        # For capturing activations
        self.captured = {}
        self.hooks = []

    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        raise ValueError("Unknown model architecture")

    def _register_hook(self, layer_idx):
        """Register hook on a specific layer."""
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
            # Store all token activations
            self.captured['hidden'] = hidden.detach()

        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def predict_with_reasoning(self, tweet_a: str, tweet_b: str, layer_idx: int):
        """
        Ask model to predict which tweet gets more engagement.
        Returns: (prediction, is_correct, activations)
        """
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

        # Register hook for this layer
        self._register_hook(layer_idx)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Get final activations (need to run forward pass on full sequence)
        with torch.no_grad():
            self.model(outputs)
            activations = self.captured['hidden'].float()  # (1, seq_len, hidden)

        # Parse prediction
        response_lower = response.lower()
        if "answer: a" in response_lower:
            pred = 0
        elif "answer: b" in response_lower:
            pred = 1
        else:
            pred = -1  # Unparseable

        return pred, response, activations

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        del self.model
        del self.tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def collect_steering_data(extractor, pairs, layer_idx, n_correct=20, n_incorrect=20):
    """
    Collect activations for correct and incorrect predictions.
    """
    correct_activations = []
    incorrect_activations = []
    correct_pairs = []
    incorrect_pairs = []

    random.shuffle(pairs)

    for i, pair in enumerate(pairs):
        if len(correct_activations) >= n_correct and len(incorrect_activations) >= n_incorrect:
            break

        text_a = pair.tweet_a.text
        text_b = pair.tweet_b.text
        true_label = pair.label  # 0 = A wins, 1 = B wins

        pred, response, activations = extractor.predict_with_reasoning(text_a, text_b, layer_idx)

        if pred == -1:
            print(f"  [{i}] Unparseable response, skipping")
            continue

        is_correct = (pred == true_label)

        # Take mean activation across all tokens (or last token)
        # Using mean of last 10 tokens to capture "answer" region
        act_mean = activations[0, -10:, :].mean(dim=0)

        if is_correct and len(correct_activations) < n_correct:
            correct_activations.append(act_mean)
            correct_pairs.append((pair, pred, response))
            print(f"  [{i}] CORRECT ({len(correct_activations)}/{n_correct})")
        elif not is_correct and len(incorrect_activations) < n_incorrect:
            incorrect_activations.append(act_mean)
            incorrect_pairs.append((pair, pred, response))
            print(f"  [{i}] INCORRECT ({len(incorrect_activations)}/{n_incorrect})")
        else:
            print(f"  [{i}] {'correct' if is_correct else 'incorrect'} - already have enough")

        torch.cuda.empty_cache()

    return correct_activations, incorrect_activations, correct_pairs, incorrect_pairs


def compute_steering_vector(correct_acts, incorrect_acts):
    """Compute steering vector as difference of means."""
    correct_mean = torch.stack(correct_acts).mean(dim=0)
    incorrect_mean = torch.stack(incorrect_acts).mean(dim=0)
    steering_vector = correct_mean - incorrect_mean
    return steering_vector


class SteeredEmbeddingExtractor:
    """Extract embeddings with optional steering."""

    def __init__(self, model_id: str, steering_vector: torch.Tensor = None,
                 steering_layers: list = None, steering_strength: float = 0.0):
        print(f"\nLoading {model_id} (steered)...")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(DEVICE)
        self.model.eval()

        self.steering_vector = steering_vector
        self.steering_layers = steering_layers if steering_layers else []
        self.steering_strength = steering_strength

        self.n_layers = self.model.config.num_hidden_layers
        # Use multiple layers for embedding
        self.layer_indices = [
            self.n_layers // 7,
            self.n_layers // 4,
            self.n_layers // 2,
            3 * self.n_layers // 4,
        ]
        print(f"  Embedding layers: {self.layer_indices}")
        if steering_vector is not None and self.steering_layers:
            print(f"  Steering layers: {self.steering_layers}, strength: {steering_strength}")

        self.captured = {}
        self.hooks = []
        self._register_hooks()

    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        raise ValueError("Unknown model architecture")

    def _register_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

        layers = self._get_layers()

        # Embedding extraction hooks
        for layer_idx in self.layer_indices:
            layer = layers[layer_idx]

            def make_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    if hidden.dim() == 3:
                        self.captured[idx] = hidden[0].detach()
                    else:
                        self.captured[idx] = hidden.detach()
                return hook_fn

            handle = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)

        # Steering hooks (modifies activations at multiple layers)
        if self.steering_vector is not None and self.steering_layers:
            for steer_idx in self.steering_layers:
                steer_layer = layers[steer_idx]

                def make_steering_hook(strength, vector):
                    def steering_hook(module, input, output):
                        if isinstance(output, tuple):
                            hidden = output[0]
                            steered = hidden + strength * vector.to(hidden.device).to(hidden.dtype)
                            return (steered,) + output[1:]
                        else:
                            return output + strength * vector.to(output.device).to(output.dtype)
                    return steering_hook

                handle = steer_layer.register_forward_hook(
                    make_steering_hook(self.steering_strength, self.steering_vector)
                )
                self.hooks.append(handle)

    def get_embedding(self, text: str) -> torch.Tensor:
        """Get concatenated embeddings from multiple layers."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(DEVICE)

        with torch.no_grad():
            self.model(**inputs)

        embeddings = [self.captured[idx].float() for idx in self.layer_indices]
        return torch.cat(embeddings, dim=-1)

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        del self.model
        del self.tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def pool_weighted(hidden: torch.Tensor) -> torch.Tensor:
    seq_len = hidden.shape[0]
    weights = torch.arange(1, seq_len + 1, device=hidden.device, dtype=hidden.dtype)
    weights = weights / weights.sum()
    return (hidden * weights.unsqueeze(1)).sum(dim=0)


def extract_and_classify(extractor, train_pairs, test_pairs):
    """Extract embeddings and train classifier."""
    # Extract embeddings
    train_a, train_b, y_train = [], [], []
    for pair in train_pairs:
        emb_a = extractor.get_embedding(pair.tweet_a.text)
        emb_b = extractor.get_embedding(pair.tweet_b.text)
        train_a.append(pool_weighted(emb_a).cpu())
        train_b.append(pool_weighted(emb_b).cpu())
        y_train.append(pair.label)

    test_a, test_b, y_test = [], [], []
    for pair in test_pairs:
        emb_a = extractor.get_embedding(pair.tweet_a.text)
        emb_b = extractor.get_embedding(pair.tweet_b.text)
        test_a.append(pool_weighted(emb_a).cpu())
        test_b.append(pool_weighted(emb_b).cpu())
        y_test.append(pair.label)

    # Make features
    def make_features(a_list, b_list):
        features = []
        for a, b in zip(a_list, b_list):
            feat = torch.cat([a, b, a - b, a * b])
            features.append(feat.numpy())
        return np.array(features)

    X_train = make_features(train_a, train_b)
    X_test = make_features(test_a, test_b)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Standardize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std

    # Simple logistic regression on GPU
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=DEVICE)

    n, d = X_train_t.shape
    w = torch.zeros(d, device=DEVICE, requires_grad=True)
    b = torch.zeros(1, device=DEVICE, requires_grad=True)

    optimizer = torch.optim.LBFGS([w, b], lr=0.1, max_iter=500)

    def closure():
        optimizer.zero_grad()
        logits = X_train_t @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y_train_t)
        loss = loss + 0.5 * (w ** 2).sum()  # L2 regularization
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        preds = (X_test_t @ w + b > 0).cpu().numpy().astype(int)
        acc = (preds == y_test).mean()

    return acc


def run():
    print("=" * 70)
    print("Steering Vector Experiment")
    print("=" * 70)

    train_pairs, test_pairs = get_pairs()
    print(f"\nTrain: {len(train_pairs)} pairs")
    print(f"Test: {len(test_pairs)} pairs")

    # Step 1: Collect steering data
    print("\n" + "=" * 70)
    print("STEP 1: Collecting steering data")
    print("=" * 70)

    extractor = SteeringVectorExtractor(MODEL_ID)

    # Use middle layer for steering
    steering_layer = extractor.n_layers // 2
    print(f"Using layer {steering_layer} for steering vector")

    print("\nAsking model to predict engagement (with reasoning)...")
    correct_acts, incorrect_acts, correct_pairs, incorrect_pairs = collect_steering_data(
        extractor, train_pairs, steering_layer, n_correct=20, n_incorrect=20
    )

    print(f"\nCollected: {len(correct_acts)} correct, {len(incorrect_acts)} incorrect")

    if len(correct_acts) < 5 or len(incorrect_acts) < 5:
        print("Not enough data collected. Exiting.")
        return

    # Step 2: Compute steering vector
    print("\n" + "=" * 70)
    print("STEP 2: Computing steering vector")
    print("=" * 70)

    steering_vector = compute_steering_vector(correct_acts, incorrect_acts)
    print(f"Steering vector shape: {steering_vector.shape}")
    print(f"Steering vector norm: {steering_vector.norm().item():.4f}")

    extractor.cleanup()

    # Step 3: Test with different steering strengths
    print("\n" + "=" * 70)
    print("STEP 3: Testing steering strengths")
    print("=" * 70)

    # First, baseline without steering
    print("\nBaseline (no steering)...")
    baseline_extractor = SteeredEmbeddingExtractor(
        MODEL_ID, steering_vector=None, steering_layers=None, steering_strength=0.0
    )
    baseline_acc = extract_and_classify(baseline_extractor, train_pairs, test_pairs)
    print(f"Baseline accuracy: {baseline_acc * 100:.1f}%")
    baseline_extractor.cleanup()

    results = {"baseline": baseline_acc}

    # Test single layer steering with various strengths
    print("\n--- Single Layer Steering (layer 12) ---")
    strengths = [5.0, 10.0, 20.0, 50.0, 100.0]
    for strength in strengths:
        print(f"\nSteering strength = {strength}...")
        steered_extractor = SteeredEmbeddingExtractor(
            MODEL_ID,
            steering_vector=steering_vector,
            steering_layers=[steering_layer],
            steering_strength=strength
        )
        acc = extract_and_classify(steered_extractor, train_pairs, test_pairs)
        results[f"L12_s{strength}"] = acc
        print(f"Accuracy: {acc * 100:.1f}%")
        steered_extractor.cleanup()

    # Test multi-layer steering
    print("\n--- Multi-Layer Steering ---")
    n_layers = 24  # Qwen 0.5B
    layer_configs = [
        ([8, 12, 16], "L8_12_16"),
        ([6, 9, 12, 15, 18], "L6-18"),
        ([12, 14, 16, 18], "L12-18"),
        (list(range(6, 18)), "L6-17_all"),
    ]

    for layers_list, name in layer_configs:
        for strength in [10.0, 50.0]:
            print(f"\n{name}, strength={strength}...")
            steered_extractor = SteeredEmbeddingExtractor(
                MODEL_ID,
                steering_vector=steering_vector,
                steering_layers=layers_list,
                steering_strength=strength
            )
            acc = extract_and_classify(steered_extractor, train_pairs, test_pairs)
            results[f"{name}_s{strength}"] = acc
            print(f"Accuracy: {acc * 100:.1f}%")
            steered_extractor.cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if acc == max(results.values()) else ""
        print(f"  {name:20s}: {acc * 100:.1f}%{marker}")

    best = max(results, key=results.get)
    improvement = results[best] - results["baseline"]
    print(f"\nBest: {best} = {results[best] * 100:.1f}%")
    print(f"Improvement over baseline: {improvement * 100:+.1f}%")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    run()
