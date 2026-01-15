#!/usr/bin/env python3
"""
Tweet A/B Testing with Steering Vectors

1. Run model on tweet pairs, collect activations for correct vs incorrect predictions
2. Compute steering vector: mean(correct_activations) - mean(incorrect_activations)
3. Apply steering vector and test if accuracy improves
"""

import random
import re
from dataclasses import dataclass

import torch
from datasets import load_dataset
from langdetect import detect, LangDetectException
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Tweet:
    text: str
    likes: int
    retweets: int

    @property
    def engagement(self) -> int:
        return self.likes + self.retweets


def is_english(text: str) -> bool:
    clean = re.sub(r'https?://\S+', '', text)
    clean = re.sub(r'@\w+', '', clean).strip()
    if len(clean) < 20:
        return False
    try:
        return detect(clean) == 'en'
    except LangDetectException:
        return False


def load_tweets() -> list[Tweet]:
    print("Loading dataset...")
    ds = load_dataset("hugginglearners/twitter-dataset-tesla", split="train")

    tweets = []
    for row in ds:
        text = row["tweet"]
        if row["retweet"] or text.startswith("RT") or text.startswith("@"):
            continue
        if len(text) < 30:
            continue
        if not is_english(text):
            continue
        text_without_urls = re.sub(r'https?://\S+', '', text).strip()
        if len(text_without_urls) < 20:
            continue

        tweets.append(Tweet(
            text=text,
            likes=int(row["nlikes"] or 0),
            retweets=int(row["nretweets"] or 0),
        ))

    print(f"Loaded {len(tweets)} tweets")
    return tweets


def create_test_pairs(tweets: list[Tweet]) -> list[tuple[Tweet, Tweet, int]]:
    """Test pairs: 50+ likes vs 0 likes (same as original experiment)."""
    high = [t for t in tweets if t.likes >= 50]
    low = [t for t in tweets if t.likes == 0]

    random.shuffle(high)
    random.shuffle(low)

    pairs = []
    for i in range(min(len(high), len(low))):
        if random.random() < 0.5:
            pairs.append((high[i], low[i], 0))
        else:
            pairs.append((low[i], high[i], 1))

    print(f"Test pairs: {len(pairs)} (50+ likes: {len(high)}, 0 likes: {len(low)})")
    return pairs


def create_steering_pairs(tweets: list[Tweet], n_pairs: int = 50) -> list[tuple[Tweet, Tweet, int]]:
    """Steering pairs: 10-49 likes vs 0 likes (separate from test set)."""
    # Use medium engagement tweets (10-49 likes) for steering
    medium = [t for t in tweets if 10 <= t.likes < 50]
    low = [t for t in tweets if t.likes == 0]

    random.shuffle(medium)
    random.shuffle(low)

    pairs = []
    for i in range(min(n_pairs, len(medium), len(low))):
        if random.random() < 0.5:
            pairs.append((medium[i], low[i], 0))
        else:
            pairs.append((low[i], medium[i], 1))

    print(f"Steering pairs: {len(pairs)} (10-49 likes: {len(medium)}, 0 likes available: {len(low)})")
    return pairs


class SteeringPredictor:
    def __init__(self, model_id: str = MODEL_ID, target_layer: int = None):
        print(f"\nLoading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(DEVICE)
        self.model.eval()

        # Determine target layer (middle of model)
        n_layers = self.model.config.num_hidden_layers
        self.target_layer = target_layer if target_layer else n_layers // 2
        print(f"Model has {n_layers} layers, targeting layer {self.target_layer}")

        # Storage for activations
        self.captured_activation = None
        self.steering_vector = None
        self.steering_coef = 0.0

        # Register hook
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hook to capture/modify activations."""
        layer = self.model.model.layers[self.target_layer]

        def hook_fn(module, input, output):
            # output can be tuple or BaseModelOutputWithPast
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output[0]

            # Capture activation - take last token's hidden state
            # hidden_states shape varies: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
            if hidden_states.dim() == 3:
                self.captured_activation = hidden_states[0, -1, :].detach().clone()
            elif hidden_states.dim() == 2:
                self.captured_activation = hidden_states[-1, :].detach().clone()
            else:
                self.captured_activation = hidden_states.flatten()[-1536:].detach().clone()  # fallback

            # Apply steering if we have a vector
            if self.steering_vector is not None and self.steering_coef != 0:
                # Add steering vector to all positions
                if hidden_states.dim() == 3:
                    steered = hidden_states + self.steering_coef * self.steering_vector.view(1, 1, -1)
                elif hidden_states.dim() == 2:
                    steered = hidden_states + self.steering_coef * self.steering_vector.view(1, -1)
                else:
                    steered = hidden_states + self.steering_coef * self.steering_vector

                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                else:
                    output.last_hidden_state = steered
                    return output

            return output

        layer.register_forward_hook(hook_fn)

    def _build_prompt(self, tweet: str) -> str:
        messages = [
            {"role": "system", "content": "Rate tweet engagement potential from 1-10. Output only a number."},
            {"role": "user", "content": f"Rate this tweet's viral potential (1=low, 10=high):\n\n{tweet}\n\nScore:"}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def score_tweet(self, tweet: str) -> tuple[float | None, torch.Tensor]:
        """Score a tweet and return (score, activation)."""
        text = self._build_prompt(tweet)
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        score = float(numbers[0]) if numbers else None

        return score, self.captured_activation.clone()

    def predict(self, tweet_a: str, tweet_b: str) -> tuple[int | None, str, torch.Tensor, torch.Tensor]:
        """Predict which tweet wins, return (prediction, info, act_a, act_b)."""
        score_a, act_a = self.score_tweet(tweet_a)
        score_b, act_b = self.score_tweet(tweet_b)

        if score_a is None or score_b is None:
            return None, f"scores: {score_a}, {score_b}", act_a, act_b

        if score_a > score_b:
            return 0, f"{score_a:.1f} vs {score_b:.1f}", act_a, act_b
        elif score_b > score_a:
            return 1, f"{score_a:.1f} vs {score_b:.1f}", act_a, act_b
        else:
            return random.choice([0, 1]), f"{score_a:.1f} vs {score_b:.1f} (tie)", act_a, act_b

    def set_steering(self, vector: torch.Tensor, coefficient: float):
        """Set the steering vector and coefficient."""
        self.steering_vector = vector.to(DEVICE)
        self.steering_coef = coefficient
        print(f"Steering enabled: coef={coefficient}, vector norm={vector.norm():.2f}")

    def clear_steering(self):
        """Disable steering."""
        self.steering_vector = None
        self.steering_coef = 0.0


def run_experiment():
    print("=" * 60)
    print("Tweet A/B Testing with Steering Vectors")
    print("=" * 60)

    # Load data
    tweets = load_tweets()

    # Create separate sets
    steering_pairs = create_steering_pairs(tweets, n_pairs=50)  # 10-49 likes vs 0
    test_pairs = create_test_pairs(tweets)  # 50+ likes vs 0 (same as original)

    # Load model
    predictor = SteeringPredictor()

    # Phase 1: Collect activations on steering pairs (10-49 likes vs 0)
    print("\n" + "=" * 60)
    print("Phase 1: Collecting activations on steering pairs (no steering)")
    print("=" * 60)

    correct_activations = []
    incorrect_activations = []
    steering_correct = 0

    for i, (t1, t2, actual_winner) in enumerate(steering_pairs):
        prediction, info, act_a, act_b = predictor.predict(t1.text, t2.text)

        # Average the activations from both tweets in the pair
        pair_activation = (act_a + act_b) / 2

        if prediction == actual_winner:
            correct_activations.append(pair_activation)
            steering_correct += 1
            status = "✓"
        elif prediction is not None:
            incorrect_activations.append(pair_activation)
            status = "✗"
        else:
            status = "?"

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(steering_pairs)}] {status} | Steering set acc: {steering_correct}/{i+1}")

    print(f"\nSteering set accuracy: {steering_correct}/{len(steering_pairs)} ({steering_correct/len(steering_pairs)*100:.1f}%)")
    print(f"Correct activations: {len(correct_activations)}")
    print(f"Incorrect activations: {len(incorrect_activations)}")

    if len(correct_activations) < 2 or len(incorrect_activations) < 2:
        print("Not enough samples to compute steering vector!")
        return

    # Compute steering vector
    mean_correct = torch.stack(correct_activations).mean(dim=0)
    mean_incorrect = torch.stack(incorrect_activations).mean(dim=0)
    steering_vector = mean_correct - mean_incorrect

    print(f"\nSteering vector computed:")
    print(f"  Shape: {steering_vector.shape}")
    print(f"  Norm: {steering_vector.norm():.4f}")
    print(f"  Mean correct norm: {mean_correct.norm():.4f}")
    print(f"  Mean incorrect norm: {mean_incorrect.norm():.4f}")

    # Phase 2: Test WITHOUT steering
    print("\n" + "=" * 60)
    print("Phase 2: Testing WITHOUT steering")
    print("=" * 60)

    predictor.clear_steering()
    baseline_correct = 0

    for i, (t1, t2, actual_winner) in enumerate(test_pairs):
        prediction, info, _, _ = predictor.predict(t1.text, t2.text)
        if prediction == actual_winner:
            baseline_correct += 1

        if (i + 1) % 5 == 0 or i == len(test_pairs) - 1:
            print(f"[{i+1}/{len(test_pairs)}] Baseline acc: {baseline_correct}/{i+1} ({baseline_correct/(i+1)*100:.1f}%)")

    baseline_acc = baseline_correct / len(test_pairs)
    print(f"\nBaseline accuracy: {baseline_correct}/{len(test_pairs)} ({baseline_acc*100:.1f}%)")

    # Phase 3: Test WITH steering (try different coefficients)
    print("\n" + "=" * 60)
    print("Phase 3: Testing WITH steering")
    print("=" * 60)

    best_coef = 0
    best_acc = baseline_acc

    for coef in [0.5, 1.0, 2.0, 3.0, 5.0]:
        predictor.set_steering(steering_vector, coef)
        steered_correct = 0

        for t1, t2, actual_winner in test_pairs:
            prediction, _, _, _ = predictor.predict(t1.text, t2.text)
            if prediction == actual_winner:
                steered_correct += 1

        acc = steered_correct / len(test_pairs)
        print(f"Coef {coef}: {steered_correct}/{len(test_pairs)} ({acc*100:.1f}%)")

        if acc > best_acc:
            best_acc = acc
            best_coef = coef

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Baseline accuracy: {baseline_acc*100:.1f}%")
    print(f"Best steered accuracy: {best_acc*100:.1f}% (coef={best_coef})")
    print(f"Improvement: {(best_acc - baseline_acc)*100:+.1f}%")
    print(f"Random baseline: 50.0%")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    run_experiment()
