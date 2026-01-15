#!/usr/bin/env python3
"""
Tweet A/B Testing with XGBoost on Activations

1. Get model embeddings for high/low engagement tweets
2. Train XGBoost on steering set (10-49 likes vs 0)
3. Test on test set (50+ likes vs 0)
"""

import random
import re
from dataclasses import dataclass

import numpy as np
import torch
import xgboost as xgb
from datasets import load_dataset
from langdetect import detect, LangDetectException
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Tweet:
    text: str
    likes: int
    retweets: int


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


class EmbeddingExtractor:
    def __init__(self, model_id: str = MODEL_ID, target_layer: int = None):
        print(f"\nLoading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(DEVICE)
        self.model.eval()

        n_layers = self.model.config.num_hidden_layers
        self.target_layer = target_layer if target_layer else n_layers // 2
        print(f"Model has {n_layers} layers, extracting from layer {self.target_layer}")

        self.captured_activation = None
        self._register_hook()

    def _register_hook(self):
        layer = self.model.model.layers[self.target_layer]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output[0]

            # Take last token's hidden state
            if hidden_states.dim() == 3:
                self.captured_activation = hidden_states[0, -1, :].detach().cpu()
            elif hidden_states.dim() == 2:
                self.captured_activation = hidden_states[-1, :].detach().cpu()

        layer.register_forward_hook(hook_fn)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single tweet."""
        messages = [
            {"role": "system", "content": "Analyze this tweet."},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            self.model(**inputs)

        return self.captured_activation.numpy()


def run_experiment():
    print("=" * 60)
    print("Tweet Classification with XGBoost on Activations")
    print("=" * 60)

    tweets = load_tweets()

    # Split tweets by engagement level
    high_engagement = [t for t in tweets if t.likes >= 50]  # Test set source
    medium_engagement = [t for t in tweets if 10 <= t.likes < 50]  # Train set source
    zero_engagement = [t for t in tweets if t.likes == 0]

    print(f"\nTweet counts:")
    print(f"  50+ likes (test high): {len(high_engagement)}")
    print(f"  10-49 likes (train high): {len(medium_engagement)}")
    print(f"  0 likes: {len(zero_engagement)}")

    random.shuffle(medium_engagement)
    random.shuffle(zero_engagement)
    random.shuffle(high_engagement)

    # Training set: 10-49 likes vs 0 likes (50 each)
    n_train = min(50, len(medium_engagement))
    train_high = medium_engagement[:n_train]
    train_low = zero_engagement[:n_train]

    # Test set: 50+ likes vs 0 likes (use remaining 0-likes tweets)
    n_test = len(high_engagement)
    test_high = high_engagement
    test_low = zero_engagement[n_train:n_train + n_test]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_high)} high + {len(train_low)} low = {len(train_high) + len(train_low)}")
    print(f"  Test: {len(test_high)} high + {len(test_low)} low = {len(test_high) + len(test_low)}")

    # Extract embeddings
    extractor = EmbeddingExtractor()

    print("\nExtracting training embeddings...")
    X_train = []
    y_train = []
    for i, tweet in enumerate(train_high + train_low):
        emb = extractor.get_embedding(tweet.text)
        X_train.append(emb)
        y_train.append(1 if tweet.likes >= 10 else 0)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(train_high) + len(train_low)}]")

    print("\nExtracting test embeddings...")
    X_test = []
    y_test = []
    for i, tweet in enumerate(test_high + test_low):
        emb = extractor.get_embedding(tweet.text)
        X_test.append(emb)
        y_test.append(1 if tweet.likes >= 50 else 0)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(test_high) + len(test_low)}]")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"\nFeature shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")

    # Train XGBoost
    print("\n" + "=" * 60)
    print("Training XGBoost")
    print("=" * 60)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nTrain accuracy: {train_acc*100:.1f}%")
    print(f"Test accuracy: {test_acc*100:.1f}%")
    print(f"Random baseline: 50.0%")

    print("\nTest set classification report:")
    print(classification_report(y_test, test_pred, target_names=["Low (0 likes)", "High (50+ likes)"]))

    # Feature importance (top 10)
    importance = model.feature_importances_
    top_features = np.argsort(importance)[-10:][::-1]
    print("\nTop 10 most important features (activation dimensions):")
    for i, idx in enumerate(top_features):
        print(f"  {i+1}. Dim {idx}: {importance[idx]:.4f}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_experiment()
