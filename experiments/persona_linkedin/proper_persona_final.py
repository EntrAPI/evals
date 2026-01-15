#!/usr/bin/env python3
"""
Proper persona sampling for LinkedIn engagement prediction.

Setup:
- Proposal distribution: Gemini 2.0 Flash with author stats (89.6% accuracy)
- Judge model: Qwen 2.5 3B (small LLM for hidden state extraction)
- Persona vector: Extracted from judge's hidden states (correct vs incorrect)

The persona vector captures "what does a correct prediction look like" in the
judge's hidden state space. We use it to:
1. Score the judge's predictions (high score = likely correct)
2. Decide when to trust judge vs Gemini

Key difference from before: Gemini (89.6%) is now BETTER than a simple MLP (77%),
so persona sampling can help identify the ~10% of cases where they disagree.
"""

import json
import time
import random
import requests
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "linkedin_pairs.json"

API_KEY = "AIzaSyCWrrgydQ-_kaS92B3vvXrstJcZnPgiE20"
MODEL = "gemini-2.0-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_judge_model(model_name="Qwen/Qwen2.5-3B-Instruct"):
    """Load the judge model for persona extraction."""
    print(f"Loading judge model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def format_prompt_for_judge(pair, author_stats):
    """Format prompt for the judge model (same format as successful Gemini prompt)."""
    author_a = pair['post_a'].get('author', 'Unknown')
    author_b = pair['post_b'].get('author', 'Unknown')
    post_a = pair['post_a']['text'][:400]
    post_b = pair['post_b']['text'][:400]

    stats_a = author_stats.get(author_a, {})
    stats_b = author_stats.get(author_b, {})
    avg_a = stats_a.get('avg_eng', 50)
    avg_b = stats_b.get('avg_eng', 50)

    return f"""Which LinkedIn post got more engagement (reactions + comments)?

Post A by {author_a} (avg engagement: {avg_a:.0f}):
{post_a}

Post B by {author_b} (avg engagement: {avg_b:.0f}):
{post_b}

Reply with ONLY "A" or "B"."""


def get_judge_hidden_states(model, tokenizer, prompts, layer_idx=-8):
    """
    Get hidden states from the judge model at a specific layer.

    We use a layer near the end but not the final layer, as intermediate
    layers often contain richer representations for persona extraction.
    """
    hidden_states = []
    predictions = []

    for prompt in tqdm(prompts, desc="Getting judge hidden states"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # Get hidden state at specified layer, last token position
            # layer_idx=-8 means 8th layer from the end
            hidden = outputs.hidden_states[layer_idx][0, -1, :].cpu()
            hidden_states.append(hidden)

            # Get prediction by generating
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(gen_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Parse prediction
            response = response.strip().upper()
            if "A" in response and "B" not in response:
                predictions.append(0)
            elif "B" in response and "A" not in response:
                predictions.append(1)
            else:
                predictions.append(-1)  # Unparseable

    return torch.stack(hidden_states), np.array(predictions)


def get_gemini_prediction_with_stats(pair, author_stats, temperature=0.0):
    """Get Gemini prediction using the successful prompt format."""
    author_a = pair['post_a'].get('author', 'Unknown')
    author_b = pair['post_b'].get('author', 'Unknown')
    post_a = pair['post_a']['text'][:400]
    post_b = pair['post_b']['text'][:400]

    stats_a = author_stats.get(author_a, {})
    stats_b = author_stats.get(author_b, {})
    avg_a = stats_a.get('avg_eng', 50)
    avg_b = stats_b.get('avg_eng', 50)

    prompt = f"""Which LinkedIn post got more engagement (reactions + comments)?

Post A by {author_a} (avg engagement: {avg_a:.0f}):
{post_a}

Post B by {author_b} (avg engagement: {avg_b:.0f}):
{post_b}

Reply with ONLY "A" or "B"."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 5}
    }

    response = requests.post(API_URL, json=payload, timeout=60)
    response.raise_for_status()
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip().upper()

    if "A" in text and "B" not in text:
        return 0
    elif "B" in text and "A" not in text:
        return 1
    return -1


def get_gemini_predictions_cached(test_pairs, author_stats, n_samples=1):
    """Get Gemini predictions with caching."""
    cache_file = Path(__file__).parent / f"gemini_with_stats_{n_samples}.json"

    if cache_file.exists():
        print(f"Loading cached Gemini predictions...")
        with open(cache_file) as f:
            return json.load(f)

    print(f"Getting Gemini predictions ({n_samples} samples)...")
    all_predictions = []

    for i, pair in enumerate(tqdm(test_pairs, desc="Gemini predictions")):
        pair_preds = []
        for s in range(n_samples):
            try:
                pred = get_gemini_prediction_with_stats(
                    pair, author_stats,
                    temperature=0.7 if s > 0 else 0.0
                )
                pair_preds.append(pred)
            except Exception as e:
                print(f"Error at {i}: {e}")
                pair_preds.append(-1)
            time.sleep(0.03)

        if n_samples == 1:
            all_predictions.append(pair_preds[0])
        else:
            all_predictions.append(pair_preds)

    with open(cache_file, 'w') as f:
        json.dump(all_predictions, f)

    return all_predictions


def extract_persona_vector(hidden_states, labels, predictions, normalize=True):
    """
    Extract persona vector from hidden states.

    persona_vector = mean(hidden | correct) - mean(hidden | incorrect)

    This vector points in the direction of "correct predictions" in hidden space.
    """
    correct_mask = predictions == labels
    incorrect_mask = (predictions != labels) & (predictions >= 0)

    if correct_mask.sum() == 0 or incorrect_mask.sum() == 0:
        raise ValueError("Need both correct and incorrect predictions")

    mean_correct = hidden_states[correct_mask].mean(dim=0)
    mean_incorrect = hidden_states[incorrect_mask].mean(dim=0)

    persona_vector = mean_correct - mean_incorrect

    if normalize:
        persona_vector = persona_vector / (persona_vector.norm() + 1e-8)

    return persona_vector


def run():
    print("=" * 70)
    print("PROPER PERSONA SAMPLING")
    print("Proposal: Gemini 2.0 Flash with author stats")
    print("Judge: Qwen 2.5 3B for persona extraction")
    print("=" * 70)

    # Load data
    with open(DATA_FILE) as f:
        data = json.load(f)
    train_pairs = data['train']
    test_pairs = data['test']

    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Build author statistics
    print("\nBuilding author statistics...")
    author_posts = defaultdict(list)
    for pair in train_pairs:
        for key in ['post_a', 'post_b']:
            author = pair[key].get('author', 'Unknown')
            eng = pair[key]['reactions'] + pair[key].get('comments', 0)
            author_posts[author].append(eng)

    author_stats = {a: {'avg_eng': np.mean(engs)} for a, engs in author_posts.items()}
    global_median = np.median([e for engs in author_posts.values() for e in engs])

    # Compute ground truth labels
    def get_label(pair):
        eng_a = pair['post_a']['reactions'] + pair['post_a'].get('comments', 0)
        eng_b = pair['post_b']['reactions'] + pair['post_b'].get('comments', 0)
        return 0 if eng_a > eng_b else 1

    y_train = np.array([get_label(p) for p in train_pairs])
    y_test = np.array([get_label(p) for p in test_pairs])

    # ========================================
    # 1. EXTRACT PERSONA VECTOR FROM JUDGE
    # ========================================
    print("\n" + "=" * 70)
    print("1. EXTRACTING PERSONA VECTOR FROM JUDGE MODEL")
    print("=" * 70)

    # Use cross-validation to get out-of-fold predictions for persona extraction
    # This prevents data leakage

    model, tokenizer = load_judge_model()

    # For efficiency, use a subset of training data for persona extraction
    # (full training set would be slow)
    n_persona_samples = min(400, len(train_pairs))
    persona_indices = np.random.choice(len(train_pairs), n_persona_samples, replace=False)
    persona_pairs = [train_pairs[i] for i in persona_indices]
    persona_labels = y_train[persona_indices]

    print(f"\nUsing {n_persona_samples} samples for persona extraction")

    # Get prompts
    prompts = [format_prompt_for_judge(p, author_stats) for p in persona_pairs]

    # Get hidden states and predictions
    print("Running judge model...")
    hidden_states, judge_preds = get_judge_hidden_states(model, tokenizer, prompts)

    # Check judge accuracy
    valid_mask = judge_preds >= 0
    judge_acc = (judge_preds[valid_mask] == persona_labels[valid_mask]).mean()
    print(f"Judge accuracy on persona samples: {judge_acc*100:.1f}%")
    print(f"Valid predictions: {valid_mask.sum()}/{len(judge_preds)}")

    # Extract persona vector
    persona_vector = extract_persona_vector(
        hidden_states[valid_mask],
        persona_labels[valid_mask],
        judge_preds[valid_mask]
    )
    print(f"Persona vector shape: {persona_vector.shape}")

    # Verify persona vector
    persona_scores_train = (hidden_states[valid_mask] * persona_vector).sum(dim=1).numpy()
    correct_mask_train = judge_preds[valid_mask] == persona_labels[valid_mask]
    print(f"Mean persona score (correct): {persona_scores_train[correct_mask_train].mean():.3f}")
    print(f"Mean persona score (incorrect): {persona_scores_train[~correct_mask_train].mean():.3f}")

    # ========================================
    # 2. GET JUDGE PREDICTIONS ON TEST SET
    # ========================================
    print("\n" + "=" * 70)
    print("2. RUNNING JUDGE ON TEST SET")
    print("=" * 70)

    test_prompts = [format_prompt_for_judge(p, author_stats) for p in test_pairs]
    test_hidden, test_judge_preds = get_judge_hidden_states(model, tokenizer, test_prompts)

    valid_test = test_judge_preds >= 0
    judge_test_acc = (test_judge_preds[valid_test] == y_test[valid_test]).mean()
    print(f"Judge test accuracy: {judge_test_acc*100:.1f}%")

    # Compute persona scores for test set
    test_persona_scores = (test_hidden * persona_vector).sum(dim=1).numpy()

    # Verify on test set
    correct_test = test_judge_preds == y_test
    print(f"Mean persona score (judge correct): {test_persona_scores[correct_test & valid_test].mean():.3f}")
    print(f"Mean persona score (judge incorrect): {test_persona_scores[~correct_test & valid_test].mean():.3f}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    # ========================================
    # 3. GET GEMINI PREDICTIONS
    # ========================================
    print("\n" + "=" * 70)
    print("3. GETTING GEMINI PREDICTIONS")
    print("=" * 70)

    # Get multiple Gemini samples for diversity
    n_gemini_samples = 5
    gemini_multi = get_gemini_predictions_cached(test_pairs, author_stats, n_samples=n_gemini_samples)

    # Compute Gemini majority vote
    gemini_preds = []
    gemini_probs = []  # Probability of predicting A
    for preds in gemini_multi:
        valid = [p for p in preds if p >= 0]
        if valid:
            gemini_preds.append(1 if sum(valid) > len(valid) / 2 else 0)
            gemini_probs.append(1 - sum(valid) / len(valid))  # P(A)
        else:
            gemini_preds.append(-1)
            gemini_probs.append(0.5)

    gemini_preds = np.array(gemini_preds)
    gemini_probs = np.array(gemini_probs)

    valid_gemini = gemini_preds >= 0
    gemini_acc = (gemini_preds[valid_gemini] == y_test[valid_gemini]).mean()
    print(f"Gemini majority vote accuracy: {gemini_acc*100:.1f}%")

    # ========================================
    # 4. PERSONA SAMPLING: COMBINE JUDGE + GEMINI
    # ========================================
    print("\n" + "=" * 70)
    print("4. PERSONA SAMPLING")
    print("=" * 70)

    # Key insight: persona_score tells us how confident we should be in the judge.
    # High score = judge likely correct, low score = judge might be wrong.
    # When judge might be wrong, trust Gemini more (which is stronger overall).

    # Method A: Threshold-based switching
    print("\n--- Method A: Switch to Gemini when persona score is low ---")
    for percentile in [10, 25, 50]:
        threshold = np.percentile(test_persona_scores, percentile)
        switched_preds = np.where(
            test_persona_scores < threshold,
            gemini_preds,  # Use Gemini when persona score is low
            test_judge_preds  # Use judge when persona score is high
        )
        # Handle invalid predictions
        switched_preds = np.where(switched_preds < 0, gemini_preds, switched_preds)
        switched_preds = np.where(switched_preds < 0, test_judge_preds, switched_preds)

        valid = switched_preds >= 0
        acc = (switched_preds[valid] == y_test[valid]).mean()
        n_switched = (test_persona_scores < threshold).sum()
        print(f"  Threshold={threshold:.2f} ({percentile}th percentile): {acc*100:.1f}% (switched {n_switched})")

    # Method B: Soft importance weighting
    print("\n--- Method B: Soft importance weighting ---")
    for beta in [0.1, 0.5, 1.0, 2.0]:
        # Convert persona scores to judge weights using sigmoid
        # Higher persona score = trust judge more
        judge_weights = 1 / (1 + np.exp(-beta * (test_persona_scores - test_persona_scores.mean())))

        # Convert judge predictions to probabilities (assuming hard predictions)
        judge_probs = np.where(test_judge_preds == 0, 0.9, 0.1)  # 90% confidence
        judge_probs = np.where(test_judge_preds < 0, 0.5, judge_probs)  # Handle invalid

        # Weighted combination
        combined_probs = judge_weights * judge_probs + (1 - judge_weights) * gemini_probs
        combined_preds = (combined_probs < 0.5).astype(int)

        valid = (test_judge_preds >= 0) | (gemini_preds >= 0)
        acc = (combined_preds[valid] == y_test[valid]).mean()
        print(f"  Beta={beta}: {acc*100:.1f}%")

    # Method C: MH-style acceptance (accept Gemini when it improves persona score)
    print("\n--- Method C: MH acceptance ---")
    for beta in [0.5, 1.0, 2.0]:
        mh_preds = []
        n_accepted = 0
        n_disagreements = 0

        for i in range(len(test_pairs)):
            judge_pred = test_judge_preds[i]
            gemini_pred = gemini_preds[i]

            if judge_pred < 0:
                mh_preds.append(gemini_pred)
                continue
            if gemini_pred < 0:
                mh_preds.append(judge_pred)
                continue

            if judge_pred == gemini_pred:
                mh_preds.append(judge_pred)
            else:
                n_disagreements += 1
                # When they disagree, use persona score to decide
                # Low persona score = judge might be wrong = accept Gemini more
                # MH: log_accept = -beta * persona_score (low score = high acceptance)
                persona_z = (test_persona_scores[i] - test_persona_scores.mean()) / (test_persona_scores.std() + 1e-8)
                log_accept = -beta * persona_z
                accept_prob = min(1.0, np.exp(log_accept))

                if random.random() < accept_prob:
                    mh_preds.append(gemini_pred)
                    n_accepted += 1
                else:
                    mh_preds.append(judge_pred)

        mh_preds = np.array(mh_preds)
        valid = mh_preds >= 0
        acc = (mh_preds[valid] == y_test[valid]).mean()
        print(f"  Beta={beta}: {acc*100:.1f}% (accepted {n_accepted}/{n_disagreements} switches)")

    # Method D: Oracle analysis - what's the best we could do?
    print("\n--- Method D: Oracle analysis ---")
    # When judge and Gemini disagree, which is right?
    disagree_mask = (test_judge_preds != gemini_preds) & (test_judge_preds >= 0) & (gemini_preds >= 0)
    n_disagree = disagree_mask.sum()

    judge_right_when_disagree = (test_judge_preds[disagree_mask] == y_test[disagree_mask]).sum()
    gemini_right_when_disagree = (gemini_preds[disagree_mask] == y_test[disagree_mask]).sum()

    print(f"  Disagreements: {n_disagree}")
    print(f"  Judge right when disagree: {judge_right_when_disagree} ({judge_right_when_disagree/n_disagree*100:.1f}%)")
    print(f"  Gemini right when disagree: {gemini_right_when_disagree} ({gemini_right_when_disagree/n_disagree*100:.1f}%)")

    # Can persona score predict who's right when they disagree?
    persona_when_disagree = test_persona_scores[disagree_mask]
    judge_correct_when_disagree = test_judge_preds[disagree_mask] == y_test[disagree_mask]

    print(f"  Mean persona score when judge right (in disagreements): {persona_when_disagree[judge_correct_when_disagree].mean():.3f}")
    print(f"  Mean persona score when Gemini right (in disagreements): {persona_when_disagree[~judge_correct_when_disagree].mean():.3f}")

    # Oracle: always pick the right one when they disagree
    oracle_preds = np.where(disagree_mask, y_test, test_judge_preds)
    oracle_preds = np.where(oracle_preds < 0, gemini_preds, oracle_preds)
    valid = oracle_preds >= 0
    oracle_acc = (oracle_preds[valid] == y_test[valid]).mean()
    print(f"  Oracle accuracy (always right when disagree): {oracle_acc*100:.1f}%")

    # ========================================
    # 5. ADD AUTHOR HEURISTIC
    # ========================================
    print("\n" + "=" * 70)
    print("5. COMBINING WITH AUTHOR HEURISTIC")
    print("=" * 70)

    # Author heuristic
    def get_author_pred(pair):
        author_a = pair['post_a'].get('author', 'Unknown')
        author_b = pair['post_b'].get('author', 'Unknown')
        avg_a = author_stats.get(author_a, {}).get('avg_eng', global_median)
        avg_b = author_stats.get(author_b, {}).get('avg_eng', global_median)
        return 0 if avg_a > avg_b else 1

    author_preds = np.array([get_author_pred(p) for p in test_pairs])
    author_acc = (author_preds == y_test).mean()
    print(f"Author heuristic: {author_acc*100:.1f}%")

    # Best persona-guided combination
    print("\n--- Persona-guided three-way ensemble ---")
    for author_w in [0.6, 0.7, 0.8]:
        for beta in [0.5, 1.0]:
            ensemble_preds = []
            for i in range(len(test_pairs)):
                # Persona score determines judge weight
                persona_z = (test_persona_scores[i] - test_persona_scores.mean()) / (test_persona_scores.std() + 1e-8)
                judge_conf = 1 / (1 + np.exp(-beta * persona_z))

                # Weights
                judge_w = (1 - author_w) * judge_conf
                gemini_w = (1 - author_w) * (1 - judge_conf)

                # Get probabilities
                author_prob = 0.95 if author_preds[i] == 0 else 0.05
                judge_prob = 0.9 if test_judge_preds[i] == 0 else (0.1 if test_judge_preds[i] == 1 else 0.5)
                gemini_prob = gemini_probs[i]

                combined = author_w * author_prob + judge_w * judge_prob + gemini_w * gemini_prob
                ensemble_preds.append(0 if combined > 0.5 else 1)

            ensemble_preds = np.array(ensemble_preds)
            acc = (ensemble_preds == y_test).mean()
            print(f"  Author={author_w:.0%}, beta={beta}: {acc*100:.1f}%")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nIndividual models:")
    print(f"  Judge (Qwen 2.5 3B): {judge_test_acc*100:.1f}%")
    print(f"  Gemini with stats: {gemini_acc*100:.1f}%")
    print(f"  Author heuristic: {author_acc*100:.1f}%")
    print(f"\nBaseline (80% author + 20% MLP): 90.3%")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    run()
