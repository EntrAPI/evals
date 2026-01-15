#!/usr/bin/env python3
"""Quick test to verify setup works."""

import torch
import requests

# Test 1: Ollama
print("=" * 50)
print("Testing Ollama...")
try:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:4b",
            "prompt": "Say hello",
            "stream": False,
            "options": {"num_predict": 10}
        },
        timeout=30
    )
    resp.raise_for_status()
    print(f"  Ollama response: {resp.json()['response'][:50]}")
    print("  Ollama: OK")
except Exception as e:
    print(f"  Ollama: FAILED - {e}")

# Test 2: HuggingFace model loading
print("\n" + "=" * 50)
print("Testing HuggingFace model loading...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"  Loading {model_name} with 4-bit quantization...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()

    # Check config
    config = model.config
    if hasattr(config, 'text_config'):
        n_layers = config.text_config.num_hidden_layers
        hidden_size = config.text_config.hidden_size
    else:
        n_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

    print(f"  Layers: {n_layers}, Hidden size: {hidden_size}")

    # Test forward pass with hidden states
    print("  Testing forward pass...")
    inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    print(f"  Hidden states: {len(outputs.hidden_states)} layers")
    print(f"  Layer 0 shape: {outputs.hidden_states[0].shape}")

    print("  HuggingFace: OK")

    del model
    torch.cuda.empty_cache()

except Exception as e:
    import traceback
    print(f"  HuggingFace: FAILED - {e}")
    traceback.print_exc()

print("\n" + "=" * 50)
print("Setup test complete!")
