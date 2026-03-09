#!/usr/bin/env python3
"""Evaluate a fine-tuned model: perplexity on validation set + sample generations.

Usage:
    python evaluate.py --model-path output/final --data-dir data
    python evaluate.py --model-path output/merged_16bit --data-dir data
"""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 256) -> float:
    """Compute perplexity over a list of texts."""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text + tokenizer.eos_token,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            input_ids = encodings["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids=input_ids, labels=input_ids)
            # Loss is averaged over tokens in the sequence
            n_tokens = input_ids.shape[1] - 1  # labels are shifted
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def generate_samples(
    model, tokenizer, prompts: list[str],
    max_new_tokens: int = 100, temperature: float = 0.7, top_p: float = 0.9,
) -> list[str]:
    """Generate completions for a list of prompts."""
    model.eval()
    device = next(model.parameters()).device
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Decode only the generated part
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        results.append(generated)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLM")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing val.jsonl")
    parser.add_argument("--max-seq-length", type=int, default=256,
                        help="Max sequence length for perplexity computation")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--compare-base", type=str, default=None,
                        help="Base model name/path to compare against (optional)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"Loading model from: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    # Try loading as PEFT model first, fall back to standard model
    try:
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(args.model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, **load_kwargs
        )
        model = PeftModel.from_pretrained(base_model, args.model_path)
        print(f"Loaded as PEFT model (base: {config.base_model_name_or_path})")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
        print("Loaded as standard model")

    # -----------------------------------------------------------------------
    # Perplexity on validation set
    # -----------------------------------------------------------------------
    val_path = Path(args.data_dir) / "val.jsonl"
    if val_path.exists():
        val_data = load_jsonl(str(val_path))
        val_texts = [s["text"] for s in val_data]
        print(f"\nComputing perplexity on {len(val_texts)} validation samples...")
        ppl = compute_perplexity(model, tokenizer, val_texts, max_length=args.max_seq_length)
        print(f"Validation perplexity: {ppl:.2f}")

        # Per-source perplexity
        sources = {}
        for s in val_data:
            src = s.get("source", "unknown")
            if src not in sources:
                sources[src] = []
            sources[src].append(s["text"])

        if len(sources) > 1:
            print("\nPer-source perplexity:")
            for src, texts in sorted(sources.items()):
                src_ppl = compute_perplexity(model, tokenizer, texts, max_length=args.max_seq_length)
                print(f"  {src}: {src_ppl:.2f} ({len(texts)} samples)")
    else:
        print(f"No validation file at {val_path}, skipping perplexity.")

    # -----------------------------------------------------------------------
    # Sample generations
    # -----------------------------------------------------------------------
    test_prompts = [
        "Word: cascade\nDefinition:",
        "Word: ephemeral\nDefinition:",
        "Continue the poem:\nThe morning light spills golden on the hill,",
        "Write a haiku about winter:",
        "Write a couplet about the moon:",
        "The key to training a small language model is",
    ]

    print(f"\n{'='*60}")
    print("Sample Generations")
    print(f"{'='*60}")

    generations = generate_samples(
        model, tokenizer, test_prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    for prompt, gen in zip(test_prompts, generations):
        print(f"\n--- Prompt ---\n{prompt}")
        print(f"--- Generated ---\n{gen}")

    # -----------------------------------------------------------------------
    # Optional: compare with base model
    # -----------------------------------------------------------------------
    if args.compare_base and val_path.exists():
        print(f"\n{'='*60}")
        print(f"Base model comparison: {args.compare_base}")
        print(f"{'='*60}")

        base_tokenizer = AutoTokenizer.from_pretrained(args.compare_base, trust_remote_code=True)
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(args.compare_base, **load_kwargs)

        base_ppl = compute_perplexity(base_model, base_tokenizer, val_texts, max_length=args.max_seq_length)
        print(f"Base model perplexity:      {base_ppl:.2f}")
        print(f"Fine-tuned perplexity:      {ppl:.2f}")
        print(f"Improvement:                {base_ppl - ppl:.2f} ({100*(base_ppl-ppl)/base_ppl:.1f}%)")

        print("\nBase model generations:")
        base_gens = generate_samples(
            base_model, base_tokenizer, test_prompts[:3],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        for prompt, gen in zip(test_prompts[:3], base_gens):
            print(f"\n--- Prompt ---\n{prompt}")
            print(f"--- Base Generated ---\n{gen}")

    print(f"\n{'='*60}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
