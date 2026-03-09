#!/usr/bin/env python3
"""Evaluate and generate from a locally-trained transformer model.

Usage:
    python evaluate_local.py --model-path output/final_model.pt
    python evaluate_local.py --model-path output/best_model.pt --interactive
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from train_local import SmallTransformerLM, CharTokenizer, TextDataset, load_jsonl
from torch.utils.data import DataLoader


def load_model(model_path: str, device: str):
    """Load model, tokenizer, and config from output directory."""
    model_dir = Path(model_path).parent

    # Load config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = CharTokenizer()
    tokenizer.load(str(model_dir / "tokenizer.json"))

    # Build model
    model = SmallTransformerLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config.get("dropout", 0.1),
        max_len=config.get("max_seq_length", 256),
    ).to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"  Vocab size: {config['vocab_size']}")
    print(f"  d_model: {config['d_model']}, layers: {config['num_layers']}, heads: {config['nhead']}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    return model, tokenizer, config


def compute_perplexity(model, dataloader, pad_token_id: int, device: str) -> float:
    """Compute perplexity on a dataset."""
    import torch.nn.functional as F
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=pad_token_id,
                reduction="sum",
            )
            # Count non-pad tokens
            non_pad = (targets != pad_token_id).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20))


def generate_text(model, tokenizer, prompt: str, device: str,
                  max_new_tokens: int = 150, temperature: float = 0.8, top_k: int = 40) -> str:
    """Generate text from a prompt."""
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special=True)], dtype=torch.long
    ).to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0].tolist(), skip_special=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate locally-trained LM")
    parser.add_argument("--model-path", type=str, default="output/final_model.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_model(args.model_path, device)

    # -----------------------------------------------------------------------
    # Perplexity
    # -----------------------------------------------------------------------
    val_path = Path(args.data_dir) / "val.jsonl"
    if val_path.exists():
        val_data = load_jsonl(str(val_path))
        val_texts = [s["text"] for s in val_data]

        val_dataset = TextDataset(val_texts, tokenizer, args.max_seq_length)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        ppl = compute_perplexity(model, val_loader, tokenizer.pad_token_id, device)
        print(f"\nValidation perplexity: {ppl:.2f}")

        # Per-source
        sources = {}
        for s in val_data:
            src = s.get("source", "unknown")
            if src not in sources:
                sources[src] = []
            sources[src].append(s["text"])

        if len(sources) > 1:
            print("Per-source perplexity:")
            for src, texts in sorted(sources.items()):
                ds = TextDataset(texts, tokenizer, args.max_seq_length)
                dl = DataLoader(ds, batch_size=args.batch_size)
                src_ppl = compute_perplexity(model, dl, tokenizer.pad_token_id, device)
                print(f"  {src}: {src_ppl:.2f} ({len(texts)} samples)")

    # -----------------------------------------------------------------------
    # Sample generations
    # -----------------------------------------------------------------------
    if args.prompt:
        prompt = args.prompt.replace("\\n", "\n")
        output = generate_text(model, tokenizer, prompt, device,
                               args.max_new_tokens, args.temperature, args.top_k)
        print(f"\n{output}")
        return

    if args.interactive:
        print("\n=== Interactive Mode ===")
        print("Type prompt, press Enter. Use \\n for newlines. 'quit' to exit.\n")
        while True:
            try:
                prompt = input("Prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            prompt = prompt.replace("\\n", "\n")
            output = generate_text(model, tokenizer, prompt, device,
                                   args.max_new_tokens, args.temperature, args.top_k)
            print(f"\n{output}\n")
        return

    # Default: run standard test prompts
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

    for prompt in test_prompts:
        output = generate_text(model, tokenizer, prompt, device,
                               args.max_new_tokens, args.temperature, args.top_k)
        print(f"\n--- Prompt ---\n{prompt}")
        print(f"--- Generated ---\n{output}")


if __name__ == "__main__":
    main()
