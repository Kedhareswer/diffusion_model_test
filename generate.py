#!/usr/bin/env python3
"""Interactive generation with a fine-tuned model.

Usage:
    # Single prompt
    python generate.py --model-path output/final --prompt "Word: luminous\nDefinition:"

    # Interactive mode
    python generate.py --model-path output/final --interactive

    # Batch mode from file
    python generate.py --model-path output/final --prompts-file prompts.txt
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str):
    """Load model and tokenizer, handling both PEFT and merged models."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    try:
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, **load_kwargs
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        print(f"Loaded PEFT model (base: {config.base_model_name_or_path})")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        print(f"Loaded model from {model_path}")

    model.eval()
    return model, tokenizer


def generate(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate text from a prompt."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated


def interactive_mode(model, tokenizer, args):
    """Run interactive generation loop."""
    print("\n=== Interactive Generation ===")
    print("Type your prompt and press Enter. Use '\\n' for newlines.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if not prompt:
            continue

        # Allow \n in input to represent actual newlines
        prompt = prompt.replace("\\n", "\n")

        output = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"\n{output}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate text with fine-tuned LLM")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to generate from")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="File with one prompt per line")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=150,
                        help="Max tokens to generate (default: 150)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p nucleus sampling (default: 0.9)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (default: 50)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty (default: 1.1)")
    args = parser.parse_args()

    if not args.prompt and not args.prompts_file and not args.interactive:
        print("Provide --prompt, --prompts-file, or --interactive")
        sys.exit(1)

    model, tokenizer = load_model(args.model_path)

    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.prompt:
        prompt = args.prompt.replace("\\n", "\n")
        output = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(output)
    elif args.prompts_file:
        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            print(f"File not found: {prompts_path}")
            sys.exit(1)
        prompts = [line.strip().replace("\\n", "\n")
                    for line in prompts_path.read_text().splitlines() if line.strip()]
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"Prompt {i+1}: {prompt[:80]}...")
            print(f"{'='*60}")
            output = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            print(output)


if __name__ == "__main__":
    main()
