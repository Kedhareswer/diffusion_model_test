#!/usr/bin/env python3
"""Fine-tune a small LLM using Unsloth + LoRA on blended dictionary/poetry/corpus data.

Usage:
    # Prepare data first
    python prepare_data.py

    # Then train
    python train.py                          # defaults
    python train.py --base-model unsloth/Qwen2.5-0.5B-bnb-4bit --epochs 1
    python train.py --max-steps 60           # quick smoke test
"""

import argparse
import json
import os
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a small LLM with Unsloth + LoRA")
    parser.add_argument("--base-model", type=str, default="unsloth/Qwen2.5-0.5B-bnb-4bit",
                        help="Base model to fine-tune")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing train.jsonl and val.jsonl")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Max sequence length (default: 256)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps. Overrides epochs if > 0 (default: -1 = use epochs)")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps (default: 10)")
    parser.add_argument("--logging-steps", type=int, default=5, help="Log every N steps (default: 5)")
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps (default: 50)")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use fp16 training")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16 training (default)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    train_path = Path(args.data_dir) / "train.jsonl"
    val_path = Path(args.data_dir) / "val.jsonl"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run `python prepare_data.py` first.")
        return

    train_data = load_jsonl(str(train_path))
    val_data = load_jsonl(str(val_path)) if val_path.exists() else []

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples:   {len(val_data)}")

    # -----------------------------------------------------------------------
    # 2. Load model with Unsloth
    # -----------------------------------------------------------------------
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth is not installed.")
        print("Install with: pip install 'unsloth[colab-new]'")
        print("Or on Linux: pip install unsloth")
        print("\nFalling back to standard transformers + PEFT...")
        _train_with_peft(args, train_data, val_data)
        return

    print(f"\nLoading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # -----------------------------------------------------------------------
    # 3. Add LoRA adapters
    # -----------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # -----------------------------------------------------------------------
    # 4. Prepare datasets
    # -----------------------------------------------------------------------
    from datasets import Dataset

    def formatting_func(examples):
        """Format each sample as a single text field for causal LM training."""
        texts = []
        for text in examples["text"]:
            texts.append(text + tokenizer.eos_token)
        return {"text": texts}

    train_dataset = Dataset.from_list([{"text": s["text"]} for s in train_data])
    val_dataset = Dataset.from_list([{"text": s["text"]} for s in val_data]) if val_data else None

    train_dataset = train_dataset.map(formatting_func, batched=True, remove_columns=["text"])
    if val_dataset:
        val_dataset = val_dataset.map(formatting_func, batched=True, remove_columns=["text"])

    # -----------------------------------------------------------------------
    # 5. Configure trainer
    # -----------------------------------------------------------------------
    from trl import SFTTrainer
    from transformers import TrainingArguments

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=args.fp16,
        bf16=args.bf16 and not args.fp16,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.save_steps if val_dataset else None,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,
    )

    # -----------------------------------------------------------------------
    # 6. Train
    # -----------------------------------------------------------------------
    print("\n=== Starting training ===")
    trainer.train()

    # -----------------------------------------------------------------------
    # 7. Save
    # -----------------------------------------------------------------------
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}")

    # Also save as merged 16-bit for easy loading
    merged_path = output_dir / "merged_16bit"
    model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {merged_path}")

    print("\n=== Training complete ===")


def _train_with_peft(args, train_data: list[dict], val_data: list[dict]):
    """Fallback: train with standard transformers + PEFT (no Unsloth)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    # Map unsloth model names to HuggingFace equivalents
    model_name = args.base_model
    if model_name.startswith("unsloth/"):
        # Strip unsloth prefix and quantization suffix for HF loading
        hf_name = model_name.replace("unsloth/", "")
        for suffix in ["-bnb-4bit", "-bnb-8bit"]:
            hf_name = hf_name.replace(suffix, "")
        # Map to known HF repos
        hf_mapping = {
            "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
            "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
            "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
        }
        model_name = hf_mapping.get(hf_name, f"Qwen/{hf_name}")

    print(f"\n[PEFT fallback] Loading: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["device_map"] = "auto"
    else:
        print("WARNING: No GPU detected. Training will be slow on CPU.")
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    def tokenize(examples):
        return tokenizer(
            [t + tokenizer.eos_token for t in examples["text"]],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )

    train_dataset = Dataset.from_list([{"text": s["text"]} for s in train_data])
    val_dataset = Dataset.from_list([{"text": s["text"]} for s in val_data]) if val_data else None

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    if val_dataset:
        val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Trainer
    from trl import SFTTrainer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=args.bf16 and not args.fp16 and torch.cuda.is_available(),
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.save_steps if val_dataset else None,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,
    )

    print("\n=== Starting training (PEFT fallback) ===")
    trainer.train()

    # Save
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}")
    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
