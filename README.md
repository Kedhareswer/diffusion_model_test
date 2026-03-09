# Small LLM Fine-Tuning: Dictionary + Poetry + Corpus

Fine-tune a small language model (Qwen2.5-0.5B) using Unsloth + LoRA on a blended dataset of dictionary entries, poetry, and domain text.

## Project Structure

```
├── prepare_data.py              # Data curation: blend 40/40/20, split train/val
├── train.py                     # Fine-tuning with Unsloth + LoRA (PEFT fallback)
├── evaluate.py                  # Perplexity + sample generation evaluation
├── generate.py                  # Interactive/batch text generation
├── train_tiny_diffusion_lm.py   # CPU-only toy diffusion demo (original prototype)
├── unsloth_small_llm_plan.py    # Environment check + training plan reference
├── requirements.txt             # Python dependencies
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Unsloth requires a CUDA GPU. On CPU-only machines, the training script falls back to standard transformers + PEFT automatically.

### 2. Prepare data

```bash
python prepare_data.py --output-dir data
```

This creates `data/train.jsonl` and `data/val.jsonl` with the 40/40/20 blend:
- **40% dictionary** — word, definition, example sentence
- **40% poetry** — prompts with completions (haiku, couplets, quatrains, stanzas)
- **20% corpus** — paragraphs about NLP, training, and language

### 3. Train

```bash
# Smoke test (60 steps)
python train.py --max-steps 60

# Full training (3 epochs)
python train.py --epochs 3

# Custom config
python train.py \
    --base-model unsloth/Qwen2.5-0.5B-bnb-4bit \
    --lora-rank 16 \
    --lr 2e-4 \
    --batch-size 2 \
    --grad-accum 8 \
    --max-seq-length 256
```

The trained model is saved to `output/final/` (LoRA adapter) and `output/merged_16bit/` (merged weights).

### 4. Evaluate

```bash
# Perplexity + sample generations
python evaluate.py --model-path output/final --data-dir data

# Compare against base model
python evaluate.py --model-path output/final --compare-base Qwen/Qwen2.5-0.5B
```

### 5. Generate

```bash
# Single prompt
python generate.py --model-path output/final \
    --prompt "Word: luminous\nDefinition:"

# Interactive mode
python generate.py --model-path output/final --interactive

# Batch from file
python generate.py --model-path output/final --prompts-file prompts.txt
```

## Data Format

Each training sample is a single `text` field in JSONL:

```json
{"text": "Word: ember\nDefinition: A small live piece of coal or wood in a dying fire.\nExample: The last embers glowed softly in the fireplace.", "source": "dictionary"}
{"text": "Continue the poem:\nThe brook runs silver under morning light,\nWhile ivy climbs the walls of ancient stone.", "source": "poetry"}
{"text": "LoRA adapters add a small number of trainable parameters on top of a frozen base model.", "source": "corpus"}
```

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Base model | `unsloth/Qwen2.5-0.5B-bnb-4bit` | 4-bit quantized Qwen 0.5B |
| LoRA rank | 16 | Rank of low-rank adaptation |
| LoRA alpha | 32 | Scaling factor |
| Learning rate | 2e-4 | AdamW learning rate |
| Batch size | 2 | Per-device batch size |
| Gradient accumulation | 8 | Effective batch = 2 × 8 = 16 |
| Max sequence length | 256 | Truncation length |
| Warmup steps | 10 | Linear warmup |

## Legacy Demo

The original prototype scripts still work as standalone demos:

```bash
python unsloth_small_llm_plan.py   # Print training plan
python train_tiny_diffusion_lm.py  # Run toy denoiser on CPU
```
