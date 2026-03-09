#!/usr/bin/env python3
"""Environment check + actionable Unsloth training plan."""

from textwrap import dedent

try:
    import unsloth  # type: ignore
    ok = True
except Exception as e:
    ok = False
    err = repr(e)

print("=== Unsloth setup check ===")
if ok:
    print("Unsloth import: OK")
else:
    print(f"Unsloth import: FAILED -> {err}")

print("\n=== Recommended small-model plan ===")
print(dedent("""
1) Data mix (your idea is good):
   - 40% dictionary entries (word + short definition)
   - 40% poetry lines/stanzas
   - 20% little corpus (domain text)

2) Suggested small base model with Unsloth:
   - unsloth/Qwen2.5-0.5B-bnb-4bit
   - LoRA rank 8 or 16
   - max_seq_length 256
   - 1 epoch to start

3) Formatting examples (instruction style):
   {"text": "Word: ember\\nDefinition: A small live piece of coal or wood in a dying fire."}
   {"text": "Continue the poem: Night folds velvet skies ..."}

4) Training hyperparameters:
   - learning_rate: 2e-4
   - per_device_train_batch_size: 2
   - gradient_accumulation_steps: 8
   - warmup_steps: 10
   - max_steps: 60 (smoke), then scale

5) Validation:
   - Keep 10% held-out samples
   - Evaluate next-token perplexity and manual generations
"""))
