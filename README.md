# Tiny Diffusion-Style LLM Starter (with Unsloth plan)

This repo now includes:

- `unsloth_small_llm_plan.py`: checks whether `unsloth` is installed and prints a practical small-model training plan.
- `train_tiny_diffusion_lm.py`: a dependency-free demo of **iterative denoising generation** (diffusion-style behavior) using your requested data recipe (dictionary + poetry + little corpus).

## Why this setup

In this execution environment, `unsloth` is not installable/importable, so a direct Unsloth run cannot be executed here. The fallback script still gives you a runnable miniature denoising model and concrete output.

## Suggested data recipe

Use a blended dataset:

1. **Dictionary data (40%)**
   - Format each sample as: `Word: ...\nDefinition: ...\nExample: ...`
2. **Poetry data (40%)**
   - Line continuation, stanza completion, style transfer prompts.
3. **Little corpus (20%)**
   - Domain-specific short paragraphs and QA pairs.

## Run

```bash
python unsloth_small_llm_plan.py
python train_tiny_diffusion_lm.py
```

## Expected result

- The Unsloth check script prints installation status + recommended hyperparameters.
- The tiny diffusion script prints a denoising trajectory across multiple iterative steps and a final generated sequence.
