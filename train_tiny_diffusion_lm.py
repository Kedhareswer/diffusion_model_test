#!/usr/bin/env python3
"""Tiny diffusion-style language model demo (dependency-free).

This is a CPU-only educational fallback when Unsloth is unavailable.
It uses iterative masked-token denoising to mimic diffusion-style refinement.
"""

from __future__ import annotations
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

SEED = 7
random.seed(SEED)

DICTIONARY_WORDS = [
    "abide", "azure", "breeze", "brook", "candle", "dawn", "ember", "field",
    "glimmer", "hush", "ivy", "lantern", "meadow", "night", "oak", "petal",
    "quill", "river", "solace", "thistle", "umbra", "velvet", "willow", "zephyr",
]

POETRY_LINES = [
    "At dawn the willow keeps the river's secret.",
    "A lantern hums where meadow grasses sleep.",
    "Night folds velvet skies above the field.",
    "Petal and ember drift in patient breeze.",
    "Quill and candle map the hush of oak.",
]

LITTLE_CORPUS = [
    "The apprentice writes a short letter with a careful hand.",
    "A small model can still learn rhythm from tiny books.",
    "The village keeps a dictionary beside the old poems.",
    "Simple systems improve when examples are clean and focused.",
    "Iterative editing can polish rough text into clear phrases.",
]


def normalize(text: str) -> list[str]:
    out = []
    cur = []
    for ch in text.lower():
        if ch.isalpha() or ch == "'":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur.clear()
    if cur:
        out.append("".join(cur))
    return out


@dataclass
class TinyDenoiser:
    unigram: Counter
    left_right: defaultdict
    vocab: list[str]

    @classmethod
    def train(cls, sentences: list[list[str]]) -> "TinyDenoiser":
        unigram = Counter()
        left_right = defaultdict(Counter)
        for sent in sentences:
            padded = ["<bos>"] + sent + ["<eos>"]
            for i in range(1, len(padded) - 1):
                token = padded[i]
                left, right = padded[i - 1], padded[i + 1]
                unigram[token] += 1
                left_right[(left, right)][token] += 1
        vocab = sorted(unigram.keys())
        return cls(unigram=unigram, left_right=left_right, vocab=vocab)

    def fill_mask(self, left: str, right: str) -> str:
        pair_counter = self.left_right.get((left, right))
        if pair_counter:
            return pair_counter.most_common(1)[0][0]
        return self.unigram.most_common(1)[0][0]

    def diffuse_generate(self, prompt: str, length: int = 12, steps: int = 6) -> tuple[list[str], list[list[str]]]:
        prompt_tokens = normalize(prompt)
        tokens = prompt_tokens + ["<mask>"] * max(0, length - len(prompt_tokens))
        history = [tokens.copy()]

        for _ in range(steps):
            # Denoise all masked positions left->right using context.
            for i, tok in enumerate(tokens):
                if tok != "<mask>":
                    continue
                left = "<bos>" if i == 0 else tokens[i - 1]
                right = "<eos>" if i == len(tokens) - 1 else tokens[i + 1]
                if right == "<mask>":
                    right = "<eos>"
                tokens[i] = self.fill_mask(left, right)

            # Add small noising schedule except final round.
            if _ != steps - 1:
                noise_count = max(1, len(tokens) // 5)
                for idx in random.sample(range(len(tokens)), noise_count):
                    if idx < len(prompt_tokens):
                        continue
                    tokens[idx] = "<mask>"
            history.append(tokens.copy())

        return tokens, history


def build_training_data() -> list[list[str]]:
    combined = []
    for w in DICTIONARY_WORDS:
        combined.append([w])
    for line in POETRY_LINES + LITTLE_CORPUS:
        combined.append(normalize(line))
    return combined


def main() -> None:
    data = build_training_data()
    model = TinyDenoiser.train(data)

    final, history = model.diffuse_generate("dawn lantern", length=10, steps=5)

    print("=== Tiny Diffusion-Style LM Demo (fallback when Unsloth is unavailable) ===")
    print(f"Training sequences: {len(data)}")
    print(f"Vocabulary size: {len(model.vocab)}")
    print("Prompt: dawn lantern")
    print("\nDenoising trajectory:")
    for i, step_tokens in enumerate(history):
        print(f"step {i}:", " ".join(step_tokens))
    print("\nFinal output:")
    print(" ".join(final))


if __name__ == "__main__":
    main()
