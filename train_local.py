#!/usr/bin/env python3
"""Train a small transformer language model from scratch on local data.

No internet or pre-trained model downloads required.

Usage:
    python prepare_data.py
    python train_local.py --epochs 5 --max-steps 200
    python train_local.py --epochs 20           # full training
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Tokenizer: character-level + common subwords for simplicity
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Simple character-level tokenizer with special tokens."""

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]

    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        self.vocab_size = 0
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def build_vocab(self, texts: list[str]):
        """Build vocabulary from a list of texts."""
        chars = set()
        for text in texts:
            chars.update(text)
        chars = sorted(chars)

        self.char2id = {}
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            self.char2id[tok] = i
        for i, ch in enumerate(chars, len(self.SPECIAL_TOKENS)):
            self.char2id[ch] = i

        self.id2char = {v: k for k, v in self.char2id.items()}
        self.vocab_size = len(self.char2id)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = []
        if add_special:
            ids.append(self.bos_token_id)
        for ch in text:
            ids.append(self.char2id.get(ch, self.unk_token_id))
        if add_special:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        chars = []
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for id_ in ids:
            if skip_special and id_ in special_ids:
                continue
            chars.append(self.id2char.get(id_, "?"))
        return "".join(chars)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"char2id": self.char2id}, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.char2id = data["char2id"]
        self.id2char = {int(v): k for k, v in self.char2id.items()}
        self.vocab_size = len(self.char2id)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: CharTokenizer, max_length: int):
        self.samples = []
        for text in texts:
            ids = tokenizer.encode(text)
            if len(ids) > max_length:
                ids = ids[:max_length - 1] + [tokenizer.eos_token_id]
            # Pad
            padding = [tokenizer.pad_token_id] * (max_length - len(ids))
            ids = ids + padding
            self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return ids


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SmallTransformerLM(nn.Module):
    """A small decoder-only transformer language model."""

    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Tie embedding weights
        self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        input_ids: (batch, seq_len)
        Returns: logits (batch, seq_len, vocab_size)
        """
        seq_len = input_ids.size(1)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        # Embedding + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)

        # Decoder (self-attention only, using memory=x as dummy)
        # We use nn.TransformerDecoder with tgt=x, memory=dummy
        memory = torch.zeros(input_ids.size(0), 1, self.d_model, device=input_ids.device)
        x = self.transformer(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
        )

        logits = self.output_proj(x)
        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 40, eos_token_id: int = 2):
        """Autoregressive generation."""
        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(generated)
                next_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    values, _ = torch.topk(next_logits, top_k)
                    min_val = values[:, -1].unsqueeze(-1)
                    next_logits = torch.where(
                        next_logits < min_val,
                        torch.full_like(next_logits, float("-inf")),
                        next_logits,
                    )

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)

                if next_token.item() == eos_token_id:
                    break

        return generated


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def compute_loss(model, batch, pad_token_id: int):
    """Compute cross-entropy loss for next-token prediction."""
    input_ids = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(input_ids)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
    )
    return loss


def evaluate_model(model, dataloader, pad_token_id: int, device: str) -> float:
    """Compute average loss on a dataset."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            loss = compute_loss(model, batch, pad_token_id)
            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Train small transformer LM from scratch")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension (default: 256)")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers (default: 4)")
    parser.add_argument("--dim-ff", type=int, default=512, help="Feed-forward dim (default: 512)")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Max sequence length (default: 256)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs (default: 10)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max steps (overrides epochs if > 0)")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps (default: 20)")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps (default: 10)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout (default: 0.1)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    train_texts = [s["text"] for s in train_data]
    val_texts = [s["text"] for s in val_data]

    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples:   {len(val_texts)}")

    # -----------------------------------------------------------------------
    # 2. Build tokenizer
    # -----------------------------------------------------------------------
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(train_texts + val_texts)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # -----------------------------------------------------------------------
    # 3. Build datasets
    # -----------------------------------------------------------------------
    train_dataset = TextDataset(train_texts, tokenizer, args.max_seq_length)
    val_dataset = TextDataset(val_texts, tokenizer, args.max_seq_length) if val_texts else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size) if val_dataset else None

    # -----------------------------------------------------------------------
    # 4. Build model
    # -----------------------------------------------------------------------
    model = SmallTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        max_len=args.max_seq_length,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # -----------------------------------------------------------------------
    # 5. Optimizer + scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Linear warmup then cosine decay
    total_steps = args.max_steps if args.max_steps > 0 else len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -----------------------------------------------------------------------
    # 6. Training loop
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training for {total_steps} steps")
    print(f"{'='*60}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs if args.max_steps <= 0 else 9999):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            batch = batch.to(device)
            loss = compute_loss(model, batch, tokenizer.pad_token_id)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                ppl = math.exp(min(avg_loss, 20))
                print(f"Step {global_step:>4d} | Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | LR: {lr:.2e} | {elapsed:.1f}s")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        # End of epoch
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        train_ppl = math.exp(min(avg_train_loss, 20))

        log = f"Epoch {epoch+1:>2d} | Train Loss: {avg_train_loss:.4f} | Train PPL: {train_ppl:.1f}"

        if val_loader:
            val_loss = evaluate_model(model, val_loader, tokenizer.pad_token_id, device)
            val_ppl = math.exp(min(val_loss, 20))
            log += f" | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": val_loss,
                    "args": vars(args),
                }, str(output_dir / "best_model.pt"))
                log += " *best*"

        print(log)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # -----------------------------------------------------------------------
    # 7. Save final model + tokenizer
    # -----------------------------------------------------------------------
    final_path = output_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "vocab_size": tokenizer.vocab_size,
        "global_step": global_step,
    }, str(final_path))

    tokenizer.save(str(output_dir / "tokenizer.json"))

    # Save model config for easy reloading
    config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_ff,
        "max_seq_length": args.max_seq_length,
        "dropout": args.dropout,
    }
    with open(str(output_dir / "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.1f}s ({global_step} steps)")
    print(f"Model saved to {final_path}")
    print(f"Tokenizer saved to {output_dir / 'tokenizer.json'}")

    # -----------------------------------------------------------------------
    # 8. Sample generations
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Sample Generations")
    print(f"{'='*60}")

    test_prompts = [
        "Word: cascade\nDefinition:",
        "Continue the poem:\nThe morning light",
        "Write a haiku about",
        "The key to training",
    ]

    model.eval()
    for prompt in test_prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special=True)], dtype=torch.long).to(device)
        output_ids = model.generate(
            input_ids, max_new_tokens=120, temperature=0.8, top_k=40,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output_ids[0].tolist(), skip_special=True)
        print(f"\n--- Prompt: {prompt[:50]}...")
        print(f"--- Output: {text[:200]}")

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
