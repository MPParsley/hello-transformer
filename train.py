#!/usr/bin/env python3
"""Hello World Transformer — training + weight export.

Usage:
    python train.py                    # train with defaults
    python train.py --steps 3000       # more training steps
    python train.py --out docs/model/weights.json
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from model import HelloTransformer

# ---------------------------------------------------------------------------
# Training corpus
# ---------------------------------------------------------------------------
TRAINING_TEXT = """\
Hello, World! This is a Hello World Transformer.
The transformer is a neural network architecture.
It uses self-attention to model sequences of tokens.
Hello, World! Attention is all you need.
The quick brown fox jumps over the lazy dog.
Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump!
Transformers power large language models today.
Hello, World! This is a character-level language model.
It learns to predict the next character given the context.
The model has two transformer layers and four attention heads.
Each attention head focuses on different parts of the input.
Hello, World! Hello, World! Hello, World!
Feed-forward networks add non-linearity after each attention block.
Layer normalisation stabilises training by scaling activations.
Residual connections let gradients flow through deep networks.
Positional encodings tell the model where each token is located.
Hello, World! The model is small but the ideas are universal.
"""


def build_vocab(text: str) -> tuple[list[str], dict[str, int]]:
    vocab = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    return vocab, char_to_idx


def get_batch(tokens: list[int], max_len: int, batch_size: int, device: str):
    max_start = len(tokens) - max_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([torch.tensor(tokens[s : s + max_len]) for s in starts])
    y = torch.stack([torch.tensor(tokens[s + 1 : s + max_len + 1]) for s in starts])
    return x.to(device), y.to(device)


def export_weights(
    model: HelloTransformer,
    vocab: list[str],
    config: dict,
    path: str,
    loss: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    params = {name: param.data.tolist() for name, param in model.named_parameters()}
    payload = {
        "config": config,
        "vocab": vocab,
        "params": params,
        "trained": True,
        "final_loss": round(loss, 4),
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"Weights exported → {path}  ({size_kb:.0f} KB)")


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    vocab, char_to_idx = build_vocab(TRAINING_TEXT)
    vocab_size = len(vocab)
    tokens = [char_to_idx[c] for c in TRAINING_TEXT]
    print(f"Vocab size: {vocab_size}  |  Corpus: {len(tokens)} chars")

    config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "max_len": args.max_len,
    }

    model = HelloTransformer(**config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr / 10
    )

    model.train()
    t0 = time.time()
    loss_val = float("inf")

    for step in range(1, args.steps + 1):
        x, y = get_batch(tokens, args.max_len, args.batch_size, device)
        logits = model(x)                                     # (B, T, V)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if step % 200 == 0 or step == 1:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(
                f"step {step:>5}/{args.steps}  "
                f"loss={loss_val:.4f}  "
                f"lr={lr:.5f}  "
                f"elapsed={elapsed:.1f}s"
            )

    # Quick demo generation
    prompt = "Hello, World"
    prompt_ids = [char_to_idx.get(c, 0) for c in prompt]
    generated_ids = model.generate(prompt_ids, max_new_tokens=80, temperature=0.8)
    generated = "".join(vocab[i] for i in generated_ids)
    print(f"\nSample generation:\n  {generated!r}\n")

    export_weights(model, vocab, config, args.out, loss_val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Hello World Transformer")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--out", type=str, default="docs/model/weights.json")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
