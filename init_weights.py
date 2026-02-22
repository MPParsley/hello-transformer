#!/usr/bin/env python3
"""Generate placeholder weights.json using only stdlib (no PyTorch).

This creates correctly-shaped Xavier-initialised weights so the browser demo
works immediately, before the first GitHub Actions training run.  The
`trained` flag is False so the UI shows a warning badge.
"""
import json
import math
import os
import random

# ── Config (must match train.py defaults) ───────────────────────────────────
D_MODEL  = 64
N_HEADS  = 4
N_LAYERS = 2
D_FF     = 128
MAX_LEN  = 64
SEED     = 42

# Build vocabulary from the same training text as train.py
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

vocab = sorted(set(TRAINING_TEXT))
VOCAB_SIZE = len(vocab)

# ── Helpers ──────────────────────────────────────────────────────────────────
random.seed(SEED)

def normal(std: float) -> float:
    return random.gauss(0.0, std)

def xavier_vec(fan_in: int, fan_out: int, size: int) -> list:
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [normal(std) for _ in range(size)]

def xavier_mat(fan_in: int, fan_out: int, rows: int, cols: int) -> list:
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [[normal(std) for _ in range(cols)] for _ in range(rows)]

def ones(n: int) -> list:   return [1.0] * n
def zeros(n: int) -> list:  return [0.0] * n

# ── Build params ─────────────────────────────────────────────────────────────
params = {}

# Embeddings
params["embed.weight"]     = xavier_mat(VOCAB_SIZE, D_MODEL, VOCAB_SIZE, D_MODEL)
params["pos_embed.weight"] = xavier_mat(MAX_LEN, D_MODEL, MAX_LEN, D_MODEL)

# Transformer blocks
for i in range(N_LAYERS):
    pfx = f"blocks.{i}"
    params[f"{pfx}.attn.qkv.weight"]      = xavier_mat(D_MODEL, 3 * D_MODEL, 3 * D_MODEL, D_MODEL)
    params[f"{pfx}.attn.out_proj.weight"] = xavier_mat(D_MODEL, D_MODEL, D_MODEL, D_MODEL)
    params[f"{pfx}.ln1.weight"]           = ones(D_MODEL)
    params[f"{pfx}.ln1.bias"]             = zeros(D_MODEL)
    params[f"{pfx}.ln2.weight"]           = ones(D_MODEL)
    params[f"{pfx}.ln2.bias"]             = zeros(D_MODEL)
    params[f"{pfx}.ff1.weight"]           = xavier_mat(D_MODEL, D_FF, D_FF, D_MODEL)
    params[f"{pfx}.ff1.bias"]             = zeros(D_FF)
    params[f"{pfx}.ff2.weight"]           = xavier_mat(D_FF, D_MODEL, D_MODEL, D_FF)
    params[f"{pfx}.ff2.bias"]             = zeros(D_MODEL)

# Final layer norm + head
params["ln_f.weight"] = ones(D_MODEL)
params["ln_f.bias"]   = zeros(D_MODEL)
params["head.weight"] = xavier_mat(D_MODEL, VOCAB_SIZE, VOCAB_SIZE, D_MODEL)

# ── Write JSON ───────────────────────────────────────────────────────────────
payload = {
    "config": {
        "vocab_size": VOCAB_SIZE,
        "d_model":    D_MODEL,
        "n_heads":    N_HEADS,
        "n_layers":   N_LAYERS,
        "d_ff":       D_FF,
        "max_len":    MAX_LEN,
    },
    "vocab":       vocab,
    "params":      params,
    "trained":     False,
    "final_loss":  None,
}

out_path = "docs/model/weights.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(payload, f, separators=(",", ":"))

size_kb = os.path.getsize(out_path) / 1024
print(f"Wrote {out_path}  ({size_kb:.0f} KB)  vocab={VOCAB_SIZE}  trained=False")
