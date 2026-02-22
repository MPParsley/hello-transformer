# Hello World Transformer

A minimal, fully-trainable **character-level transformer** that you can:

- **Train** on GitHub Actions (free, ~3 min, CPU-only)
- **Demo** instantly in your browser via GitHub Pages — no server needed

The model is small enough to understand end-to-end, but architecturally
identical to the transformers behind modern large language models.

---

## Quick start

### 1 — Enable GitHub Pages

In your repository settings → **Pages** → set Source to
**GitHub Actions**.

### 2 — Train the model

Go to **Actions → Train model → Run workflow**.
The workflow will:

1. Install PyTorch (CPU build)
2. Train the transformer for 2 000 steps (~3 min)
3. Export weights to `docs/model/weights.json` and commit them
4. Trigger the Pages deployment

### 3 — Open the demo

After the Pages workflow finishes, visit:

```
https://<your-username>.github.io/<your-repo>/
```

Type a prompt, click **Generate**, and watch the model stream characters.
The page also shows a live **attention heatmap** so you can see exactly what
the model attends to at each position.

---

## Train locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python train.py
```

Optional arguments:

| Flag | Default | Description |
|---|---|---|
| `--steps` | 2000 | Training steps |
| `--lr` | 1e-3 | Learning rate |
| `--batch-size` | 32 | Batch size |
| `--d-model` | 64 | Embedding / hidden dimension |
| `--n-heads` | 4 | Number of attention heads |
| `--n-layers` | 2 | Number of transformer blocks |
| `--d-ff` | 128 | Feed-forward hidden dimension |
| `--max-len` | 64 | Context length (characters) |
| `--out` | `docs/model/weights.json` | Output path |

---

## Architecture

```
Input text
    │
    ▼
Character embedding  (vocab_size × d_model)
    +
Positional embedding (max_len   × d_model)
    │
    ▼  ╔═══════════════════════════════╗
       ║  Transformer block  × 2       ║
       ║  ┌──────────────────────────┐ ║
       ║  │ LayerNorm                │ ║
       ║  │ Multi-head Self-Attention│ ║  4 heads, causal mask
       ║  │ + Residual               │ ║
       ║  ├──────────────────────────┤ ║
       ║  │ LayerNorm                │ ║
       ║  │ Feed-Forward  (ReLU)     │ ║  d_model → d_ff → d_model
       ║  │ + Residual               │ ║
       ║  └──────────────────────────┘ ║
       ╚═══════════════════════════════╝
    │
    ▼
Final LayerNorm
    │
    ▼
Linear projection → vocab logits
    │
    ▼
Softmax → next-character probabilities
```

**Default hyper-parameters**

| | |
|---|---|
| Vocab | ~42 unique characters (from training corpus) |
| d_model | 64 |
| Layers | 2 |
| Heads | 4 |
| d_ff | 128 |
| Context | 64 characters |
| Parameters | ~82 000 |

---

## How the browser demo works

The file `docs/inference.js` contains a **from-scratch JavaScript
implementation** of the exact same transformer defined in `model.py`.
No ML framework is needed in the browser — just plain arrays and math.

Key operations implemented in JS:

| Function | Description |
|---|---|
| `_matvec(W, v)` | Matrix–vector product |
| `_layernorm(x, γ, β)` | Layer normalisation |
| `_softmax(x)` | Numerically stable softmax |
| `forward(tokenIds)` | Full causal transformer pass |
| `generateStream(prompt)` | Async generator — streams chars to the UI |
| `attentionWeights(ids, layer, head)` | Returns T×T attention matrix for the heatmap |

Weights are loaded once from `model/weights.json` (~1.5 MB) and cached.

---

## Repository layout

```
hello-transformer/
├── model.py                   # PyTorch transformer definition
├── train.py                   # Training + weight export script
├── init_weights.py            # Generates placeholder weights (no PyTorch)
├── requirements.txt
├── .github/
│   └── workflows/
│       ├── train.yml          # Train on GitHub Actions + commit weights
│       └── pages.yml          # Deploy docs/ to GitHub Pages
└── docs/
    ├── index.html             # Browser demo UI
    ├── inference.js           # JavaScript transformer engine
    └── model/
        └── weights.json       # Trained weights (updated by CI)
```

---

## Understanding transformers

This project deliberately keeps the code small so you can read and modify
every line:

- **Self-attention** lets each position look at all previous positions.
- **Causal masking** prevents future tokens from being seen during training.
- **Multi-head attention** runs several attention patterns in parallel.
- **Residual connections** let gradients flow through deep stacks.
- **Layer normalisation** stabilises activations before each sub-layer.

Start with `model.py` (≈90 lines) and then `docs/inference.js` (the same
logic in JS) for a side-by-side comparison of Python/PyTorch vs plain JS.

---

## License

MIT
