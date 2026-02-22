/**
 * Hello World Transformer — browser inference engine.
 *
 * Pure-JS implementation of the same architecture defined in model.py.
 * Weights are loaded from /model/weights.json (exported by train.py).
 */

class HelloTransformer {
  constructor(weights) {
    this.config = weights.config;
    this.trained = weights.trained ?? false;
    this.finalLoss = weights.final_loss ?? null;

    // Build character ↔ index maps
    this.vocab = weights.vocab;
    this.charToIdx = Object.fromEntries(weights.vocab.map((c, i) => [c, i]));
    this.idxToChar = weights.vocab;

    // Cache flattened params for speed
    this.p = weights.params;
  }

  // ── Linear algebra helpers ──────────────────────────────────────────────

  /** Matrix-vector product: W (m×n) · v (n) → (m) */
  _matvec(W, v) {
    const m = W.length, n = v.length;
    const out = new Float32Array(m);
    for (let i = 0; i < m; i++) {
      const row = W[i];
      let sum = 0;
      for (let j = 0; j < n; j++) sum += row[j] * v[j];
      out[i] = sum;
    }
    return out;
  }

  /** Element-wise addition of two same-length arrays. */
  _add(a, b) {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    return out;
  }

  /** Layer normalisation: (x − μ) / σ · γ + β */
  _layernorm(x, gamma, beta, eps = 1e-5) {
    const n = x.length;
    let mean = 0, variance = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (let i = 0; i < n; i++) variance += (x[i] - mean) ** 2;
    variance /= n;
    const std = Math.sqrt(variance + eps);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = ((x[i] - mean) / std) * gamma[i] + beta[i];
    return out;
  }

  /** Softmax over a flat array. Handles -Infinity (masked) correctly. */
  _softmax(x) {
    const maxX = Math.max(...x);
    const exp = x.map(xi => Math.exp(xi - maxX));
    const sumExp = exp.reduce((a, b) => a + b, 0);
    return exp.map(e => e / sumExp);
  }

  /** ReLU */
  _relu(x) {
    return x.map(xi => Math.max(0, xi));
  }

  // ── Core forward pass ───────────────────────────────────────────────────

  /**
   * Run the transformer forward pass on a sequence of token indices.
   * @param {number[]} tokenIds  Sequence of token indices (length T).
   * @returns {Float32Array[]}   Logits per position: T × vocabSize.
   */
  forward(tokenIds) {
    const T = tokenIds.length;
    const { d_model: D, n_heads: H, n_layers: L } = this.config;
    const headDim = D / H;
    const p = this.p;

    // ── Embedding + positional encoding ──────────────────────────────────
    let h = tokenIds.map((id, pos) =>
      this._add(p["embed.weight"][id], p["pos_embed.weight"][pos])
    ); // h: T × D  (regular Array of Float32Array-like)

    // ── Transformer layers ────────────────────────────────────────────────
    for (let l = 0; l < L; l++) {
      const pfx = `blocks.${l}`;

      // Pre-norm for attention
      const hn1 = h.map(x =>
        this._layernorm(x, p[`${pfx}.ln1.weight`], p[`${pfx}.ln1.bias`])
      );

      // QKV projection (3D × D weight matrix)
      const qkvW = p[`${pfx}.attn.qkv.weight`];
      const qkvProj = hn1.map(x => this._matvec(qkvW, x)); // T × 3D

      // Split Q / K / V
      const Q = qkvProj.map(x => x.slice(0, D));
      const K = qkvProj.map(x => x.slice(D, 2 * D));
      const V = qkvProj.map(x => x.slice(2 * D));

      // Multi-head causal self-attention
      const attnOut = Array.from({ length: T }, () => new Float32Array(D));
      const outW = p[`${pfx}.attn.out_proj.weight`];

      for (let head = 0; head < H; head++) {
        const s = head * headDim, e = s + headDim;
        const scale = Math.sqrt(headDim);

        // Compute raw scores (T × T) with causal mask
        const scores = Array.from({ length: T }, (_, i) =>
          Array.from({ length: T }, (__, j) => {
            if (j > i) return -Infinity;   // causal mask
            let dot = 0;
            for (let k = s; k < e; k++) dot += Q[i][k] * K[j][k];
            return dot / scale;
          })
        );

        // Softmax each row → attention weights
        const attnW = scores.map(row => this._softmax(row));

        // Weighted sum of V
        for (let i = 0; i < T; i++) {
          for (let k = 0; k < headDim; k++) {
            let acc = 0;
            for (let j = 0; j <= i; j++) acc += attnW[i][j] * V[j][s + k];
            attnOut[i][s + k] += acc;
          }
        }
      }

      // Output projection + residual
      const projected = attnOut.map(x => this._matvec(outW, x));
      h = h.map((ht, t) => this._add(ht, projected[t]));

      // Pre-norm for FFN
      const hn2 = h.map(x =>
        this._layernorm(x, p[`${pfx}.ln2.weight`], p[`${pfx}.ln2.bias`])
      );

      // FFN: ReLU(x W1 + b1) W2 + b2
      const ff1W = p[`${pfx}.ff1.weight`], ff1B = p[`${pfx}.ff1.bias`];
      const ff2W = p[`${pfx}.ff2.weight`], ff2B = p[`${pfx}.ff2.bias`];

      const ffOut = hn2.map(x => {
        const mid = this._relu(this._add(this._matvec(ff1W, x), ff1B));
        return this._add(this._matvec(ff2W, mid), ff2B);
      });

      h = h.map((ht, t) => this._add(ht, ffOut[t]));
    }

    // Final layer norm + output projection
    const headW = p["head.weight"];
    return h.map(x => {
      const normed = this._layernorm(x, p["ln_f.weight"], p["ln_f.bias"]);
      return this._matvec(headW, normed);
    }); // T × vocabSize
  }

  // ── Generation ──────────────────────────────────────────────────────────

  /**
   * Sample the next token index given softmax probabilities.
   */
  _sample(probs) {
    let r = Math.random();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return probs.length - 1;
  }

  /**
   * Encode a prompt string → token ids (unknown chars replaced with space).
   */
  encode(text) {
    return [...text].map(c => this.charToIdx[c] ?? this.charToIdx[" "] ?? 0);
  }

  /**
   * Decode token ids → string.
   */
  decode(ids) {
    return ids.map(i => this.idxToChar[i] ?? "?").join("");
  }

  /**
   * Generate text autoregressively, yielding one character at a time.
   * Use as an async generator so the UI can stream output.
   *
   * @param {string}   prompt
   * @param {number}   maxNewTokens
   * @param {number}   temperature   >1 = more random, <1 = more focused
   * @yields {string}  One newly generated character per iteration.
   */
  async *generateStream(prompt, maxNewTokens = 200, temperature = 0.8) {
    const maxLen = this.config.max_len;
    const vocabSize = this.config.vocab_size;
    const ids = this.encode(prompt);

    for (let step = 0; step < maxNewTokens; step++) {
      const context = ids.slice(-maxLen);
      const logits = this.forward(context);
      const lastLogits = logits[logits.length - 1]; // vocabSize

      // Temperature scaling + softmax
      const scaled = Array.from(lastLogits, x => x / temperature);
      const probs = this._softmax(scaled);

      const nextId = this._sample(probs);
      ids.push(nextId);
      yield this.idxToChar[nextId] ?? "?";

      // Yield control back to the browser event loop
      await new Promise(r => setTimeout(r, 0));
    }
  }

  /**
   * Compute per-position attention weights for a given layer and head.
   * Useful for visualisation.
   *
   * @returns {number[][]}  T×T matrix of attention probabilities.
   */
  attentionWeights(tokenIds, layer = 0, head = 0) {
    const T = tokenIds.length;
    const { d_model: D, n_heads: H } = this.config;
    const headDim = D / H;
    const p = this.p;

    // We only need one forward pass up to the requested layer.
    let h = tokenIds.map((id, pos) =>
      this._add(p["embed.weight"][id], p["pos_embed.weight"][pos])
    );

    for (let l = 0; l <= layer; l++) {
      const pfx = `blocks.${l}`;
      const hn1 = h.map(x =>
        this._layernorm(x, p[`${pfx}.ln1.weight`], p[`${pfx}.ln1.bias`])
      );
      const qkvW = p[`${pfx}.attn.qkv.weight`];
      const qkvProj = hn1.map(x => this._matvec(qkvW, x));
      const Q = qkvProj.map(x => x.slice(0, D));
      const K = qkvProj.map(x => x.slice(D, 2 * D));
      const V = qkvProj.map(x => x.slice(2 * D));

      const s = head * headDim, e = s + headDim;
      const scale = Math.sqrt(headDim);

      const attnOut = Array.from({ length: T }, () => new Float32Array(D));
      const outW = p[`${pfx}.attn.out_proj.weight`];

      // Capture attention weights for the target layer/head
      let capturedAttn = null;
      for (let hd = 0; hd < H; hd++) {
        const hs = hd * headDim;
        const scores = Array.from({ length: T }, (_, i) =>
          Array.from({ length: T }, (__, j) => {
            if (j > i) return -Infinity;
            let dot = 0;
            for (let k = hs; k < hs + headDim; k++) dot += Q[i][k] * K[j][k];
            return dot / scale;
          })
        );
        const attnW = scores.map(row => this._softmax(row));
        if (l === layer && hd === head) capturedAttn = attnW;

        for (let i = 0; i < T; i++) {
          for (let k = 0; k < headDim; k++) {
            let acc = 0;
            for (let j = 0; j <= i; j++) acc += attnW[i][j] * V[j][hs + k];
            attnOut[i][hs + k] += acc;
          }
        }
      }

      const projected = attnOut.map(x => this._matvec(outW, x));
      h = h.map((ht, t) => this._add(ht, projected[t]));

      if (l < layer) {
        const hn2 = h.map(x =>
          this._layernorm(x, p[`${pfx}.ln2.weight`], p[`${pfx}.ln2.bias`])
        );
        const ff1W = p[`${pfx}.ff1.weight`], ff1B = p[`${pfx}.ff1.bias`];
        const ff2W = p[`${pfx}.ff2.weight`], ff2B = p[`${pfx}.ff2.bias`];
        const ffOut = hn2.map(x => {
          const mid = this._relu(this._add(this._matvec(ff1W, x), ff1B));
          return this._add(this._matvec(ff2W, mid), ff2B);
        });
        h = h.map((ht, t) => this._add(ht, ffOut[t]));
      }

      if (l === layer) return capturedAttn;
    }
    return null;
  }
}

// ── Model loader ────────────────────────────────────────────────────────────

let _model = null;

async function loadModel(statusCallback) {
  if (_model) return _model;
  statusCallback("Loading weights…");
  const resp = await fetch("model/weights.json");
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const weights = await resp.json();
  _model = new HelloTransformer(weights);
  return _model;
}
