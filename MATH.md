# The math behind gpt-from-scratch

Derivations of the load-bearing formulas in plain unicode notation — no LaTeX. Each section names the module that implements it. Notation: T = sequence length, d = model dimension, d_h = head dimension (= d / n_head), V = vocabulary size, ln = natural log, exp = e^x.

## 1. Scaled dot-product attention (`model.py`)

For queries Q ∈ R^(T×d_h), keys K ∈ R^(S×d_h), and values V ∈ R^(S×d_v):

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_h + M) · V

Each row of Q Kᵀ holds the dot products of one query with every key ("logits"); softmax turns each row into an attention distribution over positions, and the output is the probability-weighted sum of value vectors.

**Why divide by √d_h?** Assume entries of q and k are independent with mean 0 and variance 1. A logit z = q·k = Σᵢ₌₁^d_h qᵢkᵢ then has

    E[z]   = 0
    Var(z) = Σᵢ Var(qᵢkᵢ) = d_h        (each product qᵢkᵢ has variance 1)

so logits have standard deviation √d_h. Left unscaled, they grow with head width, pushing softmax toward its saturated regime where it outputs near-one-hot distributions and gradients vanish (∂softmax/∂z → 0). Dividing by √d_h restores unit logit variance, so entropy — and therefore gradient flow through the softmax — is independent of head dimension.

**Causal masking.** Token i must not attend to future tokens j > i. This is enforced *before* the softmax by adding a mask M with M[i,j] = −∞ for j > i and 0 elsewhere:

    exp(−∞) = 0, so masked positions get exactly zero attention weight,
    and every row still sums to 1 after softmax.

During training, query length equals key length (T = S), so the fused `is_causal=True` path of `F.scaled_dot_product_attention` applies. During cached decoding the queries cover only new tokens while keys include the cache, so `CausalSelfAttention.forward` builds an explicit lower-triangular boolean mask offset by the cached length instead; SDPA turns its False entries into −∞ logits before its internal softmax.

## 2. Cross-entropy loss and its gradient (`model.py`, `GPT.forward`)

With logits z ∈ R^V and target class y, define p = softmax(z):

    pᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}

The training loss for one position is the negative log-likelihood of the correct next token:

    L(z, y) = −log p_y = −z_y + log Σⱼ e^{zⱼ}

Differentiating (the log-sum-exp term contributes ∂/∂zᵢ = pᵢ, the −z_y term contributes −𝟙[i = y]):

    ∂L/∂zᵢ = pᵢ − 𝟙[i = y]

i.e. **the gradient is the predicted probability minus the one-hot target** — zero exactly when the model puts all its mass on y. Language modeling shifts this one step: position t's logits are trained against token t+1 as the target, implemented as `F.cross_entropy(logits.view(-1, V), targets.view(-1))` on shifted input/target pairs.

## 3. Perplexity and bits per character (`evaluate.py`)

Over N held-out tokens with model probabilities p(tokenᵢ | contextᵢ), the mean negative log-likelihood is

    L = (1/N) · Σᵢ −ln p(tokenᵢ | contextᵢ)          [in nats]

**Perplexity** exponentiates it:

    PPL = e^L

which reads as the effective number of choices the model is torn between per token.

**Bits per character.** One nat equals 1/ln(2) bits, and characters-per-token c = total chars / total tokens (c = 1 for the char tokenizer). Then

    bpc = L / ln(2) / c

`evaluate.perplexity` returns exp(mean batch cross-entropy), and `bits_per_character(loss_nats, chars_per_token)` implements the formula above directly.

## 4. AdamW with decoupled weight decay (`model.py`, `configure_optimizers`)

Adam keeps exponential moving averages of the gradient gₜ and its square:

    mₜ = β₁·mₜ₋₁ + (1 − β₁)·gₜ            m̂ = mₜ / (1 − β₁ᵗ)
    vₜ = β₂·vₜ₋₁ + (1 − β₂)·gₜ²           v̂ = vₜ / (1 − β₂ᵗ)

(the hat forms are bias corrections for the zero-initialized moments).

**Adam + L2 couples decay to adaptivity**: adding λθ to the gradient means the decay term flows through both moment estimates and ends up scaled by 1/(√v̂ + ε), so parameters with large gradient history are decayed *less* — no longer a plain weight shrinkage.

**AdamW decouples it** by applying decay outside the moment machinery, directly to the parameter:

    θ ← θ − lr · ( m̂ / (√v̂ + ε) + λ·θ )

The effective decay per step is now lr·λ regardless of each parameter's gradient statistics. This repo uses `torch.optim.AdamW`, which implements exactly this update, with β₁ = 0.9, β₂ = 0.95.

**Decay grouping.** `configure_optimizers` splits parameters into

- decay group: tensors with dim ≥ 2 (weight matrices, embeddings) → weight_decay = λ
- no-decay group: tensors with dim < 2 (biases, LayerNorm/RMSNorm gains) → weight_decay = 0

Shrinking weights toward zero regularizes; shrinking biases and normalization scales merely restricts their useful range without any regularization benefit, so those parameters are exempted.

## 5. Warmup + cosine learning-rate schedule (`schedule.py`, `get_lr`)

Given max_lr, min_lr, warmup length w, total iterations N, current step t:

    t ≤ w:            lr(t) = (t / w) · max_lr                       linear warmup
    w < t < N:        u = (t − w) / (N − w)
                      lr(t) = min_lr + ½ · (1 + cos(π·u)) · (max_lr − min_lr)
    t ≥ N:            lr(t) = min_lr                                 floor

The cosine branch starts at lr = max_lr (u = 0, cos 0 = 1), passes the midpoint min_lr + ½(max_lr − min_lr) at u = ½, and lands exactly on min_lr at u = 1. Linear warmup avoids destructive updates while the Adam moments are still cold; cosine decay anneals smoothly rather than dropping abruptly. Training defaults: `--lr-warmup-iters 100`, `min_lr = --lr × --lr-min-ratio` (ratio 0.1).

## 6. Rotary position embeddings (`model.py`, `pos_encoding="rope"`)

RoPE encodes absolute position m by rotating each head vector of q or k; after rotation, a dot product depends only on the *relative* offset m − n between the two positions.

**Angles.** For a head of width d_h, pair index i = 0 … d_h/2 − 1:

    inv_freqᵢ = 10000^(−2i/d_h)          geometric wavelength spread
    Θ[m, i]   = m · inv_freqᵢ             angle for position m, pair i

The cos/sin tables concatenate (Θ, Θ) to full width d_h and are sliced `[t_past : t_past + t]` during cached decoding so rotations continue from the right absolute offset.

**Rotation.** Split a vector into two halves a = first half, b = second half, and define `rotate_half(a, b) = (−b, a)`. The applied transform is

    q′ = q ⊙ cos Θ_full + rotate_half(q) ⊙ sin Θ_full

which expands to a pairwise rotation of each half-pair:

    a′ = a·cos Θ − b·sin Θ
    b′ = b·cos Θ + a·sin Θ

and identically for k. Because R(mΘ)ᵀ R(nΘ) = R((n − m)Θ), the attention score ⟨R(mΘ)q, R(nΘ)k⟩ depends only on n − m. No learned position table exists in rope mode (`wpe` stays None), and the cos/sin buffers are non-persistent so state-dict keys stay identical to a learned-position checkpoint.

## 7. RMSNorm vs LayerNorm (`model.py`, `RMSNorm` / `_build_norm`)

Both normalize each feature vector x ∈ R^d and rescale by a learned gain γ:

    LayerNorm(x) = γ ⊙ (x − μ) / √(σ² + ε)      μ = mean(x), σ² = var(x)
    RMSNorm(x)   = γ ⊙ x / √( mean(x²) + ε )

RMSNorm drops the mean-centering and normalizes only by the root mean square. That removes two reductions' worth of centering work while behaving almost identically in practice, which is why modern stacks (LLaMA and successors) prefer it. Both use ε = 1e-5 here and appear pre-norm in every block (`ln1`, `ln2`) plus once before the LM head (`ln_f`); `norm_type` picks which one is constructed.

## 8. Sampling filters (`model.py`: `generate`, `_apply_top_p`, `_apply_min_p`)

All filters operate on temperature-scaled logits z̃ = z/T (small T sharpens toward greedy, T > 1 flattens), set excluded entries to −∞, and rely on the final softmax over survivors to renormalize. They compose in the order top-k → top-p → min-p.

- **top-k**: keep the k largest logits; everything else becomes −∞.
- **top-p (nucleus)**: sort probabilities descending with cumulative sums c₁ ≤ c₂ ≤ …. A token survives iff the mass strictly before it is below p:

      keep i  ⟺  cᵢ₋₁ < p     equivalently remove i ⟺ cᵢ − pᵢ ≥ p

  so the surviving set is the smallest descending prefix whose cumulative probability reaches p.
- **min-p**: compute full probabilities pᵢ = softmax(z̃)ᵢ and cut relative to the most confident token:

      keep i  ⟺  pᵢ ≥ min_p · maxⱼ pⱼ

  Unlike top-k (fixed count) or top-p (fixed mass), the cutoff scales with how confident the model currently is: near-uniform distributions keep many tokens, confident ones keep few.

One token is then drawn from the renormalized distribution with `torch.multinomial`. In `generate`, top_k/top_p/min_p default to None (disabled); the same knobs surface as `--top-k`, `--top-p`, `--min-p` on the generation CLI.
