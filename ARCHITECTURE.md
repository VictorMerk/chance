# Architecture

A walkthrough of one training forward pass through `GPT` (`src/gpt_from_scratch/model.py`) using the default configuration:

| Config | Value |
| --- | --- |
| `vocab_size` | 65 (tiny Shakespeare characters) |
| `block_size` | 256 |
| `n_layer` | 6 |
| `n_head` | 6 |
| `n_embd` | 384 |
| `dropout` | 0.1 |

Batch size during training is `B = 64` and every sequence is the full context length, so `T = 256`. Each head operates on `head_dim = n_embd / n_head = 64`, so attention scores are scaled by `1/sqrt(64) = 0.125`.

## Step by step

### 1. Token and positional embeddings

Input `idx` is a `(64, 256)` integer tensor of token IDs.

```
wte(idx)          -> (64, 256, 384)   # learned token embedding table, 65 x 384
wpe(arange(256))  -> (256, 384)       # learned position embedding table, 256 x 384
sum + dropout     -> (64, 256, 384)
```

The position IDs are always `arange(t_past, t_past + t)`. During plain training `t_past = 0`; during cached generation it equals the number of tokens already in the KV cache, which keeps positions correct when only the newest token is fed in.

### 2. Transformer block (repeated 6 times)

Each `Block` keeps a residual stream `x` of shape `(64, 256, 384)`:

**Attention sub-block**

```
LN1(x)                     -> (64, 256, 384)     # pre-norm
qkv projection             -> (64, 256, 1152)    # one fused Linear(384 -> 3*384), split into q, k, v
split into heads           -> q, k, v: (64, 6, 256, 64)   # view + transpose
SDPA causal attention      -> (64, 6, 256, 64)   # F.scaled_dot_product_attention, is_causal=True
merge heads                -> (64, 256, 384)     # transpose + contiguous reshape
output projection + dropout-> (64, 256, 384)     # Linear(384 -> 384)
x = x + attn_out           -> (64, 256, 384)     # residual add
```

Attention itself never materializes an explicit `(T, T)` score matrix on the fast path: `F.scaled_dot_product_attention(q, k, v, is_causal=True)` dispatches to PyTorch's Flash Attention / memory-efficient kernels. An explicit lower-triangular mask with diagonal offset `k_len - q_len` is built only for cached decoding steps where the query is shorter than the cached keys, because `is_causal` aligns its mask top-left, which would be wrong there.

**MLP sub-block**

```
LN2(x)                 -> (64, 256, 384)     # pre-norm
fc: Linear(384 -> 1536)-> (64, 256, 1536)    # 4x expansion
GELU                   -> (64, 256, 1536)
proj: Linear(1536 -> 384) -> (64, 256, 384)
dropout, residual add  -> x                  # new residual stream
```

### 3. Final LayerNorm and tied language-model head

```
ln_f(x)            -> (64, 256, 384)
lm_head            -> logits (64, 256, 65)   # Linear(384 -> 65, bias=False)
```

`lm_head.weight` **is** `wte.weight`: input and output embeddings are tied, so the logit for token `v` at position `t` is the dot product between the final hidden state and that token's input embedding vector.

### 4. Loss

Targets are the same sequences shifted by one. Cross-entropy flattens the batch:

```
logits.view(-1, 65)  -> (16384, 65)     # 64 * 256 rows
targets.view(-1)     -> (16384,)
F.cross_entropy(...) -> scalar          # mean over all 16,384 predictions
```

## Parameter count for the default model

Counting every weight and bias, with the tied `lm_head` contributing nothing extra:

| Component | Formula | Parameters |
| --- | --- | --- |
| `wte` (token embeddings) | 65 x 384 | 24,960 |
| `wpe` (position embeddings) | 256 x 384 | 98,304 |
| Per-layer LayerNorms (`ln1` + `ln2`) | 2 x (2 x 384) | 1,536 |
| Per-layer attention QKV | 384 x 1152 + 1152 | 443,520 |
| Per-layer attention output proj | 384 x 384 + 384 | 147,840 |
| Per-layer MLP fc | 384 x 1536 + 1536 | 591,360 |
| Per-layer MLP proj | 1536 x 384 + 384 | 590,208 |
| One block total | sum of the five rows above | 1,774,464 |
| Six blocks | 6 x 1,774,464 | 10,646,784 |
| `ln_f` | 2 x 384 | 768 |
| `lm_head` (tied to `wte`) | 0 | 0 |
| **Total** | | **10,770,816** |

So the default model has **10,770,816 parameters (~10.77 M)**. `model.num_parameters()` reports this number (shared tensors are counted once); `model.num_parameters(non_embedding=True)` subtracts only `wpe` and returns 10,672,512.

Initialization follows GPT-2: all weights drawn from `N(0, 0.02^2)` with zero biases, and the two residual-output projections per block (`attn.proj`, `mlp.proj`) scaled down to std `0.02 / sqrt(2 * n_layer)` so the variance of the residual stream does not grow with depth.

## Design choices

### Why pre-norm

LayerNorm is applied *before* each sub-block (`x = x + Attn(LN1(x))`), not after. In pre-norm architectures the residual stream is a clean, unnormalized path from embeddings to the output head, so gradients flow backward through addition alone without passing through a normalization at every depth. This makes deep stacks trainable with simple AdamW-style optimizers and removes the need for careful post-norm warmup tricks. It is the standard choice in GPT-2, GPT-3, and most modern transformers.

### Why weight tying

The output projection shares its matrix with the token embedding table. Effects at this scale:

- Saves `vocab_size * n_embd = 24,960` parameters and, more importantly, avoids learning a separate mapping in and out of embedding space.
- The model size here (~10 M) is small enough that the embedding tables are a meaningful fraction of capacity; tying acts as regularization and measurably helps small language models.
- Generation stays cheap: logits require one `(…, 384) @ (384, 65)` matmul, no extra table.

### Why AdamW decay grouping

`configure_optimizers` splits parameters into two groups:

- `p.dim() >= 2` (all matrices: attention QKV and output projections, MLP layers, embedding tables) get `weight_decay`.
- `p.dim() < 2` (biases and LayerNorm gains/biases) get `weight_decay = 0`.

Weight decay pulls weights toward zero and works as a regularizer on learned feature transforms. Applying it to biases or LayerNorm parameters is actively harmful: those exist to shift and rescale activations, and shrinking them toward zero destroys their function. The decay/no-decay split is the standard GPT-2 recipe, and a dedicated test asserts the grouping is correct.

### How the KV cache changes generation complexity

Without a cache, producing token `T+1` requires re-running the full model on all `T` previous tokens, and attention recomputes the entire `(T, T)` score matrix. Generating `N` tokens therefore costs `O(N^3)` attention compute overall (the sum of quadratic passes) even though only one token is new each step.

With the KV cache (`use_cache=True`, the default in `generate`):

- Each layer stores its computed `k` and `v` — per layer, two tensors of shape `(B, n_head, t_cached, head_dim)`.
- Each step processes only the newest token (`out[:, -1:]`): one query row attends against the cached `O(t)` keys/values.
- Total generation cost drops from cubic to `O(N^2)` in sequence length, and per-step cost becomes linear instead of quadratic.

Two details matter for correctness: position IDs are derived from `t_past` (cache length), not from zero, and once the cache reaches `block_size` the sliding window `out[:, -block_size:]` re-prefills the prompt window. A test verifies that cached and uncached generation produce identical outputs at `top_k=1`.

### How BPE relates to the char tokenizer

Training uses `CharTokenizer`: a fixed sorted vocabulary of the characters seen in the corpus (65 for tiny Shakespeare), encoding one ID per character. `BPETokenizer` starts from all 256 byte values and greedily learns merge rules that fuse frequent byte pairs into new IDs up to a requested vocab size; it supports `train` / `encode` / `decode` / `save` / `load`. From the model's point of view the two are interchangeable — both just map text to a list of integer IDs under some `vocab_size` — so switching tokenizer changes only `vocab_size` and the encode/decode step, not the architecture. BPE shortens sequences (more text per token, faster generation) at the cost of a larger embedding/output matrix.
