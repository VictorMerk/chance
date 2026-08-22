from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

KVCache = list[tuple[torch.Tensor, torch.Tensor]]
GPTOutput = (
    tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, KVCache]
)


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    pos_encoding: str = "learned"  # "rope" swaps learned positions for rotary embeddings
    norm_type: str = "layernorm"  # "rmsnorm" replaces every LayerNorm
    mlp_type: str = "gelu"  # "swiglu" uses an LLaMA-style gated MLP
    tie_embeddings: bool = True
    pre_norm: bool = True  # False switches Blocks to original Transformer/GPT-1 post-norm

    def __post_init__(self) -> None:
        if self.pos_encoding not in ("learned", "rope"):
            raise ValueError(f"unknown pos_encoding: {self.pos_encoding!r}")
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError(f"unknown norm_type: {self.norm_type!r}")
        if self.mlp_type not in ("gelu", "swiglu"):
            raise ValueError(f"unknown mlp_type: {self.mlp_type!r}")

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GPTConfig:
        return cls(**d)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(rms + self.eps)


def _build_norm(config: GPTConfig) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd)
    return nn.LayerNorm(config.n_embd)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # None buffers keep state_dict keys identical to a learned-position model.
        self.rope_cos: torch.Tensor | None
        self.rope_sin: torch.Tensor | None
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        if config.pos_encoding == "rope":
            head_dim = config.n_embd // config.n_head
            inv_freq = 1.0 / (
                10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            angles = torch.outer(torch.arange(config.block_size, dtype=torch.float32), inv_freq)
            emb = torch.cat((angles, angles), dim=-1)
            self.register_buffer("rope_cos", emb.cos(), persistent=False)
            self.register_buffer("rope_sin", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = c // self.n_head
        q = q.view(b, t, self.n_head, head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, head_dim).transpose(1, 2)
        if self.rope_cos is not None:
            assert self.rope_sin is not None
            t_past = past_kv[0].size(2) if past_kv is not None else 0
            cos = self.rope_cos[t_past : t_past + t].unsqueeze(0).unsqueeze(0)
            sin = self.rope_sin[t_past : t_past + t].unsqueeze(0).unsqueeze(0)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        # is_causal aligns its mask top-left, which is wrong once q is shorter than k/v.
        if t == k.size(2):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            causal = torch.ones(t, k.size(2), dtype=torch.bool, device=x.device)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal.tril(diagonal=k.size(2) - t),
                dropout_p=dropout_p,
                is_causal=False,
            )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y)), (k, v)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc(x))
        return self.dropout(self.proj(x))


class SwiGLU(nn.Module):
    # hidden = 8/3 * n_embd rounded up to a multiple of 8 keeps the parameter count
    # close to the 4x gelu MLP despite having three projections instead of two.
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = (8 * config.n_embd // 3 + 7) // 8 * 8
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.pre_norm = config.pre_norm
        self.ln1 = _build_norm(config)
        self.attn = CausalSelfAttention(config)
        if config.pre_norm:
            self.ln2 = _build_norm(config)
        self.mlp: MLP | SwiGLU = MLP(config) if config.mlp_type == "gelu" else SwiGLU(config)

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not self.pre_norm:
            # Post-norm (original Transformer/GPT-1): a single norm after the MLP residual.
            h, present = self.attn(x, past_kv)
            x = x + h
            return self.ln1(x + self.mlp(x)), present
        h, present = self.attn(self.ln1(x), past_kv)
        x = x + h
        return x + self.mlp(self.ln2(x)), present


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    remove = probs.cumsum(dim=-1) - probs >= top_p
    filtered = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, filtered)


def _apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.amax(dim=-1, keepdim=True)
    return logits.masked_fill(probs < min_p * max_prob, float("-inf"))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe: nn.Embedding | None = None
        if config.pos_encoding == "learned":
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = _build_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            assert isinstance(block, Block)  # narrow past nn.ModuleList's loose item type
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.wpe is not None:
            n -= self.wpe.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> GPTOutput:
        _, t = idx.shape
        t_past = past_kv[0][0].size(2) if past_kv is not None else 0
        if t_past + t > self.config.block_size:
            raise ValueError(
                f"sequence length {t_past + t} exceeds block size {self.config.block_size}"
            )
        x = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(t_past, t_past + t, device=idx.device)
            x = x + self.wpe(pos)
        x = self.drop(x)
        present: KVCache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, past_kv[i] if past_kv is not None else None)
            present.append(kv)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if use_cache:
            return logits, loss, present
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        self.eval()
        out = idx
        cache: KVCache | None = None
        for _ in range(max_new_tokens):
            if use_cache:
                if cache is not None and cache[0][0].size(2) < self.config.block_size:
                    logits, _, cache = self(out[:, -1:], past_kv=cache, use_cache=True)
                else:
                    logits, _, cache = self(out[:, -self.config.block_size :], use_cache=True)
            else:
                logits, _ = self(out[:, -self.config.block_size :])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            if top_p is not None:
                logits = _apply_top_p(logits, top_p)
            if min_p is not None:
                logits = _apply_min_p(logits, min_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out = torch.cat((out, next_id), dim=1)
        return out

    def configure_optimizers(
        self, lr: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.95)
    ) -> torch.optim.AdamW:
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


GPT_PRESETS: dict[str, GPTConfig] = {
    "nano": GPTConfig(n_layer=3, n_head=3, n_embd=192),
    "micro": GPTConfig(n_layer=4, n_head=4, n_embd=256),
    "small": GPTConfig(n_layer=6, n_head=6, n_embd=384),
    "medium": GPTConfig(n_layer=8, n_head=8, n_embd=512),
}
