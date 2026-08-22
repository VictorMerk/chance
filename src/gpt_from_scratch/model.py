from __future__ import annotations

import math
from dataclasses import asdict, dataclass

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

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, int | float]) -> GPTConfig:
        return cls(**d)


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

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = c // self.n_head
        q = q.view(b, t, self.n_head, head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, head_dim).transpose(1, 2)
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


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, present = self.attn(self.ln1(x), past_kv)
        x = x + h
        return x + self.mlp(self.ln2(x)), present


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    remove = probs.cumsum(dim=-1) - probs >= top_p
    filtered = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, filtered)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
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
        if non_embedding:
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
        pos = torch.arange(t_past, t_past + t, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
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
