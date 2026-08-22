import math

import pytest
import torch
from torch.nn import functional as F

from gpt_from_scratch.model import GPT, GPT_PRESETS, GPTConfig, _apply_top_p


def tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=11,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )


def test_forward_returns_logits_and_loss() -> None:
    model = GPT(tiny_config())
    x = torch.randint(0, 11, (2, 8))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 8, 11)
    assert loss is not None
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_forward_without_targets_has_no_loss() -> None:
    model = GPT(tiny_config())
    x = torch.randint(0, 11, (1, 4))
    _, loss = model(x)
    assert loss is None


def test_attention_is_causal() -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config())
    model.eval()
    x = torch.randint(0, 11, (1, 8))
    x_mutated = x.clone()
    x_mutated[0, -1] = (x[0, -1] + 1) % 11
    logits, _ = model(x)
    logits_mutated, _ = model(x_mutated)
    assert torch.allclose(logits[:, :-1], logits_mutated[:, :-1], atol=1e-6)


def test_generate_produces_requested_length() -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config())
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=20, temperature=1.0, top_k=5)
    assert out.shape == (1, 21)


def test_sequence_longer_than_block_size_raises() -> None:
    model = GPT(tiny_config())
    x = torch.randint(0, 11, (1, tiny_config().block_size + 1))
    with pytest.raises(ValueError, match="block size"):
        model(x)


def test_optimizer_param_groups_split_decay() -> None:
    model = GPT(tiny_config())
    optimizer = model.configure_optimizers(lr=1e-3, weight_decay=0.1)
    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() < 2 for p in no_decay_group["params"])
    total = len(decay_group["params"]) + len(no_decay_group["params"])
    assert total == len(list(model.parameters()))


def test_loss_decreases_on_tiny_overfit() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    model = GPT(config)
    optimizer = model.configure_optimizers(lr=3e-3, weight_decay=0.0)
    x = torch.randint(0, 11, (4, 8))
    y = torch.randint(0, 11, (4, 8))
    _, first_loss = model(x, y)
    for _ in range(30):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert loss.item() < first_loss.item()


def test_cached_forward_matches_full_forward() -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config())
    model.eval()
    x = torch.randint(0, 11, (2, 8))
    full_logits, _ = model(x)
    split = x.size(1) // 2
    logits, _, cache = model(x[:, :split], use_cache=True)
    assert torch.allclose(full_logits[:, :split], logits, atol=1e-4)
    for pos in range(split, x.size(1)):
        logits, _, cache = model(x[:, pos : pos + 1], past_kv=cache, use_cache=True)
        assert torch.allclose(full_logits[:, pos], logits[:, 0], atol=1e-4)


def test_generate_cached_and_uncached_agree_with_top_k_1() -> None:
    torch.manual_seed(42)
    model = GPT(tiny_config())
    start = torch.tensor([[1, 2, 3]])
    torch.manual_seed(123)
    cached = model.generate(start.clone(), max_new_tokens=10, top_k=1, use_cache=True)
    torch.manual_seed(123)
    uncached = model.generate(start.clone(), max_new_tokens=10, top_k=1, use_cache=False)
    assert cached.shape == (1, 13)
    assert uncached.shape == (1, 13)
    assert torch.equal(cached, uncached)


def test_apply_top_p_keeps_prefix_mass() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1, -1.0]])
    out = _apply_top_p(logits, top_p=0.6)
    probs = F.softmax(logits, dim=-1)
    kept = torch.isfinite(out)
    assert kept.any()
    assert probs[kept].sum().item() >= 0.6
    assert torch.equal(out[kept], logits[kept])
    removed = ~kept
    assert torch.all(out[removed] == float("-inf"))
    flags = kept.squeeze(0)[logits.squeeze(0).argsort(descending=True)]
    n_kept = int(flags.sum())
    assert flags.tolist() == [True] * n_kept + [False] * (flags.numel() - n_kept)
    assert torch.isfinite(_apply_top_p(logits, top_p=1.0)).all()


def test_residual_projections_scaled_by_depth() -> None:
    config = tiny_config()
    model = GPT(config)
    weights = [block.attn.proj.weight.flatten() for block in model.blocks]
    weights += [block.mlp.proj.weight.flatten() for block in model.blocks]
    flat = torch.cat(weights)
    expected = 0.02 / math.sqrt(2 * config.n_layer)
    assert abs(flat.std().item() - expected) < 0.001


def test_gpt_presets_are_valid() -> None:
    assert set(GPT_PRESETS) == {"nano", "micro", "small", "medium"}
    for name, config in GPT_PRESETS.items():
        assert config.n_embd % config.n_head == 0, name
        assert GPT(config).num_parameters() > 0
