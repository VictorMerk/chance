import torch
from torch.nn import functional as F

from gpt_from_scratch.model import GPT, GPTConfig, RMSNorm, _apply_min_p


def arch_config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": 11,
        "block_size": 16,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 32,
        "dropout": 0.0,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


def train_tiny_steps(model: GPT, steps: int = 5) -> torch.Tensor:
    optimizer = model.configure_optimizers(lr=1e-3, weight_decay=0.0)
    x = torch.randint(0, 11, (4, 8))
    y = torch.randint(0, 11, (4, 8))
    loss = None
    for _ in range(steps):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert loss is not None
    return loss


def test_rope_cached_forward_matches_full_forward() -> None:
    torch.manual_seed(0)
    model = GPT(arch_config(pos_encoding="rope"))
    model.eval()
    x = torch.randint(0, 11, (2, 8))
    full_logits, _ = model(x)
    split = x.size(1) // 2
    logits, _, cache = model(x[:, :split], use_cache=True)
    assert torch.allclose(full_logits[:, :split], logits, atol=1e-4)
    for pos in range(split, x.size(1)):
        logits, _, cache = model(x[:, pos : pos + 1], past_kv=cache, use_cache=True)
        assert torch.allclose(full_logits[:, pos], logits[:, 0], atol=1e-4)


def test_rope_config_has_no_position_parameters() -> None:
    learned = GPT(arch_config())
    rope = GPT(arch_config(pos_encoding="rope"))
    assert any("wpe" in name for name, _ in learned.named_parameters())
    assert not [name for name, _ in rope.named_parameters() if "wpe" in name]
    assert rope.wpe is None


def test_rmsnorm_output_rms_is_one_per_token() -> None:
    norm = RMSNorm(16)
    x = torch.randn(4, 10, 16) * 3.0 + 1.5
    out = norm(x)  # weight starts at ones, so output is the normalized activation
    per_token_rms = out.pow(2).mean(dim=-1)
    assert torch.allclose(per_token_rms, torch.ones_like(per_token_rms), atol=1e-3)


def test_rmsnorm_model_trains_without_nan() -> None:
    torch.manual_seed(0)
    model = GPT(arch_config(norm_type="rmsnorm"))
    loss = train_tiny_steps(model, steps=5)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_swiglu_shapes_and_parameter_count_close_to_gelu() -> None:
    gelu = GPT(arch_config(mlp_type="gelu"))
    swiglu = GPT(arch_config(mlp_type="swiglu"))
    x = torch.randint(0, 11, (2, 8))
    logits, loss = swiglu(x, targets=x)
    assert logits.shape == (2, 8, 11)
    assert loss is not None and torch.isfinite(loss)
    gelu_params = sum(p.numel() for p in gelu.parameters())
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    ratio = swiglu_params / gelu_params
    assert 0.85 <= ratio <= 1.15


def test_embedding_tying_follows_config() -> None:
    tied = GPT(arch_config())
    untied = GPT(arch_config(tie_embeddings=False))
    assert tied.lm_head.weight.data_ptr() == tied.wte.weight.data_ptr()
    assert untied.lm_head.weight.data_ptr() != untied.wte.weight.data_ptr()
    # Untied lm_head still receives the standard normal_ init.
    assert abs(untied.lm_head.weight.std().item() - 0.02) < 0.005
    loss = train_tiny_steps(untied, steps=5)
    assert torch.isfinite(loss)


def test_apply_min_p_filters_below_scaled_threshold() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1, -1.0]])
    out = _apply_min_p(logits, min_p=0.1)
    probs = F.softmax(logits, dim=-1)
    expected_keep = probs >= 0.1 * probs.max()
    assert torch.equal(torch.isfinite(out).squeeze(0), expected_keep.squeeze(0))
    assert probs[expected_keep].sum().item() >= 1.0 - 0.1 - 1e-6
    assert torch.all(out[:, ~expected_keep.squeeze(0)] == float("-inf"))
    assert torch.equal(out[:, expected_keep.squeeze(0)], logits[:, expected_keep.squeeze(0)])


def test_apply_min_p_zero_keeps_everything() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1, -1.0]])
    assert torch.isfinite(_apply_min_p(logits, min_p=0.0)).all()


def test_min_p_does_not_change_deterministic_top_k_1_generate() -> None:
    torch.manual_seed(0)
    model = GPT(arch_config())
    start = torch.tensor([[1, 2, 3]])
    torch.manual_seed(123)
    base = model.generate(start.clone(), max_new_tokens=8, top_k=1)
    torch.manual_seed(123)
    filtered = model.generate(start.clone(), max_new_tokens=8, top_k=1, min_p=0.99)
    assert base.shape == (1, 11)
    assert torch.equal(base, filtered)


PRE_NORM_EXPECTED_KEYS = {
    "wte.weight",
    "wpe.weight",
    "blocks.0.ln1.weight",
    "blocks.0.ln1.bias",
    "blocks.0.attn.qkv.weight",
    "blocks.0.attn.qkv.bias",
    "blocks.0.attn.proj.weight",
    "blocks.0.attn.proj.bias",
    "blocks.0.ln2.weight",
    "blocks.0.ln2.bias",
    "blocks.0.mlp.fc.weight",
    "blocks.0.mlp.fc.bias",
    "blocks.0.mlp.proj.weight",
    "blocks.0.mlp.proj.bias",
    "blocks.1.ln1.weight",
    "blocks.1.ln1.bias",
    "blocks.1.attn.qkv.weight",
    "blocks.1.attn.qkv.bias",
    "blocks.1.attn.proj.weight",
    "blocks.1.attn.proj.bias",
    "blocks.1.ln2.weight",
    "blocks.1.ln2.bias",
    "blocks.1.mlp.fc.weight",
    "blocks.1.mlp.fc.bias",
    "blocks.1.mlp.proj.weight",
    "blocks.1.mlp.proj.bias",
    "ln_f.weight",
    "ln_f.bias",
    "lm_head.weight",
}


def test_pre_norm_state_dict_keys_are_stable() -> None:
    torch.manual_seed(0)
    model = GPT(arch_config())
    assert set(model.state_dict()) == PRE_NORM_EXPECTED_KEYS


def test_post_norm_trains_with_finite_decreasing_loss() -> None:
    torch.manual_seed(0)
    model = GPT(arch_config(pre_norm=False))
    optimizer = model.configure_optimizers(lr=1e-3, weight_decay=0.0)
    x = torch.randint(0, 11, (4, 8))
    y = torch.randint(0, 11, (4, 8))
    _, initial_loss = model(x, y)
    loss: torch.Tensor | None = None
    for _ in range(5):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert loss is not None
    assert torch.isfinite(initial_loss).all()
    assert torch.isfinite(loss).all()
    assert loss.item() < initial_loss.item()


def test_post_norm_changes_forward_computation() -> None:
    torch.manual_seed(0)
    pre = GPT(arch_config())
    torch.manual_seed(0)
    post = GPT(arch_config(pre_norm=False))
    pre.eval()
    post.eval()
    # Post-norm blocks have no per-block pre-MLP norm.
    assert not [name for name in post.state_dict() if ".ln2." in name]
    x = torch.randint(0, 11, (2, 8))
    pre_logits, _ = pre(x)
    post_logits, _ = post(x)
    assert not torch.allclose(pre_logits, post_logits)


def test_post_norm_composes_with_rmsnorm_and_rope() -> None:
    torch.manual_seed(0)
    model = GPT(arch_config(pos_encoding="rope", norm_type="rmsnorm", pre_norm=False))
    assert all(isinstance(block.ln1, RMSNorm) for block in model.blocks)
    model.eval()
    x = torch.randint(0, 11, (2, 8))
    full_logits, _ = model(x)
    split = x.size(1) // 2
    logits, _, cache = model(x[:, :split], use_cache=True)
    assert torch.allclose(full_logits[:, :split], logits, atol=1e-4)
    for pos in range(split, x.size(1)):
        logits, _, cache = model(x[:, pos : pos + 1], past_kv=cache, use_cache=True)
        assert torch.allclose(full_logits[:, pos], logits[:, 0], atol=1e-4)


def test_config_roundtrip_includes_pre_norm() -> None:
    assert GPTConfig().to_dict()["pre_norm"] is True
    cfg = arch_config(pre_norm=False)
    restored = GPTConfig.from_dict(cfg.to_dict())
    assert restored == cfg
    assert restored.pre_norm is False
