from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from gpt_from_scratch.export import export_hf, export_onnx
from gpt_from_scratch.model import GPT, GPTConfig

VOCAB = sorted(set("hello world"))


def build_model(**overrides: Any) -> GPT:
    torch.manual_seed(11)
    config = GPTConfig(
        vocab_size=len(VOCAB),
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        **overrides,
    )
    return GPT(config).eval()


def test_export_hf_writes_exactly_three_files(tmp_path: Path) -> None:
    out = export_hf(build_model(), VOCAB, tmp_path / "hf")
    assert out == tmp_path / "hf"
    assert sorted(p.name for p in out.iterdir()) == [
        "config.json",
        "pytorch_model.bin",
        "vocab.json",
    ]


def test_config_json_has_gpt2_fields(tmp_path: Path) -> None:
    model = build_model()
    config = json.loads((export_hf(model, VOCAB, tmp_path / "hf") / "config.json").read_text())
    assert config["model_type"] == "gpt2"
    assert config["n_positions"] == 16
    assert config["n_embd"] == 16
    assert config["n_layer"] == 2
    assert config["n_head"] == 2
    assert config["vocab_size"] == len(VOCAB)


def test_weights_roundtrip_with_gpt2_names_and_conv1d_transpose(tmp_path: Path) -> None:
    model = build_model()
    loaded = torch.load(export_hf(model, VOCAB, tmp_path / "hf") / "pytorch_model.bin")
    src = model.state_dict()

    # Embeddings and final norm keep their names (with the transformer. prefix).
    assert torch.equal(loaded["transformer.wte.weight"], src["wte.weight"])
    assert torch.equal(loaded["transformer.wpe.weight"], src["wpe.weight"])
    assert torch.equal(loaded["ln_f.weight"], src["ln_f.weight"])
    assert torch.equal(loaded["transformer.h.1.ln_2.bias"], src["blocks.1.ln2.bias"])

    # Conv1D stores (in, out): nn.Linear weights must arrive transposed.
    qkv = loaded["transformer.h.0.attn.c_attn.weight"]
    e = 16
    assert qkv.shape == (e, 3 * e)
    assert torch.equal(qkv.t(), src["blocks.0.attn.qkv.weight"])
    assert qkv.shape != src["blocks.0.attn.qkv.weight"].shape
    assert torch.equal(
        loaded["transformer.h.0.attn.c_proj.weight"].t(), src["blocks.0.attn.proj.weight"]
    )
    fc = loaded["transformer.h.0.mlp.c_fc.weight"]
    assert fc.shape == (e, 4 * e)
    assert torch.equal(fc.t(), src["blocks.0.mlp.fc.weight"])
    assert torch.equal(
        loaded["transformer.h.1.mlp.c_proj.weight"].t(), src["blocks.1.mlp.proj.weight"]
    )

    # Biases are copied without transposition.
    assert torch.equal(loaded["transformer.h.0.attn.c_attn.bias"], src["blocks.0.attn.qkv.bias"])
    assert torch.equal(loaded["transformer.h.0.mlp.c_fc.bias"], src["blocks.0.mlp.fc.bias"])


def test_lm_head_equals_wte_when_tied(tmp_path: Path) -> None:
    model = build_model()
    assert model.config.tie_embeddings
    loaded = torch.load(export_hf(model, VOCAB, tmp_path / "hf") / "pytorch_model.bin")
    assert torch.equal(loaded["lm_head.weight"], loaded["transformer.wte.weight"])
    assert torch.equal(loaded["lm_head.weight"], model.state_dict()["wte.weight"])


def test_vocab_json_maps_token_to_id(tmp_path: Path) -> None:
    vocab_path = export_hf(build_model(), VOCAB, tmp_path / "hf") / "vocab.json"
    mapping = json.loads(vocab_path.read_text(encoding="utf-8"))
    assert mapping == {token: i for i, token in enumerate(VOCAB)}


@pytest.mark.parametrize(
    "overrides",
    [
        {"pos_encoding": "rope"},
        {"norm_type": "rmsnorm"},
        {"mlp_type": "swiglu"},
        {"tie_embeddings": False},
        {"pre_norm": False},
    ],
)
def test_non_gpt2_representable_configs_raise_not_implemented(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    model = build_model(**overrides)
    with pytest.raises(NotImplementedError, match="does not support"):
        export_hf(model, VOCAB, tmp_path / "hf")
    with pytest.raises(NotImplementedError, match="does not support"):
        export_onnx(model, VOCAB, tmp_path / "model.onnx")


def test_export_onnx_produces_file(tmp_path: Path) -> None:
    pytest.importorskip("onnx")  # torch's exporter needs the onnx package
    path = export_onnx(build_model(), VOCAB, tmp_path / "model.onnx")
    assert path == tmp_path / "model.onnx"
    assert path.stat().st_size > 0


def test_export_onnx_names_extra_when_onnx_missing(tmp_path: Path) -> None:
    if importlib.util.find_spec("onnx") is not None:
        pytest.skip("onnx is installed; the missing-dependency path is not exercised")
    with pytest.raises(RuntimeError, match=r"\[onnx\]"):
        export_onnx(build_model(), VOCAB, tmp_path / "model.onnx")


def test_exported_onnx_accepts_dynamic_shapes(tmp_path: Path) -> None:
    ort = pytest.importorskip("onnxruntime")
    path = export_onnx(build_model(), VOCAB, tmp_path / "model.onnx")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch, seq in ((1, 2), (3, 5)):
        ids = torch.randint(0, len(VOCAB), (batch, seq)).numpy()
        logits = session.run(["logits"], {"input_ids": ids})[0]
        assert logits.shape == (batch, seq, len(VOCAB))
