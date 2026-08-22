from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from gpt_from_scratch.model import GPT
from gpt_from_scratch.sample import load_model

# Suffixes whose weight matrices are nn.Linear (out, in) upstream and must be
# transposed to GPT-2 Conv1D layout (in, out).
_CONV1D_WEIGHT_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)

_BLOCK_KEY_RENAMES: tuple[tuple[str, str], ...] = (
    ("attn.qkv.", "attn.c_attn."),
    ("attn.proj.", "attn.c_proj."),
    ("mlp.fc.", "mlp.c_fc."),
    ("mlp.proj.", "mlp.c_proj."),
    ("ln1.", "ln_1."),
    ("ln2.", "ln_2."),
)


def _require_gpt2_representable(model: GPT) -> None:
    """Raise NotImplementedError for configs the GPT-2/HF format cannot express."""
    config = model.config
    unsupported: list[str] = []
    if config.pos_encoding != "learned":
        unsupported.append(f"pos_encoding={config.pos_encoding!r} (rope)")
    if config.norm_type != "layernorm":
        unsupported.append(f"norm_type={config.norm_type!r} (rmsnorm)")
    if config.mlp_type != "gelu":
        unsupported.append(f"mlp_type={config.mlp_type!r} (swiglu)")
    if not config.tie_embeddings:
        unsupported.append("tie_embeddings=False (untied lm_head)")
    if not config.pre_norm:
        unsupported.append("pre_norm=False (post-norm)")
    if unsupported:
        raise NotImplementedError(
            f"GPT-2 export does not support: {', '.join(unsupported)}; only learned positions, "
            "layernorm, gelu MLP, pre-norm blocks and tied embeddings are representable"
        )


def _gpt2_state_dict(model: GPT) -> dict[str, torch.Tensor]:
    """Rename/transplant our state dict onto GPT2LMHeadModel parameter names."""
    exported: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        parts = key.split(".")
        if parts[0] == "blocks":
            rest = ".".join(parts[2:])
            for old, new in _BLOCK_KEY_RENAMES:
                rest = rest.replace(old, new)
            if rest.endswith(_CONV1D_WEIGHT_SUFFIXES):
                value = value.t().contiguous()
            key = f"transformer.h.{parts[1]}.{rest}"
        elif key == "wte.weight":
            key = "transformer.wte.weight"
        elif key == "wpe.weight":
            key = "transformer.wpe.weight"
        # ln_f.* and lm_head.weight already match the GPT-2 names.
        exported[key] = value.detach().clone()
    return exported


def export_hf(model: GPT, tokenizer_vocab: list[str], out_dir: Path) -> Path:
    """Write a Hugging Face ``GPT2LMHeadModel``-compatible directory.

    Contains exactly ``config.json``, ``pytorch_model.bin`` (renamed GPT-2 keys,
    attention/MLP weights transposed to Conv1D ``(in, out)`` layout) and
    ``vocab.json`` mapping token -> id.
    """
    _require_gpt2_representable(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "gpt2",
        "n_positions": model.config.block_size,
        "n_embd": model.config.n_embd,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "vocab_size": model.config.vocab_size,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    torch.save(_gpt2_state_dict(model), out_dir / "pytorch_model.bin")
    vocab = {token: i for i, token in enumerate(tokenizer_vocab)}
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return out_dir


class _LogitsOnly(nn.Module):
    """Adapter so the exported ONNX graph maps input_ids -> logits directly."""

    def __init__(self, model: GPT) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids)
        return logits


def export_onnx(model: GPT, tokenizer_vocab: list[str], out_path: Path) -> Path:
    """Export the model to ONNX (opset 17) with dynamic batch and sequence axes."""
    del tokenizer_vocab  # accepted for CLI symmetry with export_hf
    _require_gpt2_representable(model)
    try:
        import onnx  # noqa: F401  # torch's exporter serializes through it
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires the 'onnx' package; install it with: uv pip install -e '.[onnx]'"
        ) from exc
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, min(2, model.config.block_size), dtype=torch.long)
    torch.onnx.export(
        _LogitsOnly(model),
        (dummy,),
        str(out_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        dynamo=False,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained checkpoint for serving")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/checkpoint.pt"))
    parser.add_argument("--format", choices=("hf", "onnx"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint, torch.device("cpu"))
    if args.format == "hf":
        saved = export_hf(model, tokenizer.vocab, args.out)
    else:
        saved = export_onnx(model, tokenizer.vocab, args.out)
    print(f"saved {saved}")


if __name__ == "__main__":
    main()
