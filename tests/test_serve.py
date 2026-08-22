from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.serve import create_app, parse_args

VOCAB = sorted(set("hello world"))


def tiny_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(3)
    config = GPTConfig(
        vocab_size=len(VOCAB), block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0
    )
    model = GPT(config).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"vocab": VOCAB, "config": config.to_dict(), "model_state": model.state_dict()}, path
    )
    return path


def client(tmp_path: Path) -> Any:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient  # optional dependency

    app = create_app(tiny_checkpoint(tmp_path), device=torch.device("cpu"))
    return TestClient(app)


def test_parse_args_defaults_match_cli_contract() -> None:
    args = parse_args([])
    assert args.checkpoint == Path("checkpoints/checkpoint.pt")
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_completions_returns_openai_shaped_response(tmp_path: Path) -> None:
    response = client(tmp_path).post("/v1/completions", json={"prompt": "hello", "max_tokens": 4})
    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("cmpl-")
    assert body["object"] == "text_completion"
    choice = body["choices"][0]
    assert isinstance(choice["text"], str)
    assert choice["finish_reason"] in ("length", "stop")
    usage = body["usage"]
    assert usage["prompt_tokens"] == 5
    assert usage["completion_tokens"] == 4
    assert all(isinstance(value, int) for value in usage.values())


def test_completions_finish_reason_is_length_without_stop(tmp_path: Path) -> None:
    body = client(tmp_path).post("/v1/completions", json={"prompt": "he", "max_tokens": 3}).json()
    assert body["choices"][0]["finish_reason"] == "length"


def test_stop_string_truncates_text_and_sets_finish_reason(tmp_path: Path) -> None:
    test_client = client(tmp_path)
    model = test_client.app.state.model
    tok = test_client.app.state.tokenizer
    scripted = torch.tensor([tok.encode("llo")])

    def fake_generate(idx: torch.Tensor, max_new_tokens: int, **_: object) -> torch.Tensor:
        return torch.cat((idx, scripted.expand(idx.shape[0], -1)), dim=1)

    original = model.generate
    model.generate = fake_generate  # type: ignore[method-assign]
    try:
        body = test_client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 10, "stop": ["lo"]},
        ).json()
    finally:
        model.generate = original  # type: ignore[method-assign]

    assert body["choices"][0]["text"] == "l"  # truncated before the stop string
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 5
    assert body["usage"]["completion_tokens"] == 3  # all generated tokens are counted


def test_empty_prompt_rejected_with_400(tmp_path: Path) -> None:
    response = client(tmp_path).post("/v1/completions", json={"prompt": "", "max_tokens": 4})
    assert response.status_code == 400
