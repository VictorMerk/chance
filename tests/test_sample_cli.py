from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.sample import (
    SamplingConfig,
    apply_slash_command,
    check_stop,
    generate_stream,
    load_model,
    parse_args,
    parse_prompts_file,
)
from gpt_from_scratch.tokenizer import CharTokenizer

VOCAB = sorted(set("hello world"))


def tiny_model() -> GPT:
    torch.manual_seed(7)
    config = GPTConfig(
        vocab_size=len(VOCAB), block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0
    )
    return GPT(config).eval()


def tiny_tokenizer() -> CharTokenizer:
    return CharTokenizer(vocab=VOCAB)


class ScriptedModel:
    """Minimal stand-in for GPT.generate: replays a fixed id sequence one token per call."""

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.calls = 0
        self.config = SimpleNamespace(block_size=16)

    def generate(self, idx: torch.Tensor, max_new_tokens: int, **_: object) -> torch.Tensor:
        token = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1
        return torch.tensor([[token]])

    def parameters(self) -> Iterator[torch.Tensor]:
        return iter([torch.empty(0)])


def test_check_stop_no_hit_returns_text() -> None:
    assert check_stop("hello world", ["xyz"]) == ("hello world", False)
    assert check_stop("hello world", []) == ("hello world", False)


def test_check_stop_truncates_at_stop() -> None:
    assert check_stop("helloSTOPworld", ["STOP"]) == ("hello", True)
    assert check_stop("abcX", ["X"]) == ("abc", True)
    assert check_stop("Xabc", ["X"]) == ("", True)


def test_check_stop_earliest_of_multiple_stops_wins() -> None:
    assert check_stop("abXXcYY", ["YY", "XX"]) == ("ab", True)
    assert check_stop("abXXcYY", ["XX", "YY"]) == ("ab", True)


def test_check_stop_first_occurrence_wins() -> None:
    assert check_stop("aaaXbX", ["X"]) == ("aaa", True)


def test_check_stop_ignores_empty_string() -> None:
    assert check_stop("abc", [""]) == ("abc", False)


def test_check_stop_across_token_boundaries() -> None:
    # Simulates streaming: each piece arrives as one decoded token.
    pieces = ["hel", "lo w", "orld"]
    emitted = ""
    hit = False
    for piece in pieces:
        emitted += piece
        emitted, hit = check_stop(emitted, ["lo wo"])
        if hit:
            break
    assert hit
    assert emitted == "hel"


def test_parse_prompts_file_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    path = tmp_path / "prompts.txt"
    path.write_text("# header comment\n\nalpha\n   \nbeta\n# another\ngamma\n", encoding="utf-8")
    assert parse_prompts_file(path) == ["alpha", "beta", "gamma"]


def test_parse_args_defaults_are_unchanged() -> None:
    args = parse_args([])
    assert args.checkpoint == Path("checkpoints/checkpoint.pt")
    assert args.prompt == "\n"
    assert args.max_new_tokens == 500
    assert args.temperature == 0.8
    assert args.top_k is None
    assert args.top_p is None
    assert args.min_p is None
    assert args.seed == 1337
    assert args.stream is False
    assert args.stop is None
    assert args.prompts_file is None
    assert args.interactive is False


def test_parse_args_rejects_interactive_with_prompts_file() -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--interactive", "--prompts-file", "prompts.txt"])
    assert excinfo.value.code == 2


def test_generate_stream_matches_non_streaming_with_top_k_one() -> None:
    model = tiny_model()
    tok = tiny_tokenizer()
    prompt = "hello"
    prompt_ids = tok.encode(prompt)

    torch.manual_seed(42)
    reference = model.generate(
        torch.tensor([prompt_ids]), max_new_tokens=30, temperature=1.0, top_k=1
    )
    expected = tok.decode(reference[0].tolist()[len(prompt_ids) :])

    torch.manual_seed(999)  # different seed must not matter under top_k=1
    settings = SamplingConfig(max_new_tokens=30, temperature=1.0, top_k=1)
    chunks = list(generate_stream(model, tok, prompt_ids, settings))

    assert "".join(chunks) == expected


def test_generate_stream_halts_at_stop_spanning_tokens() -> None:
    tok = CharTokenizer.from_text("abcd")  # a=0 b=1 c=2 d=3
    model = ScriptedModel([1, 2, 3])  # generates "bcd"
    settings = SamplingConfig(max_new_tokens=5, stop=["bc"])
    chunks = list(generate_stream(model, tok, [tok.encode("a")[0]], settings))
    assert "".join(chunks) == ""
    assert model.calls == 2  # halted on the second token, never generated 'd'


def test_generate_stream_emits_text_before_stop_without_partial_match() -> None:
    tok = CharTokenizer.from_text("axy")
    model = ScriptedModel(tok.encode("xay"))
    settings = SamplingConfig(max_new_tokens=3, stop=["ay"])
    chunks = list(generate_stream(model, tok, [], settings))
    assert chunks == ["x"]
    assert model.calls == 3


def test_apply_slash_command_adjusts_settings() -> None:
    settings = SamplingConfig()
    assert apply_slash_command("/temp 0.5", settings) is None
    assert settings.temperature == 0.5
    assert apply_slash_command("/top-k 5", settings) is None
    assert settings.top_k == 5
    assert apply_slash_command("/top-p 0.9", settings) is None
    assert settings.top_p == 0.9
    assert apply_slash_command("/min-p 0.1", settings) is None
    assert settings.min_p == 0.1


def test_apply_slash_command_rejects_bad_input() -> None:
    settings = SamplingConfig()
    assert apply_slash_command("/bogus 1", settings) is not None
    assert apply_slash_command("/temp abc", settings) is not None
    assert apply_slash_command("/temp", settings) is not None
    assert settings.temperature == 0.8  # unchanged after failed commands


def test_load_model_roundtrip_from_in_memory_checkpoint(tmp_path: Path) -> None:
    model = tiny_model()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "vocab": VOCAB,
            "config": model.config.to_dict(),
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )

    loaded, tokenizer = load_model(checkpoint_path, torch.device("cpu"))

    assert tokenizer.vocab == VOCAB
    assert loaded.config == model.config
    loaded.eval()
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
