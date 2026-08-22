from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.tokenizer import CharTokenizer


@dataclass
class SamplingConfig:
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    stop: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/checkpoint.pt"))
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--stream", action="store_true", help="print each token as it is generated")
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        metavar="STR",
        help="halt generation once this string appears (repeatable)",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="file with one prompt per line; blank lines and lines starting with # are skipped",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="REPL mode: read prompts from stdin until /quit or /exit",
    )
    args = parser.parse_args(argv)
    if args.interactive and args.prompts_file is not None:
        parser.error("--interactive and --prompts-file are mutually exclusive")
    return args


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GPT, CharTokenizer]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = CharTokenizer(vocab=checkpoint["vocab"])
    model = GPT(GPTConfig.from_dict(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def check_stop(text: str, stop_strings: Sequence[str]) -> tuple[str, bool]:
    """Truncate ``text`` at the earliest stop string; report whether one was found."""
    cut: int | None = None
    for stop in stop_strings:
        if not stop:
            continue
        i = text.find(stop)
        if i != -1 and (cut is None or i < cut):
            cut = i
    if cut is None:
        return text, False
    return text[:cut], True


def parse_prompts_file(path: Path) -> list[str]:
    """Return one prompt per non-blank, non-comment line of ``path``."""
    prompts: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


def generate_stream(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt_ids: list[int],
    settings: SamplingConfig,
) -> Iterator[str]:
    """Yield the continuation in chunks as tokens are produced.

    The chunks concatenate to exactly the continuation text that a single
    ``model.generate`` call with the same seed would produce. Stop strings are
    never emitted: a tail of ``len(longest_stop) - 1`` characters is held back
    until it is known to be free of a partial match.
    """
    device = next(model.parameters()).device
    stop = list(settings.stop)
    holdback = max((len(s) for s in stop), default=1) - 1
    ids = list(prompt_ids)
    emitted = 0
    text = ""
    for _ in range(settings.max_new_tokens):
        context = torch.tensor([ids[-model.config.block_size :]], dtype=torch.long, device=device)
        out = model.generate(
            context,
            max_new_tokens=1,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            min_p=settings.min_p,
            use_cache=True,
        )
        ids.append(int(out[0, -1]))
        text, hit = check_stop(tokenizer.decode(ids[len(prompt_ids) :]), stop)
        if hit:
            break
        safe = max(emitted, len(text) - holdback)
        if safe > emitted:
            yield text[emitted:safe]
            emitted = safe
    if len(text) > emitted:
        yield text[emitted:]


def complete(model: GPT, tokenizer: CharTokenizer, prompt: str, settings: SamplingConfig) -> str:
    """Return the generated continuation for ``prompt`` without printing."""
    prompt_ids = tokenizer.encode(prompt)
    return "".join(generate_stream(model, tokenizer, prompt_ids, settings))


def print_completion(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    settings: SamplingConfig,
    *,
    stream: bool,
    echo: bool = True,
) -> str:
    """Generate and print the continuation; returns the continuation text."""
    if stream:
        if echo:
            print(prompt, end="")
        chunks: list[str] = []
        prompt_ids = tokenizer.encode(prompt)
        for chunk in generate_stream(model, tokenizer, prompt_ids, settings):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        print()
        return "".join(chunks)
    continuation = complete(model, tokenizer, prompt, settings)
    print((prompt if echo else "") + continuation)
    return continuation


_SLASH_PARAMS: dict[str, tuple[str, type]] = {
    "/temp": ("temperature", float),
    "/top-k": ("top_k", int),
    "/top-p": ("top_p", float),
    "/min-p": ("min_p", float),
}

INTERACTIVE_HELP = "commands: /quit /exit /temp X /top-k X /top-p X /min-p X"


def apply_slash_command(line: str, settings: SamplingConfig) -> str | None:
    """Apply a '/temp 0.5' style command to ``settings``; return an error message or None."""
    name, _, value = line.partition(" ")
    entry = _SLASH_PARAMS.get(name)
    if entry is None:
        return f"unknown command: {name}"
    attr, cast = entry
    try:
        setattr(settings, attr, cast(value.strip()))
    except ValueError:
        return f"invalid value for {name}: {value!r}"
    return None


def run_interactive(
    model: GPT, tokenizer: CharTokenizer, settings: SamplingConfig, *, stream: bool
) -> None:
    print(INTERACTIVE_HELP)
    try:
        while True:
            try:
                line = input("> ")
            except EOFError:
                print()
                return
            line = line.strip()
            if not line:
                continue
            if line in ("/quit", "/exit"):
                return
            if line.startswith("/"):
                error = apply_slash_command(line, settings)
                if error is not None:
                    print(error)
                continue
            print_completion(model, tokenizer, line, settings, stream=stream, echo=False)
    except KeyboardInterrupt:
        print()


def run_prompts_file(
    model: GPT,
    tokenizer: CharTokenizer,
    prompts: Sequence[str],
    settings: SamplingConfig,
    *,
    stream: bool,
) -> None:
    for n, prompt in enumerate(prompts, start=1):
        print(f"--- prompt {n} ---")
        print_completion(model, tokenizer, prompt, settings, stream=stream, echo=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()
    model, tokenizer = load_model(args.checkpoint, device)
    settings = SamplingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        stop=list(args.stop or []),
    )
    if args.interactive:
        run_interactive(model, tokenizer, settings, stream=args.stream)
    elif args.prompts_file is not None:
        run_prompts_file(
            model,
            tokenizer,
            parse_prompts_file(args.prompts_file),
            settings,
            stream=args.stream,
        )
    else:
        print_completion(model, tokenizer, args.prompt, settings, stream=args.stream, echo=True)


if __name__ == "__main__":
    main()
