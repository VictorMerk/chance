# Contributing

Thanks for considering a contribution. This document covers the development setup, the conventions the codebase follows, and how to land a change.

## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python 3.12 or newer.

```bash
git clone https://github.com/VictorMerk/gpt-from-scratch.git
cd gpt-from-scratch
uv sync --dev          # create .venv and install runtime + dev dependencies
uv run pytest -q       # run the test suite
uv run ruff check .    # lint
uv run ruff format .   # format (CI checks with --check)
```

Enable the pre-commit hooks so linting and formatting run automatically on every commit:

```bash
uvx pre-commit install
```

All tests run on CPU in a few seconds; no GPU is needed for development.

## Code conventions

- **Type hints everywhere.** Every function signature, including tests, is fully annotated. Modules start with `from __future__ import annotations`.
- **Line length is 100 columns**, enforced by ruff (`[tool.ruff] line-length = 100` in `pyproject.toml`).
- **Comments only when non-obvious.** Prefer clear names over narration. Comment the *why* when the code encodes a subtle decision (for example, why `is_causal=True` cannot be used once queries are shorter than keys).
- **Dataclass configs.** Model and training configuration live in dataclasses following the `GPTConfig` pattern, with `to_dict()` / `from_dict()` so configs round-trip through checkpoints.
- **CLIs use argparse** with one `parse_args()` function per script and kebab-case flags.
- Keep modules small and single-purpose: model, tokenizers, data, schedule, and one module per CLI entry point.

## Test organization

Tests mirror the source layout under `tests/`:

| File | Covers |
| --- | --- |
| `test_model.py` | forward shapes, causal masking, KV-cache equivalence, top-p, initialization, presets, optimizer grouping, overfit sanity check |
| `test_bpe.py` | BPE training, roundtrips, save/load, error cases |
| `test_tokenizer.py` | char tokenizer vocab, roundtrips, unknown-character rejection |
| `test_schedule.py` | warmup linearity, cosine decay endpoints, validation errors |
| `test_data.py` | train/val split proportions and input validation |
| `test_train.py` | batching shapes, shifted targets, loss estimation |
| `test_evaluate.py` | perplexity, bits-per-character conversion, generation benchmark |

Run coverage the same way CI does:

```bash
uv run --no-sync pytest -q --cov=gpt_from_scratch --cov-report=term-missing
```

New behavior needs a test. Bug fixes need a regression test that fails without the fix.

## Picking up a ROADMAP item

1. Pick an unchecked item from [ROADMAP.md](ROADMAP.md) and comment on (or open) an issue so work is not duplicated.
2. Keep pull requests small: one roadmap item per PR.
3. Include tests for the new feature.
4. Match existing conventions; run `uv run ruff check . && uv run pytest -q` before pushing.
5. Update ROADMAP.md by checking off the item in the same PR, adding a `*(milestone N)*` marker if it closes out a milestone.
6. Add an entry under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for user-facing changes.

## Pull request checklist

- [ ] Tests pass locally: `uv run pytest -q`
- [ ] Lint and format clean: `uv run ruff check . && uv run ruff format --check .`
- [ ] New functionality covered by tests
- [ ] Documentation updated (README / ARCHITECTURE.md) if flags, signatures, or behavior changed
- [ ] CHANGELOG.md entry added under "Unreleased" if user-facing
- [ ] ROADMAP.md checklist updated if this completes an item
