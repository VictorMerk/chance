# Experiments

Short, reproducible studies that answer one question each about training dynamics, built on the same data-loading, batching, and evaluation helpers as the trainer. Every study is a standalone CLI under `src/gpt_from_scratch/experiments/` and shares one seeded runner (`experiments/_runner.py`).

The reproducibility contract: seeding happens before model construction, so two runs with the same seed and config see identical initial weights and identical batch order. Ablations therefore differ in exactly one knob — the comparison is a twin experiment, not a statistical argument.

All commands run from the repository root via `uv`. Plots require the optional plot extra (`uv sync --extra plot`); the JSON output never does.

## The shared runner

`run_short_train` in `experiments/_runner.py` trains one char-level model on tiny Shakespeare with these fixed defaults: batch size 32, block size 128, dropout 0.1, AdamW (lr 1e-3, weight decay 0.1), gradient clipping 1.0, warmup + cosine schedule decaying to 10% of peak lr, and 25-batch final evaluation. Studies override only what they are measuring. Each run reports `train_loss`, `val_loss`, `params`, and `seconds`.

## Study 1: Scaling probe

**Question:** how does validation loss fall as parameter count grows, holding data, budget, and recipe fixed?

Trains each named preset from `GPT_PRESETS` for an identical short budget on shakespeare-char and records final validation loss against parameter count.

```bash
uv run python -m gpt_from_scratch.experiments.scaling --max-iters 300 --plot
```

Useful flags: `--sizes nano,micro,small` (any comma-separated subset of `nano/micro/small/medium`; default `nano,micro,small`), `--max-iters 300`, `--device cpu` (auto-selects CUDA otherwise), `--seed 1337`, `--out scaling_results.json`, `--plot-out scaling.png`.

**Runtime expectations (300 iterations, ballpark planning figures — measure your own):**

| Device | Expected total time |
| --- | --- |
| Laptop CPU | minutes; grows steeply with the largest preset chosen |
| T4 GPU | well under a minute |

**Reading the results:** the script prints a `name / params / val_loss / seconds` table and writes `scaling_results.json`, a JSON array of objects:

```json
[{"name": "nano", "params": 0, "val_loss": 0.0, "seconds": 0.0}]
```

With `--plot` it saves `scaling.png`: validation loss versus parameter count on a log-x axis, one labeled point per size. A healthy curve falls monotonically with flattening returns; a flat segment means the short budget, not capacity, is the binding constraint.

**Results (fill after running; do not commit numbers without the command and seed used):**

| Size | Params | Val loss | Seconds |
| --- | --- | --- | --- |
| | | | |
| | | | |
| | | | |

## Study 2: LR-schedule ablation

**Question:** does warmup plus cosine decay beat constant or cosine-only schedules when everything else is held fixed?

Trains three identical twin runs (default 2 layers / 2 heads / 128 dims) differing only in the learning-rate schedule: `constant`, `cosine` (no warmup), and `warmup_cosine` (the repo default).

```bash
uv run python -m gpt_from_scratch.experiments.lr_ablation --max-iters 300 --plot
```

Useful flags: `--lr 1e-3` (peak), `--warmup-iters 100`, tiny-model dimensions via `--n-layer/--n-head/--n-embd` (2/2/128), `--device`, `--seed 1337`, `--out lr_ablation_results.json`, `--plot-out lr_ablation.png`.

**Runtime expectations (three runs of 300 iterations at the default tiny config, ballpark):**

| Device | Expected total time |
| --- | --- |
| Laptop CPU | low single-digit minutes for all three arms |
| T4 GPU | seconds per arm |

**Reading the results:** the printed table has `mode / final_lr / val_loss / seconds`; the JSON array stores, per arm, `mode`, `train_loss`, `val_loss`, `params`, `seconds`, and `final_lr`. With `--plot` it saves `lr_ablation.png`: left panel overlays the three schedules step by step (useful to sanity-check warmup length against the run budget), right panel bars the final validation losses. Expect differences between arms to be modest at this scale — the point is the method, not a dramatic effect.

**Results (empty until you run it):**

| Mode | Final lr | Val loss | Seconds |
| --- | --- | --- | --- |
| constant | | | |
| cosine | | | |
| warmup_cosine | | | |

## Planned studies

- **Tied vs untied embeddings ablation** — driver script ready: `python -m gpt_from_scratch.experiments.tying_ablation --max-iters 300`; full run pending.
- **Pre-norm vs post-norm ablation** — `pre_norm=False` switches blocks to post-norm today (settable through the runner's config overrides), but no dedicated script or full run yet.
- **Research-log series** — each finished study gets a write-up under `docs/research/`.

## Conventions for adding experiments

1. New module goes in `src/gpt_from_scratch/experiments/`, invoked as `python -m gpt_from_scratch.experiments.<name>` with an argparse `parse_args`/`main` pair. Never call `train.main()`; reuse `run_short_train` so data handling, batching, and evaluation stay comparable across studies.
2. Change exactly one thing per ablation and keep the seed fixed. If a study needs a custom schedule, pass `lr_fn` rather than reimplementing the loop.
3. Results go to a JSON file (`--out`), human-readable summary via the shared `format_table`, and plotting stays opt-in behind `--plot` using `require_matplotlib` so the JSON path never needs extras.
4. Mirror pure helpers (schedule builders, row formatters) in `tests/` so studies stay testable without training.
5. Record findings in `docs/research/<name>.md` only after a committed command and seed can reproduce them, then paste the numbers into this file's tables.
