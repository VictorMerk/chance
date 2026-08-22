# Research log

Write-ups of the ablation and scaling studies shipped in
`src/gpt_from_scratch/experiments/`. Each script produces a JSON results file;
a write-up turns one result file into a documented finding.

## Experiments

### Scaling study — `experiments.scaling`

- **Question:** how does validation loss fall with parameter count at fixed
  training budget?
- **Run:** `python -m gpt_from_scratch.experiments.scaling --max-iters 300 --plot`
- **Record:** params vs val_loss per preset, wall-clock per size, plot
  `scaling.png`.
- **Runtime:** minutes on CPU for 300 iters; faster on a T4.

### LR schedule ablation — `experiments.lr_ablation`

- **Question:** do warmup and cosine decay beat a constant LR at this scale?
- **Run:** `python -m gpt_from_scratch.experiments.lr_ablation --max-iters 300 --plot`
- **Record:** final val_loss per schedule mode, schedule shapes in
  `lr_ablation.png`.
- **Runtime:** three short runs, minutes on CPU.

### Tied vs untied embeddings — `experiments.tying_ablation`

- **Question:** does weight tying still help at tiny scale?
- **Run:** `python -m gpt_from_scratch.experiments.tying_ablation --max-iters 300`
- **Record:** val_loss and parameter count for both variants, the reported
  delta.
- **Runtime:** two short runs, minutes on CPU.

## Write-up template

Copy this block into a new `docs/research/<experiment>.md`:

```markdown
# <Experiment title>

## Motivation

Why this knob, what the literature suggests, what we expect.

## Setup

Exact command, config, seed, hardware, training budget.

## Results

| variant | params | val_loss | seconds |
| ------- | ------ | -------- | ------- |
|         |        |          |         |

Plot: `<file>.png`

## Conclusion

What the numbers show, what we would change next (more iters, bigger models,
repeat seeds).
```

## Conventions

- One knob per experiment; twins share seed, data order, and schedule.
- Results always go to JSON via `--out`; plots are opt-in via `--plot`.
- Never fabricate numbers — a write-up cites its own results file.
