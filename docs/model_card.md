# Model Card

> **TEMPLATE** — copy this file (e.g. to `docs/model_card_<name>.md`), replace every
> `{{PLACEHOLDER}}`, and delete this banner. A filled example row-set for the default
> shakespeare-char configuration is included at the bottom for reference.

## Model Details

| Field | Value |
| --- | --- |
| Model name | {{MODEL_NAME}} |
| Architecture | GPT — decoder-only transformer, pre-norm, learned absolute positions, GELU MLP (trained with `gpt_from_scratch`) |
| Parameter count | {{PARAMETER_COUNT}} |
| Layers / heads / embedding dim | {{N_LAYER}} / {{N_HEAD}} / {{N_EMBD}} |
| Context length | {{BLOCK_SIZE}} tokens |
| Tokenizer & vocab size | {{TOKENIZER_KIND_AND_SIZE}} |
| Checkpoint | {{CHECKPOINT_PATH_OR_URL}} |
| Training run date | {{TRAINING_DATE}} |
| License | MIT |

## Intended Use

- **Primary use cases:** {{PRIMARY_USE_CASES}}
- **Out of scope:** {{OUT_OF_SCOPE_USES}}

## Training Data

| Field | Value |
| --- | --- |
| Dataset | {{DATASET_NAME}} |
| Size | {{DATASET_SIZE_TOKENS_OR_CHARS}} |
| Source | {{DATASET_SOURCE_URL_OR_PATH}} |
| Preprocessing & splits | {{PREPROCESSING_AND_SPLIT}} |

## Evaluation

| Metric | Value |
| --- | --- |
| Validation loss | {{VAL_LOSS}} |
| Other metric | {{OTHER_METRIC}} |

Qualitative notes: {{QUALITY_NOTES}}

## Limitations

{{LIMITATIONS_BULLETS}}

## Usage

Load and sample from a checkpoint with `gpt_from_scratch`:

```python
from pathlib import Path

import torch

from gpt_from_scratch.sample import load_model

model, tokenizer = load_model(Path("{{CHECKPOINT_PATH}}"), torch.device("cpu"))
prompt = "{{EXAMPLE_PROMPT}}"
ids = model.generate(
    torch.tensor([tokenizer.encode(prompt)]), max_new_tokens=200, temperature=0.8, top_k=None
)
print(tokenizer.decode(ids[0].tolist()))
```

Optional — load the exported HF directory (`gpt-from-scratch-export --checkpoint ... --format hf --out exports/hf`)
with `transformers`:

```python
# pip install transformers
import json

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("exports/hf")
itos = {i: tok for tok, i in json.load(open("exports/hf/vocab.json")).items()}
# decode generated ids with: "".join(itos[i] for i in ids)
```

---

## Filled Example — `shakespeare-char` default configuration

Illustrative values for the repo's default char-level Shakespeare run
(`gpt-from-scratch-train --preset small --data data/tiny_shakespeare.txt`). Replace with
your own measured numbers; do not treat these as guarantees.

### Model Details (example)

| Field | Value |
| --- | --- |
| Model name | shakespeare-char-small |
| Parameter count | ~10.8M |
| Layers / heads / embedding dim | 6 / 6 / 384 |
| Context length | 256 tokens |
| Tokenizer & vocab size | character-level, 65 characters |

### Intended Use (example)

- **Primary use cases:** education/demonstration of training a GPT from scratch;
  unconditional generation of Shakespeare-flavoured character text.
- **Out of scope:** factual question answering, production text generation, any
  downstream deployment, or text in alphabets outside the 65-character training vocab.

### Training Data (example)

| Field | Value |
| --- | --- |
| Dataset | Tiny Shakespeare |
| Size | ~1.1M characters (~300k tokens at char level) |
| Source | downloaded by `gpt_from_scratch.data.download_tiny_shakespeare` (karpathy/tiny_shakespeare) |
| Preprocessing & splits | raw text as-is; first 90% train / last 10% validation |

### Evaluation (example)

| Metric | Value |
| --- | --- |
| Validation loss | ~1.5 after 5000 steps (AdamW, lr 1e-3, batch size 64) |

Qualitative notes: samples form plausible pseudo-Shakespeare line shapes and dialogue
structure, but contain frequent nonsense words and no coherent long-range plot.

### Limitations (example)

- Character-level model: no subword knowledge, short effective context, English-only.
- Generates fiction-like babble; output is not factual and frequently hallucinated.
- Trained on a single public-domain play corpus; reflects its biases and style only.
- Toy scale (~10M params); not competitive with any production language model.
