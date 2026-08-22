"""Short, reproducible experiment drivers built on the core training pieces.

Each module is a standalone CLI (``python -m gpt_from_scratch.experiments.<name>``)
that reuses the data loading, batching, and evaluation helpers from ``train.py``
without invoking ``train.main()``. Importing any module here has no side effects.
"""

from gpt_from_scratch.experiments._runner import RunResult, run_short_train

__all__ = ["RunResult", "run_short_train"]
