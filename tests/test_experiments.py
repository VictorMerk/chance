from pathlib import Path

import pytest

from gpt_from_scratch.experiments import lr_ablation, tying_ablation
from gpt_from_scratch.experiments._runner import build_config, format_table

DATA_FILE = Path("data/input.txt")


def test_build_config_merges_overrides_and_validates() -> None:
    config = build_config(65, {"n_layer": 2, "tie_embeddings": False}, block_size=64, dropout=0.0)
    assert config.vocab_size == 65
    assert config.n_layer == 2
    assert config.block_size == 64
    assert config.tie_embeddings is False
    with pytest.raises(ValueError, match="unknown GPTConfig field"):
        build_config(65, {"not_a_field": 1}, block_size=64, dropout=0.0)


def test_lr_ablation_schedule_points_match_get_lr() -> None:
    warmup = lr_ablation.build_schedule(
        "warmup_cosine", 100, max_lr=1e-3, min_lr=1e-4, warmup_iters=10
    )
    assert warmup[0] == pytest.approx(1e-3 / 10)
    assert warmup[9] == pytest.approx(1e-3)
    assert warmup[99] == pytest.approx(1e-4)
    constant = lr_ablation.build_schedule("constant", 50, max_lr=1e-3, min_lr=1e-4, warmup_iters=5)
    assert constant == [1e-3] * 50
    with pytest.raises(ValueError, match="unknown lr mode"):
        lr_ablation.lr_for_step("bogus", 1, max_iters=10, max_lr=1e-3, min_lr=1e-4, warmup_iters=1)


def test_tying_ablation_overrides_and_delta() -> None:
    assert tying_ablation.overrides_for("tied")["tie_embeddings"] is True
    assert tying_ablation.overrides_for("untied")["tie_embeddings"] is False
    with pytest.raises(ValueError, match="unknown variant"):
        tying_ablation.overrides_for("bogus")
    results = [
        {"variant": "tied", "val_loss": 1.5, "params": 100, "seconds": 1.0},
        {"variant": "untied", "val_loss": 1.6, "params": 200, "seconds": 1.0},
    ]
    assert tying_ablation.delta_val_loss(results) == pytest.approx(0.1)
    assert tying_ablation.delta_val_loss([{"variant": "tied"}]) is None
    rows = tying_ablation.build_rows(results)
    assert rows[0] == ["tied", "100", "1.5000", "1.0"]


def test_format_table_aligns_columns() -> None:
    table = format_table(("a", "bb"), [["1", "2"], ["333", "4"]])
    lines = table.splitlines()
    assert lines[0] == "|   a | bb |"
    assert lines[1] == "|-----|----|"
    assert lines[2] == "|   1 |  2 |"
    assert lines[3] == "| 333 |  4 |"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="tiny shakespeare not downloaded")
def test_run_short_train_tiny_twin_runs_differ_by_tying() -> None:
    from gpt_from_scratch.experiments._runner import run_short_train

    common = {"n_layer": 2, "n_head": 2, "n_embd": 64}
    tied = run_short_train({**common, "tie_embeddings": True}, 3, "cpu")
    untied = run_short_train({**common, "tie_embeddings": False}, 3, "cpu")
    for run in (tied, untied):
        assert run["train_loss"] > 0.0
        assert run["val_loss"] > 0.0
        assert run["seconds"] > 0.0
    assert untied["params"] > tied["params"]
