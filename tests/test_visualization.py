from pathlib import Path

import numpy as np
import pandas as pd

from rocu_net.visualization import error_map, select_samples, training_figure
from visualize import load_evaluation_pairs


def test_selection_includes_failures_and_is_reproducible():
    frame = pd.DataFrame({"sample_id": list("abcdefg"), "dice": [0, .2, .4, .5, .7, .9, 1]})
    assert select_samples(frame, 3, "quantiles", 42) == ["a", "d", "g"]
    assert select_samples(frame, 2, "best", 42) == ["g", "f"]
    assert select_samples(frame, 2, "worst", 42) == ["a", "b"]
    assert select_samples(frame, 3, "random", 42) == select_samples(frame, 3, "random", 42)


def test_error_colors_distinguish_false_positive_and_false_negative():
    target = np.array([[0, 1], [0, 1]], dtype=bool)
    pred = np.array([[0, 1], [1, 0]], dtype=bool)
    actual = error_map(pred, target)
    assert actual.tolist() == [[[0, 0, 0], [0, 158, 115]], [[230, 159, 0], [204, 121, 167]]]


def test_paper_evaluation_requires_original_split(tmp_path):
    try:
        load_evaluation_pairs({"data": {"root": str(tmp_path)}}, tmp_path, "test")
    except FileNotFoundError as error:
        assert "original evaluation manifest" in str(error)
    else:
        raise AssertionError("Missing manifest was silently accepted")
    assert not (tmp_path / "splits").exists()


def test_training_log_exports_actual_values(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("Epoch 001 | train 2.1000 | val 2.3000 | Dice 0.5000 | IoU 0.3500\nEpoch 002 | train 1.1000 | val 1.3000 | Dice 0.8000 | IoU 0.7000\n")
    training_figure(log, tmp_path / "figures", dpi=72)
    frame = pd.read_csv(tmp_path / "figures/training_data.csv")
    assert frame.val_dice.tolist() == [0.5, 0.8]
    assert (tmp_path / "figures/training_curves.pdf").is_file()


def test_new_dataset_configs_inherit_training_settings():
    from rocu_net.config import load_config
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/kvasir.yaml")
    for name, dataset in [("cvc_colondb", "CVC-ColonDB"), ("etis", "ETIS")]:
        cfg = load_config(root / f"configs/{name}.yaml")
        assert cfg["data"]["root"] == f"dataset/{dataset}"
        expected_training = dict(base["training"])
        if name == "cvc_colondb":
            expected_training["test_after_training"] = False
        assert cfg["training"] == expected_training
        assert cfg["experiment"]["name"] != base["experiment"]["name"]
