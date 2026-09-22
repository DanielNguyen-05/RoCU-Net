from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import rocu_net.model as model_module
from rocu_net.model import RoCUNet
from rocu_net.routing_figure import (balanced_mode_orders, extract_routing_maps, measure_routing,
                                    repeat_measurements, save_dense_routed_table, save_paper_table,
                                    set_routing_mode, summarize_measurements)


def small_model():
    torch.manual_seed(4)
    return RoCUNet(backbone="efficient", pretrained=False,
                   backbone_channels=(8, 12, 16, 24), backbone_depths=(1, 1, 1, 1),
                   decoder_channels=(16, 12), carrier_channels=8, shallow_guide_channels=4).eval()


def test_active_maps_are_actual_solver_inputs_and_nearest_parent_cells():
    model = small_model()
    set_routing_mode(model, .3)
    captures = []
    original = model_module._routed_occupancy_solve

    def capture(parent, scores, active, iterations, epsilon):
        captures.append((parent.clone(), active.clone()))
        return original(parent, scores, active, iterations, epsilon)

    with patch("rocu_net.model._routed_occupancy_solve", side_effect=capture), torch.inference_mode():
        outputs = model(torch.randn(1, 3, 64, 64))
        maps = extract_routing_maps(outputs, .3)
    assert len(captures) == 2
    for stage, (parent, active) in enumerate(captures, 1):
        assert np.array_equal(maps[f"parent_{stage}"], parent[0, 0].numpy())
        assert np.array_equal(maps[f"active_{stage}"], active[0, 0].numpy())
        scale = 4 if stage == 1 else 2
        assert np.array_equal(maps[f"active_overlay_{stage}"],
                              np.repeat(np.repeat(maps[f"active_{stage}"], scale, 0), scale, 1))


def test_dense_preserves_soft_blending_and_zero_threshold_matches_dense():
    model = small_model()
    image = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        set_routing_mode(model, None)
        dense = model(image)
        assert model.rocu1.routing_enabled and not model.rocu1.hard_routing_inference
        assert torch.all(dense["routing_gate_1"] < 1)
        assert torch.all(dense["routing_active_1"] == 1)
        set_routing_mode(model, 0.)
        routed = model(image)
        assert torch.allclose(dense["p_full"], routed["p_full"], atol=2e-6)
        assert torch.allclose(dense["routing_gate_1"], routed["routing_gate_1"])


def test_confident_cells_are_bypassed_and_tampered_maps_are_rejected():
    model = small_model()
    torch.nn.init.zeros_(model.coarse_head.weight)
    torch.nn.init.constant_(model.coarse_head.bias, -5)
    with torch.inference_mode():
        set_routing_mode(model, .2)
        outputs = model(torch.randn(1, 3, 64, 64))
        maps = extract_routing_maps(outputs, .2)
        assert not maps["active_1"].any() and not maps["active_2"].any()
        outputs["routing_active_2"] = torch.ones_like(outputs["routing_active_2"])
        try:
            extract_routing_maps(outputs, .2)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Wrong solver masks must not be plotted")


def test_repeated_real_tensor_measurements_keep_identical_image_sets_and_accounting():
    model = small_model()
    dataset = [{"id": str(i), "image": torch.randn(3, 64, 64),
                "mask": (torch.rand(1, 64, 64) > .5).float()} for i in range(2)]
    modes = [("dense", None), ("tau_0.2", .2)]
    timings, metrics, samples, orders = measure_routing(
        model, dataset, torch.device("cpu"), modes, warmup=1, repeats=2, sample_id="0")
    assert len(timings) == 8 and len(metrics) == 4 and len(orders) == 2
    assert (timings.latency_ms_batch > 0).all()
    for label, _ in modes:
        assert set(metrics[metrics["mode"] == label].sample_id) == {"0", "1"}
        assert samples[label]["image"].shape == (64, 64, 3)
    summary = summarize_measurements(timings, metrics, modes)
    dense = summary.iloc[0]
    assert dense.speedup_vs_dense == 1 and dense.solver_skip_fraction == 0
    assert dense.dice == metrics[metrics["mode"] == "dense"].dice.mean()
    assert dense.latency_mean_ms_image == timings[timings["mode"] == "dense"].latency_ms_batch.mean()
    # A metric prediction is exactly the corresponding timed forward's prediction.
    with torch.inference_mode():
        set_routing_mode(model, .2)
        direct = model(dataset[0]["image"][None])["p_full"][0, 0].numpy()
    assert np.array_equal(direct, samples["tau_0.2"]["probability"])


def test_mode_order_balances_each_position_and_is_reproducible():
    orders = np.array(balanced_mode_orders(5, 10, 42))
    assert np.array_equal(orders, balanced_mode_orders(5, 10, 42))
    for cycle in (orders[:5], orders[5:]):
        for row in cycle:
            assert sorted(row) == list(range(5))
        for col in cycle.T:
            assert sorted(col) == list(range(5))


def test_paper_summary_uses_all_repeats_sample_sd_and_paired_dense(tmp_path):
    times = pd.DataFrame([{"mode": mode, "repeat": r, "n_images": 1,
                           "latency_ms_batch": value, "latency_ms_image": value}
                          for mode, values in (("dense", [10., 14.]), ("tau_0.2", [8., 12.]))
                          for r, value in enumerate(values)])
    metrics = pd.DataFrame([{"mode": mode, "sample_id": "image", "dice": dice, "iou": .7,
                             "boundary_f1": .6, "stage1_active_cells": active,
                             "stage1_total_cells": 4, "stage2_active_cells": 4 * active,
                             "stage2_total_cells": 16}
                            for mode, dice, active in (("dense", .8, 4), ("tau_0.2", .79, 1))])
    result = summarize_measurements(times, metrics, [("dense", None), ("tau_0.2", .2)])
    hard = result.iloc[1]
    assert hard.latency_mean_ms_image == 10 and hard.fps == 100
    assert np.isclose(hard.latency_repeat_sd_ms, np.sqrt(8))
    assert hard.paired_latency_saved_mean_ms == 2 and hard.paired_latency_saved_sd_ms == 0
    assert hard.repeats_faster_than_dense == 2 and hard.solver_skip_fraction == .75
    assert np.isclose(hard.dice_delta_vs_dense, -.01)
    assert hard.speedup_vs_dense == 1.2
    assert len(repeat_measurements(times)) == 4
    save_paper_table(result, tmp_path)
    assert "10.000 ± 2.828" in (tmp_path / "routing_summary.md").read_text()
    assert r"\tau_r=0.20" in (tmp_path / "routing_summary.tex").read_text()
    compact = save_dense_routed_table(result, tmp_path, .2)
    assert list(compact.active_cell_ratio) == [1., .25]
    assert "Dense solver & 0.800000 & 100.00" in (tmp_path / "routing_dense_vs_routed.tex").read_text()
    assert r"Routed, $\tau_r=0.20$ & 0.790000 & 25.00" in (tmp_path / "routing_dense_vs_routed.tex").read_text()
