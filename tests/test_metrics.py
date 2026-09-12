import torch

from rocu_net.metrics import batch_metrics


def test_perfect_segmentation_metrics():
    target = torch.zeros(2, 1, 16, 16)
    target[0, :, 3:10, 4:12] = 1
    probability = target.clone()
    values = batch_metrics(probability, target)
    for key in ("dice", "iou", "precision", "recall", "specificity", "accuracy", "f2", "boundary_f1"):
        assert torch.allclose(values[key], torch.ones_like(values[key]))
    assert torch.allclose(values["mae"], torch.zeros_like(values["mae"]))


def test_metrics_are_bounded():
    probability = torch.rand(3, 1, 12, 12)
    target = (torch.rand(3, 1, 12, 12) > 0.6).float()
    values = batch_metrics(probability, target)
    for tensor in values.values():
        assert torch.isfinite(tensor).all()
        assert (tensor >= 0).all() and (tensor <= 1).all()



def test_disjoint_boundaries_have_zero_f1():
    target = torch.zeros(1, 1, 32, 32)
    prediction = torch.zeros_like(target)
    target[:, :, 2:8, 2:8] = 1
    prediction[:, :, 20:26, 20:26] = 1
    assert batch_metrics(prediction, target)["boundary_f1"].item() == 0.0


def test_empty_boundaries_have_perfect_f1():
    empty = torch.zeros(1, 1, 32, 32)
    assert batch_metrics(empty, empty)["boundary_f1"].item() == 1.0
