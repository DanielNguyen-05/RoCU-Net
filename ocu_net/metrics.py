from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor, empty_value: float) -> torch.Tensor:
    value = numerator / denominator.clamp_min(1e-8)
    return torch.where(denominator > 0, value, torch.full_like(value, empty_value))


def _boundary_map(mask: torch.Tensor) -> torch.Tensor:
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (mask - eroded).clamp_min(0.0)


@torch.no_grad()
def batch_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    boundary_tolerance: int = 2,
) -> dict[str, torch.Tensor]:
    prediction = (probability >= threshold).float()
    target_binary = (target >= 0.5).float()
    dims = tuple(range(1, prediction.ndim))
    tp = (prediction * target_binary).sum(dim=dims)
    fp = (prediction * (1.0 - target_binary)).sum(dim=dims)
    fn = ((1.0 - prediction) * target_binary).sum(dim=dims)
    tn = ((1.0 - prediction) * (1.0 - target_binary)).sum(dim=dims)

    dice_den = 2.0 * tp + fp + fn
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn
    specificity_den = tn + fp
    accuracy_den = tp + tn + fp + fn
    beta2 = 4.0
    f2_den = (1.0 + beta2) * tp + beta2 * fn + fp

    pred_boundary = _boundary_map(prediction)
    target_boundary = _boundary_map(target_binary)
    kernel = 2 * int(boundary_tolerance) + 1
    pred_dilated = F.max_pool2d(pred_boundary, kernel_size=kernel, stride=1, padding=boundary_tolerance)
    target_dilated = F.max_pool2d(target_boundary, kernel_size=kernel, stride=1, padding=boundary_tolerance)
    pred_boundary_count = pred_boundary.sum(dim=dims)
    target_boundary_count = target_boundary.sum(dim=dims)
    boundary_precision = _safe_ratio(
        (pred_boundary * target_dilated).sum(dim=dims), pred_boundary_count, 1.0
    )
    boundary_recall = _safe_ratio(
        (target_boundary * pred_dilated).sum(dim=dims), target_boundary_count, 1.0
    )
    boundary_f1 = _safe_ratio(
        2.0 * boundary_precision * boundary_recall,
        boundary_precision + boundary_recall,
        0.0,
    )

    return {
        "dice": _safe_ratio(2.0 * tp, dice_den, 1.0),
        "iou": _safe_ratio(tp, iou_den, 1.0),
        "precision": _safe_ratio(tp, precision_den, 1.0),
        "recall": _safe_ratio(tp, recall_den, 1.0),
        "specificity": _safe_ratio(tn, specificity_den, 1.0),
        "accuracy": _safe_ratio(tp + tn, accuracy_den, 1.0),
        "f2": _safe_ratio((1.0 + beta2) * tp, f2_den, 1.0),
        "mae": torch.abs(probability - target).mean(dim=dims),
        "boundary_f1": boundary_f1,
    }


class BinarySegmentationMeter:
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.records: list[dict[str, float | str]] = []

    def update(self, probability: torch.Tensor, target: torch.Tensor, sample_ids: list[str]) -> None:
        values = batch_metrics(probability, target, threshold=self.threshold)
        for index, sample_id in enumerate(sample_ids):
            record: dict[str, float | str] = {"sample_id": sample_id}
            for name, tensor in values.items():
                record[name] = float(tensor[index].detach().cpu())
            self.records.append(record)

    def summary(self) -> dict:
        if not self.records:
            raise RuntimeError("No samples were added to the metric meter")
        numeric: dict[str, list[float]] = defaultdict(list)
        for record in self.records:
            for key, value in record.items():
                if key != "sample_id":
                    numeric[key].append(float(value))
        return {
            "n_images": len(self.records),
            "threshold": self.threshold,
            "mean": {key: float(np.mean(values)) for key, values in numeric.items()},
            "std": {key: float(np.std(values, ddof=0)) for key, values in numeric.items()},
        }

