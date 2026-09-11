from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def probability_bce(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    prediction = prediction.float().clamp(epsilon, 1.0 - epsilon)
    # Explicit probability BCE is safe inside CUDA autocast.
    return -(target.float() * prediction.log() + (1.0 - target.float()) * torch.log1p(-prediction)).mean()


def soft_dice_loss(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction * target).sum(dim=dims)
    denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def weighted_structure_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_size: int = 31,
    edge_weight: float = 5.0,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Boundary-aware weighted BCE + weighted IoU region loss."""

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    prediction = prediction.float().clamp(1e-7, 1.0 - 1e-7)
    local_average = F.avg_pool2d(
        target,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    weights = 1.0 + float(edge_weight) * torch.abs(local_average - target)
    bce = -(target.float() * prediction.log() + (1.0 - target.float()) * torch.log1p(-prediction))
    weighted_bce = (weights * bce).sum(dim=(2, 3)) / weights.sum(dim=(2, 3)).clamp_min(1e-7)
    intersection = (prediction * target * weights).sum(dim=(2, 3))
    union = ((prediction + target) * weights).sum(dim=(2, 3))
    weighted_iou = 1.0 - (intersection + smooth) / (union - intersection + smooth)
    return (weighted_bce + weighted_iou).mean()


def boundary_target(target: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Differentiation-free morphological band around the ground-truth contour."""

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    padding = kernel_size // 2
    dilated = F.max_pool2d(target, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-target, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


class MultiScaleOccupancyLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        p176_weight: float = 0.5,
        p88_weight: float = 0.25,
        structure_weight: float = 0.0,
        boundary_weight: float = 0.0,
        routing_weight: float = 0.0,
        boundary_kernel_size: int = 5,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.p176_weight = float(p176_weight)
        self.p88_weight = float(p88_weight)
        self.structure_weight = float(structure_weight)
        self.boundary_weight = float(boundary_weight)
        self.routing_weight = float(routing_weight)
        self.boundary_kernel_size = int(boundary_kernel_size)

    def _scale_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = probability_bce(prediction, target)
        dice = soft_dice_loss(prediction, target)
        return self.bce_weight * bce + self.dice_weight * dice, bce, dice

    def forward(self, outputs: dict[str, torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        target176 = F.adaptive_avg_pool2d(target, outputs["p176"].shape[-2:])
        target88 = F.adaptive_avg_pool2d(target, outputs["p88"].shape[-2:])
        final, final_bce, final_dice = self._scale_loss(outputs["p352"], target)
        loss176, _, _ = self._scale_loss(outputs["p176"], target176)
        loss88, _, _ = self._scale_loss(outputs["p88"], target88)
        zero = final.new_zeros(())
        structure = (
            weighted_structure_loss(outputs["p352"], target)
            if self.structure_weight > 0.0
            else zero
        )
        target_boundary = boundary_target(target, self.boundary_kernel_size)
        predicted_boundaries = [
            outputs[key]
            for key in ("boundary_full", "boundary_half")
            if key in outputs
        ]
        boundary_losses: list[torch.Tensor] = []
        for prediction in predicted_boundaries:
            scaled_target = F.adaptive_max_pool2d(target_boundary, prediction.shape[-2:])
            boundary_losses.append(
                probability_bce(prediction, scaled_target)
                + soft_dice_loss(prediction, scaled_target)
            )
        boundary = torch.stack(boundary_losses).mean() if boundary_losses else zero

        routing_losses: list[torch.Tensor] = []
        for key in ("routing_gate_1", "routing_gate_2"):
            if key not in outputs:
                continue
            gate = outputs[key]
            scaled_target = F.adaptive_max_pool2d(target_boundary, gate.shape[-2:])
            routing_losses.append(probability_bce(gate, scaled_target))
        routing = torch.stack(routing_losses).mean() if routing_losses else zero

        total = (
            final
            + self.p176_weight * loss176
            + self.p88_weight * loss88
            + self.structure_weight * structure
            + self.boundary_weight * boundary
            + self.routing_weight * routing
        )
        components = {
            "loss": float(total.detach()),
            "final_loss": float(final.detach()),
            "final_bce": float(final_bce.detach()),
            "final_dice_loss": float(final_dice.detach()),
            "p176_loss": float(loss176.detach()),
            "p88_loss": float(loss88.detach()),
            "structure_loss": float(structure.detach()),
            "boundary_loss": float(boundary.detach()),
            "routing_loss": float(routing.detach()),
        }
        return total, components


def build_loss(config: dict) -> MultiScaleOccupancyLoss:
    cfg = config.get("loss", {})
    return MultiScaleOccupancyLoss(
        bce_weight=cfg.get("bce_weight", 1.0),
        dice_weight=cfg.get("dice_weight", 1.0),
        p176_weight=cfg.get("p176_weight", 0.5),
        p88_weight=cfg.get("p88_weight", 0.25),
        structure_weight=cfg.get("structure_weight", 0.0),
        boundary_weight=cfg.get("boundary_weight", 0.0),
        routing_weight=cfg.get("routing_weight", 0.0),
        boundary_kernel_size=cfg.get("boundary_kernel_size", 5),
    )
