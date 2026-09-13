"""Shared inference for evaluation, prediction and paper figures."""
from __future__ import annotations

import torch
from torch import nn


class InferenceModel(nn.Module):
    """Average aligned probabilities from identity and three image flips.

    No resizing or threshold change is applied. Averaging aligned occupancy
    pyramids preserves the model's linear cross-scale conservation constraint.
    This wrapper is used for inference only; save the underlying model weights.
    """

    def __init__(self, model: nn.Module, tta: str = "none"):
        super().__init__()
        if tta not in {"none", "flip"}:
            raise ValueError("inference.tta must be 'none' or 'flip'")
        self.model = model
        self.tta = tta

    @property
    def rocu1(self):
        return self.model.rocu1

    @property
    def rocu2(self):
        return self.model.rocu2

    @property
    def occupancy1(self):
        return self.model.occupancy1

    def forward(self, image):
        if self.tta == "none":
            return self.model(image)
        if self.training:
            raise RuntimeError("Flip TTA requires eval() mode")
        totals = {}
        for dims in ((), (-1,), (-2,), (-2, -1)):
            outputs = self.model(image.flip(dims) if dims else image)
            for stage in (1, 2):
                uncertainty = outputs.get(f"routing_uncertainty_{stage}")
                block = getattr(self.model, f"rocu{stage}", None)
                if uncertainty is not None and block is not None:
                    active = (uncertainty >= block.uncertainty_threshold
                              if block.routing_enabled and block.hard_routing_inference
                              else torch.ones_like(uncertainty, dtype=torch.bool))
                    if not getattr(block, "occupancy_constraint", True):
                        active = torch.zeros_like(uncertainty, dtype=torch.bool)
                    outputs[f"routing_active_{stage}"] = active.float()
            for key, value in outputs.items():
                if dims and value.ndim >= 4:
                    value = value.flip(dims)
                value = value.float()
                totals[key] = totals[key] + value if key in totals else value
        return {key: value / 4 for key, value in totals.items()}


def inference_model(model: nn.Module, config: dict, override: str | None = None) -> InferenceModel:
    tta = override if override is not None else config.get("inference", {}).get("tta", "none")
    return InferenceModel(model, tta).eval()
