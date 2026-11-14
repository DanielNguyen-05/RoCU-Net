from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .compat import CheckpointCompatibleModule, normalize_config


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        padding: int = 1,
        groups: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ConvBNSiLU(nn.Sequential):
    """Convolution, batch normalization and SiLU used by the light path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.silu(self.reduce(scale), inplace=True)
        return x * torch.sigmoid(self.expand(scale))


class InvertedResidualBlock(nn.Module):
    """Small MBConv-style block for the self-contained encoder fallback."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        expansion: float = 2.0,
        use_se: bool = True,
    ):
        super().__init__()
        hidden = max(out_channels, int(round(in_channels * expansion)))
        layers: list[nn.Module] = []
        if hidden != in_channels:
            layers.append(ConvBNSiLU(in_channels, hidden, kernel_size=1, padding=0))
        layers.append(
            ConvBNSiLU(
                hidden,
                hidden,
                kernel_size=3,
                stride=stride,
                groups=hidden,
            )
        )
        if use_se:
            layers.append(SqueezeExcitation(hidden))
        layers.extend(
            [
                nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block(x)
        return output + x if self.use_residual else output


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = ConvBNSiLU(
            in_channels,
            in_channels,
            kernel_size=3,
            groups=in_channels,
        )
        self.pointwise = ConvBNSiLU(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class EfficientEncoder(nn.Module):
    """Dependency-free light encoder used for ablations and offline tests."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (16, 24, 40, 80),
        depths: Sequence[int] = (1, 2, 2, 2),
        expansion: float = 2.0,
    ):
        super().__init__()
        if len(channels) != 4 or len(depths) != 4:
            raise ValueError("EfficientEncoder requires four channel and depth values")
        c1, c2, c3, c4 = (int(value) for value in channels)
        self.out_channels = (c1, c2, c3, c4)
        self.stem = ConvBNSiLU(in_channels, c1, kernel_size=3, stride=2)

        def stage(in_ch: int, out_ch: int, depth: int, downsample: bool) -> nn.Sequential:
            blocks: list[nn.Module] = [
                InvertedResidualBlock(
                    in_ch,
                    out_ch,
                    stride=2 if downsample else 1,
                    expansion=expansion,
                )
            ]
            blocks.extend(
                InvertedResidualBlock(out_ch, out_ch, expansion=expansion)
                for _ in range(max(0, int(depth) - 1))
            )
            return nn.Sequential(*blocks)

        self.stage1 = stage(c1, c1, int(depths[0]), False)
        self.stage2 = stage(c1, c2, int(depths[1]), True)
        self.stage3 = stage(c2, c3, int(depths[2]), True)
        self.stage4 = stage(c3, c4, int(depths[3]), True)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        e1 = self.stage1(self.stem(image))
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)
        return e1, e2, e3, e4


class MobileNetV3LargeEncoder(nn.Module):
    """MobileNetV3-Large stem through r9, matching LV-UNet's strong encoder."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import (
                MobileNet_V3_Large_Weights,
                mobilenet_v3_large,
            )
        except ImportError as error:
            raise ImportError(
                "torchvision is required for backbone='mobilenet_v3_large'. "
                "Install requirements.txt or set backbone='efficient'."
            ) from error
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        try:
            network = mobilenet_v3_large(weights=weights)
        except Exception as error:
            if pretrained:
                raise RuntimeError(
                    "Could not load ImageNet MobileNetV3 weights. Cache/download the "
                    "weights first or set model.pretrained=false."
                ) from error
            raise
        self.features = nn.ModuleList(list(network.features[:10]))
        self.out_channels = (16, 24, 40, 80)
        self.output_indices = {1, 3, 6, 9}

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        x = image
        for index, block in enumerate(self.features):
            x = block(x)
            if index in self.output_indices:
                outputs.append(x)
        if len(outputs) != 4:
            raise RuntimeError("Unexpected torchvision MobileNetV3 feature layout")
        return tuple(outputs)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            ConvBNAct(in_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0)
        self.fuse = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.fuse(torch.cat((x, skip), dim=1))


class _OccupancyConstraint(torch.autograd.Function):
    """Redistribute four child scores while exactly conserving parent occupancy.

    Forward solves a monotone scalar equation by bisection without building an
    iteration graph. Backward uses implicit differentiation of
    mean(sigmoid(score + bias)) = parent.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        parent: torch.Tensor,
        scores: torch.Tensor,
        iterations: int,
        epsilon: float,
    ) -> torch.Tensor:
        if scores.shape[1] != 4 or parent.shape[1] != 1:
            raise ValueError("Expected parent Bx1xHxW and child scores Bx4xHxW")
        if parent.shape[0] != scores.shape[0] or parent.shape[-2:] != scores.shape[-2:]:
            raise ValueError("Parent and grouped child scores must share batch/spatial dimensions")
        score_dtype = scores.dtype
        with torch.no_grad():
            parent32 = parent.float().clamp(float(epsilon), 1.0 - float(epsilon))
            scores32 = scores.float()
            # These bounds guarantee that every sigmoid is approximately 0 or 1.
            lower = -32.0 - scores32.amax(dim=1, keepdim=True)
            upper = 32.0 - scores32.amin(dim=1, keepdim=True)
            for _ in range(int(iterations)):
                midpoint = (lower + upper) * 0.5
                child_mean = torch.sigmoid(scores32 + midpoint).mean(dim=1, keepdim=True)
                go_right = child_mean < parent32
                lower = torch.where(go_right, midpoint, lower)
                upper = torch.where(go_right, upper, midpoint)
            bias = (lower + upper) * 0.5
            children = torch.sigmoid(scores32 + bias)
        ctx.save_for_backward(children)
        ctx.epsilon = float(epsilon)
        ctx.parent_dtype = parent.dtype
        ctx.score_dtype = score_dtype
        return children.to(score_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (children,) = ctx.saved_tensors
        children = children.float()
        grad_output = grad_output.float()
        sigmoid_slope = children * (1.0 - children)
        slope_sum = sigmoid_slope.sum(dim=1, keepdim=True).clamp_min(ctx.epsilon)
        weighted_grad_sum = (grad_output * sigmoid_slope).sum(dim=1, keepdim=True)

        # Common shifts of all four scores are absorbed by the solved bias.
        grad_scores = sigmoid_slope * (grad_output - weighted_grad_sum / slope_sum)
        # d(mean(q))/d(parent) = 1; four children produce the factor four.
        grad_parent = 4.0 * weighted_grad_sum / slope_sum
        return (
            grad_parent.to(ctx.parent_dtype),
            grad_scores.to(ctx.score_dtype),
            None,
            None,
        )


class OccupancyBlock(nn.Module):
    """Occupancy-Constrained Upsampling block for a 2x scale transition."""

    def __init__(
        self,
        encoder_channels: int,
        guide_channels: int = 8,
        solver_iterations: int = 24,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.guide_projection = ConvBNAct(
            encoder_channels, guide_channels, kernel_size=1, padding=0
        )
        predictor_channels = guide_channels + 1
        self.score_predictor = nn.Sequential(
            ConvBNAct(
                predictor_channels,
                predictor_channels,
                kernel_size=3,
                padding=1,
                groups=predictor_channels,
            ),
            nn.Conv2d(predictor_channels, 1, kernel_size=1, bias=True),
        )
        self.solver_iterations = int(solver_iterations)
        self.epsilon = float(epsilon)

    def forward(self, parent: torch.Tensor, encoder_feature: torch.Tensor) -> torch.Tensor:
        expected_size = (parent.shape[-2] * 2, parent.shape[-1] * 2)
        if encoder_feature.shape[-2:] != expected_size:
            raise ValueError(
                f"Occupancy guide size {tuple(encoder_feature.shape[-2:])} must be 2x parent "
                f"size {tuple(parent.shape[-2:])}."
            )
        parent_up = F.interpolate(parent, size=expected_size, mode="nearest")
        guide = self.guide_projection(encoder_feature)
        dense_scores = self.score_predictor(torch.cat((parent_up, guide), dim=1))
        grouped_scores = F.pixel_unshuffle(dense_scores, downscale_factor=2)
        grouped_children = _OccupancyConstraint.apply(
            parent, grouped_scores, self.solver_iterations, self.epsilon
        )
        return F.pixel_shuffle(grouped_children, upscale_factor=2)


class LiteDecoderBlock(nn.Module):
    """Addition-based decoder block with depthwise spatial mixing."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.input_projection = ConvBNSiLU(
            in_channels, out_channels, kernel_size=1, padding=0
        )
        self.skip_projection = ConvBNSiLU(
            skip_channels, out_channels, kernel_size=1, padding=0
        )
        self.fusion = nn.Sequential(
            DepthwiseSeparableBlock(out_channels, out_channels),
            InvertedResidualBlock(out_channels, out_channels, expansion=2.0),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.input_projection(x)
        skip = self.skip_projection(skip)
        return self.fusion(x + skip)


@torch.no_grad()
def _routed_occupancy_solve(
    parent: torch.Tensor,
    scores: torch.Tensor,
    active: torch.Tensor,
    iterations: int,
    epsilon: float,
) -> torch.Tensor:
    """Inference-only solver that skips confident parent cells.

    Inactive cells use the exact conservative identity allocation q_j = p.
    Active cells use the same monotone bisection equation as the occupancy block.
    """

    dtype = scores.dtype
    parent_flat = parent.float().permute(0, 2, 3, 1).reshape(-1)
    scores_flat = scores.float().permute(0, 2, 3, 1).reshape(-1, 4)
    active_flat = active.reshape(-1).bool()
    children_flat = parent_flat[:, None].expand(-1, 4).clone()
    if bool(active_flat.any()):
        selected_parent = parent_flat[active_flat].clamp(float(epsilon), 1.0 - float(epsilon))
        selected_scores = scores_flat[active_flat]
        lower = -32.0 - selected_scores.amax(dim=1)
        upper = 32.0 - selected_scores.amin(dim=1)
        for _ in range(int(iterations)):
            midpoint = (lower + upper) * 0.5
            child_mean = torch.sigmoid(selected_scores + midpoint[:, None]).mean(dim=1)
            go_right = child_mean < selected_parent
            lower = torch.where(go_right, midpoint, lower)
            upper = torch.where(go_right, upper, midpoint)
        bias = (lower + upper) * 0.5
        children_flat[active_flat] = torch.sigmoid(selected_scores + bias[:, None])
    batch, _, height, width = scores.shape
    return (
        children_flat.reshape(batch, height, width, 4)
        .permute(0, 3, 1, 2)
        .to(dtype)
    )


class RoCUBlock(nn.Module):
    """Confidence-Routed Semantic Occupancy upsampling.

    A thin semantic carrier survives each 2x transition. The learned allocation
    is blended with the conservative identity allocation using one gate per
    parent cell, so the local-mean constraint remains valid. During inference,
    bisection can be skipped for confident cells without changing occupancy.
    """

    def __init__(
        self,
        encoder_channels: int,
        carrier_channels: int = 24,
        *,
        solver_iterations: int = 24,
        epsilon: float = 1e-6,
        routing_enabled: bool = True,
        hard_routing_inference: bool = True,
        uncertainty_threshold: float = 0.20,
        use_semantic_carrier: bool = True,
        occupancy_constraint: bool = True,
    ):
        super().__init__()
        self.carrier_channels = int(carrier_channels)
        self.solver_iterations = int(solver_iterations)
        self.epsilon = float(epsilon)
        self.routing_enabled = bool(routing_enabled)
        self.hard_routing_inference = bool(hard_routing_inference)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.use_semantic_carrier = bool(use_semantic_carrier)
        self.occupancy_constraint = bool(occupancy_constraint)
        self.carrier_upsampling = "pixelshuffle"
        if not self.occupancy_constraint and self.hard_routing_inference:
            raise ValueError("Without occupancy constraint, hard_routing_inference must be false")

        self.carrier_expansion = ConvBNSiLU(
            self.carrier_channels,
            4 * self.carrier_channels,
            kernel_size=1,
            padding=0,
        )
        self.guide_projection = ConvBNSiLU(
            int(encoder_channels),
            self.carrier_channels,
            kernel_size=1,
            padding=0,
        )
        self.semantic_fusion = nn.Sequential(
            DepthwiseSeparableBlock(self.carrier_channels, self.carrier_channels),
            InvertedResidualBlock(
                self.carrier_channels,
                self.carrier_channels,
                expansion=2.0,
            ),
        )
        self.boundary_head = nn.Conv2d(
            self.carrier_channels, 1, kernel_size=3, padding=1
        )
        gate_channels = max(8, self.carrier_channels // 2)
        self.gate_head = nn.Sequential(
            DepthwiseSeparableBlock(self.carrier_channels + 1, gate_channels),
            nn.Conv2d(gate_channels, 1, kernel_size=1),
        )
        self.score_head = nn.Sequential(
            ConvBNSiLU(
                self.carrier_channels + 2,
                self.carrier_channels + 2,
                kernel_size=3,
                groups=self.carrier_channels + 2,
            ),
            nn.Conv2d(self.carrier_channels + 2, 1, kernel_size=1),
        )

    def forward(
        self,
        parent: torch.Tensor,
        carrier: torch.Tensor,
        encoder_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        expected_size = (parent.shape[-2] * 2, parent.shape[-1] * 2)
        if encoder_feature.shape[-2:] != expected_size:
            raise ValueError(
                f"RoCU guide size {tuple(encoder_feature.shape[-2:])} must be "
                f"2x parent size {tuple(parent.shape[-2:])}."
            )
        if carrier.shape[1] != self.carrier_channels or carrier.shape[-2:] != parent.shape[-2:]:
            raise ValueError(
                "Carrier must match parent resolution and configured carrier_channels"
            )

        parent_up = F.interpolate(parent, size=expected_size, mode="nearest")
        expanded_carrier = (F.pixel_shuffle(self.carrier_expansion(carrier), 2)
                            if self.carrier_upsampling == "pixelshuffle"
                            else self.carrier_upsampler(carrier))
        if not self.use_semantic_carrier:
            expanded_carrier = torch.zeros_like(expanded_carrier)
        guide = self.guide_projection(encoder_feature)
        carrier_out = self.semantic_fusion(expanded_carrier + guide)

        boundary = torch.sigmoid(self.boundary_head(carrier_out))
        parent_uncertainty = (4.0 * parent * (1.0 - parent)).clamp(0.0, 1.0)
        uncertainty_up = F.interpolate(parent_uncertainty, size=expected_size, mode="nearest")
        gate_dense_logits = self.gate_head(torch.cat((carrier_out, uncertainty_up), dim=1))
        gate_parent_logits = F.pixel_unshuffle(gate_dense_logits, 2).mean(dim=1, keepdim=True)
        learned_gate = torch.sigmoid(gate_parent_logits)
        if self.routing_enabled:
            gate_parent = parent_uncertainty * learned_gate
        else:
            gate_parent = torch.ones_like(parent)
        gate_dense = F.interpolate(gate_parent, size=expected_size, mode="nearest")

        dense_scores = self.score_head(
            torch.cat((carrier_out, boundary, parent_up), dim=1)
        )
        grouped_scores = F.pixel_unshuffle(dense_scores, downscale_factor=2)
        active = (
            parent_uncertainty >= self.uncertainty_threshold
            if self.routing_enabled
            else torch.ones_like(parent, dtype=torch.bool)
        )
        use_sparse_solver = (
            self.routing_enabled
            and self.hard_routing_inference
            and not self.training
            and not torch.is_grad_enabled()
        )
        if not self.occupancy_constraint:
            grouped_children = torch.sigmoid(grouped_scores)
            solver_active = torch.zeros_like(parent)
        elif use_sparse_solver:
            grouped_children = _routed_occupancy_solve(
                parent,
                grouped_scores,
                active,
                self.solver_iterations,
                self.epsilon,
            )
            solver_active = active.float()
        else:
            grouped_children = _OccupancyConstraint.apply(
                parent,
                grouped_scores,
                self.solver_iterations,
                self.epsilon,
            )
            solver_active = torch.ones_like(parent)
        refined = F.pixel_shuffle(grouped_children, upscale_factor=2)
        children = (1.0 - gate_dense) * parent_up + gate_dense * refined
        diagnostics = {
            "boundary": boundary,
            "gate_parent": gate_parent,
            "uncertainty_parent": parent_uncertainty,
            "active_fraction": solver_active.mean(),
            "active_parent": solver_active,
        }
        return children, carrier_out, diagnostics


class OccupancyUNet(CheckpointCompatibleModule):
    """Legacy five-stage U-Net with two high-resolution occupancy stages."""

    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: Sequence[int] = (16, 32, 64, 128, 256),
        guide_channels: int = 8,
        solver_iterations: int = 24,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        if len(encoder_channels) != 5:
            raise ValueError("encoder_channels must contain exactly five stages")
        c0, c1, c2, c3, c4 = (int(value) for value in encoder_channels)
        self.encoder_channels = (c0, c1, c2, c3, c4)
        self.encoder0 = DoubleConv(in_channels, c0)
        self.encoder1 = DoubleConv(c0, c1)
        self.encoder2 = DoubleConv(c1, c2)
        self.encoder3 = DoubleConv(c2, c3)
        self.encoder4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder3 = DecoderBlock(c4, c3, c3)
        self.decoder2 = DecoderBlock(c3, c2, c2)
        self.occupancy_head = nn.Conv2d(c2, 1, kernel_size=1)
        self.occupancy1 = OccupancyBlock(c1, guide_channels, solver_iterations, epsilon)
        self.occupancy2 = OccupancyBlock(c0, guide_channels, solver_iterations, epsilon)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.shape[-2] % 16 != 0 or image.shape[-1] % 16 != 0:
            raise ValueError("Input height and width must be divisible by 16")
        e0 = self.encoder0(image)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        d3 = self.decoder3(e4, e3)
        d2 = self.decoder2(d3, e2)
        logits88 = self.occupancy_head(d2)
        p88 = torch.sigmoid(logits88)
        p176 = self.occupancy1(p88, e1)
        p352 = self.occupancy2(p176, e0)
        return {
            "p352": p352,
            "p176": p176,
            "p88": p88,
            "logits88": logits88,
        }


class RoCUNet(CheckpointCompatibleModule):
    """RoCU-Net with two confidence-routed semantic occupancy stages."""

    def __init__(
        self,
        in_channels: int = 3,
        *,
        backbone: str = "mobilenet_v3_large",
        pretrained: bool = True,
        backbone_channels: Sequence[int] = (16, 24, 40, 80),
        backbone_depths: Sequence[int] = (1, 2, 2, 2),
        decoder_channels: Sequence[int] = (64, 48),
        carrier_channels: int = 24,
        shallow_guide_channels: int = 16,
        solver_iterations: int = 24,
        epsilon: float = 1e-6,
        routing_enabled: bool = True,
        hard_routing_inference: bool = True,
        uncertainty_threshold: float = 0.20,
        use_semantic_carrier: bool = True,
        occupancy_constraint: bool = True,
        carrier_upsampling: str = "pixelshuffle",
    ):
        super().__init__()
        if len(decoder_channels) != 2:
            raise ValueError("decoder_channels must contain [eighth, quarter] values")
        backbone_name = str(backbone).lower().replace("-", "_")
        if backbone_name in {"mobilenet_v3_large", "mobilenetv3_large", "mobilenetv3"}:
            if in_channels != 3:
                raise ValueError("The torchvision MobileNetV3 backbone requires in_channels=3")
            self.encoder = MobileNetV3LargeEncoder(pretrained=pretrained)
        elif backbone_name in {"efficient", "custom", "offline"}:
            self.encoder = EfficientEncoder(
                in_channels=in_channels,
                channels=backbone_channels,
                depths=backbone_depths,
            )
        else:
            raise ValueError(f"Unsupported RoCU backbone: {backbone}")
        c1, c2, c3, c4 = self.encoder.out_channels
        d3_channels, d2_channels = (int(value) for value in decoder_channels)
        carrier_channels = int(carrier_channels)
        shallow_guide_channels = int(shallow_guide_channels)

        self.input_guide = nn.Sequential(
            ConvBNSiLU(in_channels, shallow_guide_channels, kernel_size=3),
            DepthwiseSeparableBlock(shallow_guide_channels, shallow_guide_channels),
        )
        self.decoder3 = LiteDecoderBlock(c4, c3, d3_channels)
        self.decoder2 = LiteDecoderBlock(d3_channels, c2, d2_channels)
        self.coarse_head = nn.Conv2d(d2_channels, 1, kernel_size=1)
        self.carrier_head = ConvBNSiLU(
            d2_channels, carrier_channels, kernel_size=1, padding=0
        )
        block_kwargs = {
            "carrier_channels": carrier_channels,
            "solver_iterations": solver_iterations,
            "epsilon": epsilon,
            "routing_enabled": routing_enabled,
            "hard_routing_inference": hard_routing_inference,
            "uncertainty_threshold": uncertainty_threshold,
            "use_semantic_carrier": use_semantic_carrier,
            "occupancy_constraint": occupancy_constraint,
        }
        self.rocu1 = RoCUBlock(c1, **block_kwargs)
        self.rocu2 = RoCUBlock(shallow_guide_channels, **block_kwargs)
        self.backbone_name = backbone_name
        self.pretrained = bool(pretrained)
        self.carrier_channels = carrier_channels

        custom_modules = [
            self.input_guide,
            self.decoder3,
            self.decoder2,
            self.coarse_head,
            self.carrier_head,
            self.rocu1,
            self.rocu2,
        ]
        if isinstance(self.encoder, EfficientEncoder):
            custom_modules.append(self.encoder)
        for module in custom_modules:
            module.apply(self._initialize_weights)
        if self.coarse_head.bias is not None:
            nn.init.constant_(self.coarse_head.bias, -2.0)
        for block in (self.rocu1, self.rocu2):
            nn.init.constant_(block.boundary_head.bias, -1.0)
        if carrier_upsampling != "pixelshuffle":
            if occupancy_constraint or hard_routing_inference:
                raise ValueError("Upsampling controls require occupancy_constraint=false and dense inference")
            from .upsampling import build_carrier_upsampler
            # Install after all common layers are initialized. Preserve CPU RNG
            # state so unrelated initialization/loader RNG is not shifted.
            with torch.random.fork_rng(devices=[]):
                for block in (self.rocu1, self.rocu2):
                    block.carrier_upsampler = build_carrier_upsampler(carrier_upsampling, carrier_channels)
                    block.carrier_upsampling = carrier_upsampling
                    block.carrier_expansion = nn.Identity()  # Do not count unused PixelShuffle weights.

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.shape[-2] % 16 != 0 or image.shape[-1] % 16 != 0:
            raise ValueError("Input height and width must be divisible by 16")
        full_guide = self.input_guide(image)
        e1, e2, e3, e4 = self.encoder(image)
        d3 = self.decoder3(e4, e3)
        d2 = self.decoder2(d3, e2)
        logits_quarter = self.coarse_head(d2)
        p_quarter = torch.sigmoid(logits_quarter)
        carrier_quarter = self.carrier_head(d2)
        p_half, carrier_half, diagnostics1 = self.rocu1(
            p_quarter, carrier_quarter, e1
        )
        p_full, _, diagnostics2 = self.rocu2(p_half, carrier_half, full_guide)
        return {
            # Historical keys keep the complete train/evaluate pipeline compatible.
            "p352": p_full,
            "p176": p_half,
            "p88": p_quarter,
            "logits88": logits_quarter,
            "p_full": p_full,
            "p_half": p_half,
            "p_quarter": p_quarter,
            "logits_quarter": logits_quarter,
            "boundary_full": diagnostics2["boundary"],
            "boundary_half": diagnostics1["boundary"],
            "routing_gate_1": diagnostics1["gate_parent"],
            "routing_gate_2": diagnostics2["gate_parent"],
            "routing_uncertainty_1": diagnostics1["uncertainty_parent"],
            "routing_uncertainty_2": diagnostics2["uncertainty_parent"],
            "routing_fraction_1": diagnostics1["active_fraction"],
            "routing_fraction_2": diagnostics2["active_fraction"],
            "routing_active_1": diagnostics1["active_parent"],
            "routing_active_2": diagnostics2["active_parent"],
        }


def build_model(config: dict) -> nn.Module:
    cfg = normalize_config(config)["model"]
    architecture = cfg.get("architecture", "rocu_net")
    if architecture == "rocu_net":
        return RoCUNet(
            in_channels=int(cfg.get("in_channels", 3)),
            backbone=str(cfg.get("backbone", "mobilenet_v3_large")),
            pretrained=bool(cfg.get("pretrained", True)),
            backbone_channels=tuple(cfg.get("backbone_channels", (16, 24, 40, 80))),
            backbone_depths=tuple(cfg.get("backbone_depths", (1, 2, 2, 2))),
            decoder_channels=tuple(cfg.get("decoder_channels", (64, 48))),
            carrier_channels=int(cfg.get("carrier_channels", 24)),
            shallow_guide_channels=int(cfg.get("shallow_guide_channels", 16)),
            solver_iterations=int(cfg.get("solver_iterations", 24)),
            epsilon=float(cfg.get("epsilon", 1e-6)),
            routing_enabled=bool(cfg.get("routing_enabled", True)),
            hard_routing_inference=bool(cfg.get("hard_routing_inference", True)),
            uncertainty_threshold=float(cfg.get("routing_uncertainty_threshold", 0.20)),
            use_semantic_carrier=bool(cfg.get("use_semantic_carrier", True)),
            occupancy_constraint=bool(cfg.get("occupancy_constraint", True)),
            carrier_upsampling=str(cfg.get("carrier_upsampling", "pixelshuffle")),
        )
    if architecture != "occupancy_unet":
        raise ValueError(f"Unsupported model architecture: {architecture}")
    return OccupancyUNet(
        in_channels=int(cfg.get("in_channels", 3)),
        encoder_channels=tuple(cfg.get("encoder_channels", (16, 32, 64, 128, 256))),
        guide_channels=int(cfg.get("guide_channels", 8)),
        solver_iterations=int(cfg.get("solver_iterations", 24)),
        epsilon=float(cfg.get("epsilon", 1e-6)),
    )
