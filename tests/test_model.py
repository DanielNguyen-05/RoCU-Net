import torch

from rocu_net.losses import MultiScaleOccupancyLoss
from rocu_net.model import RoCUNet, RoCUBlock, OccupancyUNet, OccupancyBlock, build_model


def test_model_shapes_and_cross_scale_conservation():
    model = OccupancyUNet(encoder_channels=(8, 16, 24, 32, 48), guide_channels=4, solver_iterations=28)
    model.eval()
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        outputs = model(image)
    assert outputs["p352"].shape == (2, 1, 64, 64)
    assert outputs["p176"].shape == (2, 1, 32, 32)
    assert outputs["p88"].shape == (2, 1, 16, 16)
    pooled_high = torch.nn.functional.avg_pool2d(outputs["p352"], 2)
    pooled_mid = torch.nn.functional.avg_pool2d(outputs["p176"], 2)
    assert torch.allclose(pooled_high, outputs["p176"], atol=2e-5, rtol=2e-5)
    assert torch.allclose(pooled_mid, outputs["p88"], atol=2e-5, rtol=2e-5)


def test_occupancy_backward_is_finite_and_reaches_both_inputs():
    block = OccupancyBlock(encoder_channels=7, guide_channels=3, solver_iterations=28)
    parent = torch.sigmoid(torch.randn(2, 1, 8, 8)).requires_grad_()
    guide = torch.randn(2, 7, 16, 16, requires_grad=True)
    output = block(parent, guide)
    weights = torch.randn_like(output)
    loss = (output * weights).sum()
    loss.backward()
    assert parent.grad is not None and torch.isfinite(parent.grad).all()
    assert guide.grad is not None and torch.isfinite(guide.grad).all()
    assert parent.grad.abs().sum() > 0
    assert guide.grad.abs().sum() > 0


def test_multiscale_loss_backward():
    model = OccupancyUNet(encoder_channels=(8, 16, 24, 32, 48), guide_channels=4)
    image = torch.randn(1, 3, 64, 64)
    target = (torch.rand(1, 1, 64, 64) > 0.7).float()
    outputs = model(image)
    loss, components = MultiScaleOccupancyLoss()(outputs, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(components) == {
        "loss",
        "final_loss",
        "final_bce",
        "final_dice_loss",
        "p176_loss",
        "p88_loss",
        "structure_loss",
        "boundary_loss",
        "routing_loss",
    }


def test_rocu_model_shapes_conservation_and_diagnostics():
    model = RoCUNet(
        backbone="efficient",
        pretrained=False,
        backbone_channels=(8, 12, 16, 24),
        backbone_depths=(1, 1, 1, 1),
        decoder_channels=(20, 16),
        carrier_channels=8,
        shallow_guide_channels=8,
        solver_iterations=28,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 64, 64))
    assert outputs["p352"].shape == (2, 1, 64, 64)
    assert outputs["p176"].shape == (2, 1, 32, 32)
    assert outputs["p88"].shape == (2, 1, 16, 16)
    assert outputs["boundary_full"].shape == (2, 1, 64, 64)
    assert outputs["routing_gate_1"].shape == (2, 1, 16, 16)
    assert outputs["routing_gate_2"].shape == (2, 1, 32, 32)
    assert torch.allclose(
        torch.nn.functional.avg_pool2d(outputs["p352"], 2),
        outputs["p176"],
        atol=3e-5,
        rtol=3e-5,
    )
    assert torch.allclose(
        torch.nn.functional.avg_pool2d(outputs["p176"], 2),
        outputs["p88"],
        atol=3e-5,
        rtol=3e-5,
    )


def test_rocu_confident_cells_take_identity_route_at_inference():
    block = RoCUBlock(
        encoder_channels=6,
        carrier_channels=4,
        solver_iterations=24,
        uncertainty_threshold=0.20,
    )
    block.eval()
    parent = torch.full((1, 1, 4, 4), 0.01)
    carrier = torch.randn(1, 4, 4, 4)
    guide = torch.randn(1, 6, 8, 8)
    with torch.no_grad():
        children, _, diagnostics = block(parent, carrier, guide)
    expected = torch.nn.functional.interpolate(parent, scale_factor=2, mode="nearest")
    assert torch.allclose(children, expected, atol=1e-7, rtol=1e-7)
    assert float(diagnostics["active_fraction"]) == 0.0


def test_build_model_selects_rocu_architecture():
    config = {
        "model": {
            "architecture": "rocu_net",
            "backbone": "efficient",
            "pretrained": False,
            "backbone_channels": [8, 12, 16, 24],
            "backbone_depths": [1, 1, 1, 1],
            "decoder_channels": [20, 16],
            "carrier_channels": 8,
            "shallow_guide_channels": 8,
        }
    }
    assert isinstance(build_model(config), RoCUNet)
