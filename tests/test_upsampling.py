from copy import deepcopy

import torch
from torch.nn import functional as F

from rocu_net.losses import build_loss
from rocu_net.model import build_model
from rocu_net.upsampling import CARAFE, DySampleLP
from scripts.run_kvasir_ablations import suite_configs, validate_variants


def test_carafe_matches_explicit_neighborhood_reassembly_and_gradients():
    torch.manual_seed(2)
    fast = CARAFE(4, kernel_size=3, compressed_channels=4).double()
    slow = deepcopy(fast)
    x = torch.randn(2, 4, 3, 5, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    actual = fast(x)
    masks = F.pixel_shuffle(slow.content_encoder(slow.channel_compressor(other)), 2).softmax(dim=1)
    padded = F.pad(other, (1, 1, 1, 1))
    expected = torch.zeros_like(actual)
    for ky in range(3):
        for kx in range(3):
            neighbors = padded[:, :, ky:ky + 3, kx:kx + 5]
            expected = expected + F.interpolate(neighbors, scale_factor=2, mode="nearest") * masks[:, ky * 3 + kx:ky * 3 + kx + 1]
    assert torch.allclose(actual, expected, atol=1e-12)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    assert torch.allclose(x.grad, other.grad, atol=1e-12)
    for a, b in zip(fast.parameters(), slow.parameters()):
        assert a.grad is not None and torch.allclose(a.grad, b.grad, atol=1e-11)


def test_dysample_zero_offset_equals_bilinear_and_group_offsets_have_correct_direction():
    model = DySampleLP(8, groups=4)
    torch.nn.init.zeros_(model.offset.weight)
    torch.nn.init.zeros_(model.offset.bias)
    x = torch.randn(2, 8, 3, 5)
    assert torch.allclose(model(x), F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), atol=2e-6)
    with torch.no_grad():
        # First four entries are the four children's X offsets for group zero.
        model.offset.bias[:4] = 1.
    yy, xx = torch.meshgrid((torch.arange(6) + .5) / 2,
                            (torch.arange(10) + .5) / 2, indexing="ij")
    grid = torch.stack((2 * (xx + .25) / 5 - 1, 2 * yy / 3 - 1), dim=-1)
    expected_first_group = F.grid_sample(x[:, :2], grid[None].expand(2, -1, -1, -1),
                                         padding_mode="border", align_corners=False)
    actual = model(x)
    assert torch.allclose(actual[:, :2], expected_first_group, atol=2e-6)
    assert torch.allclose(actual[:, 2:], F.interpolate(x[:, 2:], scale_factor=2, mode="bilinear", align_corners=False), atol=2e-6)
    actual.square().mean().backward()
    assert model.offset.weight.grad.abs().sum() > 0


def test_shared_weights_rng_and_pixelshuffle_equivalence_to_a1(tmp_path):
    configs = suite_configs(tmp_path, smoke=True, suite="upsampling")
    states, rngs = [], []
    for cfg in configs:
        torch.manual_seed(42)
        model = build_model(cfg)
        states.append(model.state_dict())
        rngs.append(torch.random.get_rng_state())
    for state, rng in zip(states[1:], rngs[1:]):
        for name, value in states[0].items():
            if ".carrier_expansion." not in name:
                assert name in state and torch.equal(value, state[name]), name
        assert torch.equal(rngs[0], rng)
    a1_cfg = suite_configs(tmp_path, smoke=True)[1]
    torch.manual_seed(42)
    a1 = build_model(a1_cfg).eval()
    torch.manual_seed(42)
    u2 = build_model(configs[2]).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(a1(x)["p_full"], u2(x)["p_full"])


def test_all_upsamplers_train_with_guide_gate_and_multiscale_losses(tmp_path):
    configs = suite_configs(tmp_path, smoke=True, suite="upsampling")
    image, target = torch.randn(2, 3, 64, 64), (torch.rand(2, 1, 64, 64) > .6).float()
    for index, cfg in enumerate(configs):
        model = build_model(cfg).train()
        outputs = model(image)
        loss, components = build_loss(cfg)(outputs, target)
        loss.backward()
        assert torch.isfinite(loss) and components['p176_loss'] > 0 and components['p88_loss'] > 0
        for block in (model.rocu1, model.rocu2):
            assert block.guide_projection[0].weight.grad.abs().sum() > 0
            assert block.gate_head[-1].weight.grad.abs().sum() > 0
            assert block.boundary_head.weight.grad.abs().sum() > 0
        assert outputs['routing_active_1'].sum() == (outputs['p_quarter'].numel() if index == 0 else 0)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_upsampling_suite_rejects_changed_loss_or_training_protocol(tmp_path):
    configs = suite_configs(tmp_path, smoke=True, suite="upsampling")
    for section, key in (("loss", "routing_weight"), ("training", "learning_rate")):
        wrong = deepcopy(configs)
        wrong[3][section][key] += .01
        try:
            validate_variants(wrong, "upsampling")
        except ValueError:
            pass
        else:
            raise AssertionError("A second changed factor must be rejected")
