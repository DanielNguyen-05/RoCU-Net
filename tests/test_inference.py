import torch
from torch import nn

from rocu_net.inference import inference_model
from rocu_net.model import RoCUNet
from rocu_net.profiler import count_conv_linear_macs


def small_model():
    return RoCUNet(backbone="efficient", pretrained=False,
                   backbone_channels=(8, 12, 16, 24), backbone_depths=(1, 1, 1, 1),
                   decoder_channels=(20, 16), carrier_channels=8, shallow_guide_channels=8).eval()


def test_flip_tta_alignment_and_original_weights_unchanged():
    model = small_model()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(1, 3, 32, 48)
    with torch.inference_mode():
        plain = model(x)
        unchanged = inference_model(model, {})(x)
        for key in plain:
            assert torch.equal(plain[key], unchanged[key])
        outputs = inference_model(model, {}, "flip")(x)
        expected = sum(model(x.flip(d))["p352"].flip(d)
                       for d in ((), (-1,), (-2,), (-2, -1))) / 4
    assert torch.equal(outputs["p352"], expected)
    assert torch.allclose(nn.functional.avg_pool2d(outputs["p352"], 2), outputs["p176"], atol=3e-5)
    assert torch.allclose(nn.functional.avg_pool2d(outputs["p176"], 2), outputs["p88"], atol=3e-5)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    assert outputs["routing_active_2"].min() >= 0
    assert outputs["routing_active_2"].max() <= 1


def test_flip_profiling_counts_four_forward_passes():
    model = small_model()
    x = torch.randn(1, 3, 32, 32)
    assert count_conv_linear_macs(inference_model(model, {}, "flip"), x) == 4 * count_conv_linear_macs(model, x)
