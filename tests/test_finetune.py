import torch
from unittest.mock import patch
from rocu_net.engine import train_one_epoch
from rocu_net.losses import MultiScaleOccupancyLoss
from rocu_net.model import RoCUNet
from rocu_net.utils import build_grad_scaler


def test_freeze_encoder_bn_keeps_statistics_and_trains_weights():
    model = RoCUNet(backbone="efficient", pretrained=False,
                     backbone_channels=(8, 12, 16, 24), backbone_depths=(1, 1, 1, 1),
                     decoder_channels=(20, 16), carrier_channels=8, shallow_guide_channels=8)
    bn = next(m for m in model.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d))
    mean, var, count = bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone()
    parameter = next(model.encoder.parameters())
    before = parameter.detach().clone()
    batch = {"image": torch.randn(2, 3, 32, 32), "mask": (torch.rand(2, 1, 32, 32) > .8).float()}
    loss = MultiScaleOccupancyLoss(tversky_weight=.3)
    result = train_one_epoch(model, [batch], loss, torch.optim.AdamW(model.parameters(), lr=1e-3),
                            build_grad_scaler(False), torch.device('cpu'), amp=False, freeze_encoder_bn=True)
    assert result['loss'] > 0
    assert torch.equal(mean, bn.running_mean) and torch.equal(var, bn.running_var)
    assert torch.equal(count, bn.num_batches_tracked)
    assert not torch.equal(before, parameter)


def test_multiscale_training_resizes_images_and_binary_masks_together():
    model = RoCUNet(backbone="efficient", pretrained=False,
                   backbone_channels=(8, 12, 16, 24), backbone_depths=(1, 1, 1, 1),
                   decoder_channels=(20, 16), carrier_channels=8, shallow_guide_channels=8)
    batch = {"image": torch.randn(2, 3, 64, 64), "mask": (torch.rand(2, 1, 64, 64) > .8).float()}
    seen = []
    loss_fn = MultiScaleOccupancyLoss()
    def checked_loss(outputs, target):
        seen.append(target.shape[-2:])
        assert outputs["p352"].shape == target.shape
        assert set(target.unique().tolist()) <= {0.0, 1.0}
        return loss_fn(outputs, target)
    with patch("rocu_net.engine.random.choice", return_value=1.25):
        train_one_epoch(model, [batch], checked_loss, torch.optim.AdamW(model.parameters(), lr=1e-4),
                        build_grad_scaler(False), torch.device('cpu'), amp=False,
                        freeze_encoder_bn=True, multi_scale_factors=(1.0, 1.25))
    assert seen == [(80, 80)]
