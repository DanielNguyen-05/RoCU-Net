import torch
from ocu_net.losses import probability_bce, weighted_structure_loss


def test_probability_bce_matches_reference_and_gradients():
    p = torch.tensor([0.1, 0.8, 0.35], requires_grad=True)
    target = torch.tensor([0.0, 1.0, 0.5])
    actual = probability_bce(p, target)
    expected = torch.nn.functional.binary_cross_entropy(p, target)
    assert torch.allclose(actual, expected)
    assert torch.allclose(torch.autograd.grad(actual, p)[0], torch.autograd.grad(expected, p)[0])


def test_structure_loss_rewards_correct_confident_predictions():
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1
    good = (target * 0.98 + 0.01).requires_grad_()
    bad = 1 - good.detach()
    loss = weighted_structure_loss(good, target)
    assert loss < 0.05
    assert loss < weighted_structure_loss(bad, target)
    loss.backward()
    assert torch.isfinite(good.grad).all()
