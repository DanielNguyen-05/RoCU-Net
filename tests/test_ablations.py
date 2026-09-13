from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json

import torch
from torch import nn
from PIL import Image

from rocu_net.losses import MultiScaleOccupancyLoss
from rocu_net.model import RoCUBlock, build_model
from scripts.run_kvasir_ablations import suite_configs, validate_variants, prepare, collect, clean_config, sha256


def test_a1_direct_sigmoid_keeps_blending_and_backpropagates_without_solver():
    block = RoCUBlock(encoder_channels=6, carrier_channels=4,
                     occupancy_constraint=False, hard_routing_inference=False).eval()
    nn.init.zeros_(block.score_head[-1].weight)
    nn.init.zeros_(block.score_head[-1].bias)
    parent = torch.full((2, 1, 4, 4), 0.2, requires_grad=True)
    carrier, guide = torch.randn(2, 4, 4, 4), torch.randn(2, 6, 8, 8)
    with patch("rocu_net.model._OccupancyConstraint.apply", side_effect=AssertionError("Solver called")), \
         patch("rocu_net.model._routed_occupancy_solve", side_effect=AssertionError("Sparse solver called")):
        children, _, diagnostic = block(parent, carrier, guide)
        gate = nn.functional.interpolate(diagnostic["gate_parent"], scale_factor=2, mode="nearest")
        assert torch.allclose(children, (1 - gate) * 0.2 + gate * 0.5)
        assert (nn.functional.avg_pool2d(children, 2) - parent).abs().mean() > 0.01
        children.mean().backward()
    assert block.score_head[-1].bias.grad.abs().sum() > 0
    assert parent.grad is not None and torch.isfinite(parent.grad).all()
    assert diagnostic["active_fraction"] == 0


def test_a2_removes_carrier_information_but_keeps_guide_path():
    block = RoCUBlock(encoder_channels=6, carrier_channels=4,
                     use_semantic_carrier=False, hard_routing_inference=False).eval()
    parent = torch.full((2, 1, 4, 4), 0.4)
    carrier = torch.randn(2, 4, 4, 4, requires_grad=True)
    guide = torch.randn(2, 6, 8, 8, requires_grad=True)
    first, features, _ = block(parent, carrier, guide)
    second, other_features, _ = block(parent, torch.randn_like(carrier) * 10, guide)
    assert torch.equal(first, second)
    assert torch.equal(features, other_features)
    features.square().mean().backward()
    assert carrier.grad is None
    assert guide.grad is not None and guide.grad.abs().sum() > 0


def test_a3_uses_final_gate_one_and_omits_gate_supervision(tmp_path):
    cfg = suite_configs(tmp_path, smoke=True)[3]
    model = build_model(cfg).eval()
    outputs = model(torch.randn(2, 3, 64, 64))
    for stage in (1, 2):
        assert torch.equal(outputs[f"routing_gate_{stage}"], torch.ones_like(outputs[f"routing_gate_{stage}"]))
        assert outputs[f"routing_fraction_{stage}"] == 1
    loss, components = MultiScaleOccupancyLoss(boundary_weight=.25, routing_weight=0)(
        outputs, (torch.rand(2, 1, 64, 64) > .7).float())
    loss.backward()
    assert components["routing_loss"] == 0
    assert components["boundary_loss"] > 0
    assert all(p.grad is None for p in model.rocu1.gate_head.parameters())
    assert model.rocu1.boundary_head.weight.grad is not None


def test_all_variants_share_initial_weights_and_only_intended_config_changes(tmp_path):
    configs = suite_configs(tmp_path, smoke=True)
    states = []
    for cfg in configs:
        torch.manual_seed(42)
        model = build_model(cfg)
        states.append(model.state_dict())
        assert model.rocu1.occupancy_constraint == cfg['model']['occupancy_constraint']
        assert model.rocu2.occupancy_constraint == cfg['model']['occupancy_constraint']
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        assert all(torch.equal(v, states[0][k]) for k, v in state.items())
    broken = deepcopy(configs)
    broken[2]['training']['learning_rate'] *= 2
    try:
        validate_variants(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('Unintended second-factor changes must be rejected')


def test_dense_solver_reports_all_cells_active_even_when_confident():
    block = RoCUBlock(encoder_channels=6, carrier_channels=4, hard_routing_inference=False).eval()
    with torch.no_grad():
        children, _, diagnostics = block(torch.full((1, 1, 4, 4), .001),
                                        torch.randn(1, 4, 4, 4), torch.randn(1, 6, 8, 8))
    assert diagnostics['active_fraction'] == 1
    assert torch.allclose(nn.functional.avg_pool2d(children, 2), torch.full((1, 1, 4, 4), .001), atol=3e-5)


def test_suite_rejects_changed_masks_on_rerun(tmp_path):
    root = tmp_path / 'data'
    (root / 'images').mkdir(parents=True)
    (root / 'masks').mkdir()
    for i in range(12):
        Image.new('RGB', (32, 32), (i * 10, 20, 30)).save(root / 'images' / f'{i}.png')
        Image.new('L', (32, 32), 255).save(root / 'masks' / f'{i}.png')
    output = tmp_path / 'suite'
    configs = suite_configs(output, root, smoke=True)
    original = prepare(output, configs, True)
    assert prepare(output, configs, True) == original
    Image.new('L', (32, 32), 0).save(root / 'masks' / '0.png')
    try:
        prepare(output, configs, True)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed dataset content must invalidate a suite')
    assert json.loads((output / 'protocol.json').read_text()) == original


def test_table_rejects_test_results_from_a_replaced_checkpoint(tmp_path):
    configs = suite_configs(tmp_path, smoke=True)
    shared = tmp_path / 'splits'
    shared.mkdir()
    for split in ('train', 'val', 'test'):
        (shared / f'{split}.csv').write_text('sample_id,image,mask\n1,images/1.png,masks/1.png\n')
    protocol = {'smoke_test': True, 'split_sha256': {s: sha256(shared / f'{s}.csv') for s in ('train', 'val', 'test')},
                'split_sizes': {'train': 1, 'val': 1, 'test': 1}}
    metrics = {'n_images': 1, 'threshold': .5, 'inference': {'tta': 'none'},
               'mean': {'dice': .8, 'iou': .7, 'boundary_f1': .6},
               'occupancy_conservation': {'p352_to_p176_mae': 1e-7, 'p176_to_p88_mae': 2e-7}}
    for cfg in configs:
        run = tmp_path / cfg['experiment']['name']
        (run / 'splits').mkdir(parents=True)
        for split in ('train', 'val', 'test'):
            (run / 'splits' / f'{split}.csv').write_bytes((shared / f'{split}.csv').read_bytes())
        torch.save({'epoch': 1, 'config': clean_config(cfg)}, run / 'best.pt')
        (run / 'summary.json').write_text(json.dumps({'best_epoch': 1}))
        (run / 'test_metrics.json').write_text(json.dumps(metrics))
        (run / 'test_provenance.json').write_text(json.dumps({
            'checkpoint_sha256': sha256(run / 'best.pt'), 'split_sha256': protocol['split_sha256']['test'],
            'metrics_sha256': sha256(run / 'test_metrics.json')}))
    assert len(collect(tmp_path, configs, protocol, 'test')) == 4
    run = tmp_path / configs[0]['experiment']['name']
    torch.save({'epoch': 1, 'config': clean_config(configs[0]), 'changed_weights': True}, run / 'best.pt')
    try:
        collect(tmp_path, configs, protocol, 'test')
    except ValueError:
        pass
    else:
        raise AssertionError('Stale checkpoint/metric combinations must be rejected')
