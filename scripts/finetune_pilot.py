"""Short CPU validation-only trial of a fine-tuning config; never evaluates test."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocu_net.config import load_config, resolve_project_path, save_config
from ocu_net.data import build_dataloaders, create_or_load_splits
from ocu_net.engine import evaluate_model, train_one_epoch
from ocu_net.losses import build_loss
from ocu_net.model import build_model
from ocu_net.utils import set_seed, build_grad_scaler, save_json
from train import make_optimizer, make_scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/cvc_colondb_finetune.yaml')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('reports/colondb_pilot'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = load_config(args.config)
    cfg['data'].update(num_workers=0, pin_memory=False)
    cfg['training']['amp'] = False
    device = torch.device('cpu')
    set_seed(cfg['experiment']['seed'])
    args.output.mkdir(parents=True, exist_ok=True)
    splits = create_or_load_splits(resolve_project_path(cfg['data']['root']), args.output / 'splits',
                                  source_split_dir=resolve_project_path(cfg['data']['split_source']))
    loaders = build_dataloaders(cfg, splits)
    model_cfg = {**cfg, 'model': {**cfg['model'], 'pretrained': False}}
    model = build_model(model_cfg)
    checkpoint = torch.load(resolve_project_path(cfg['training']['init_checkpoint']), map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    loss_fn = build_loss(cfg)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    scaler = build_grad_scaler(False)
    records = []
    save_config(cfg, args.output / 'config_resolved.yaml')
    for epoch in range(args.epochs + 1):
        train_losses = None
        if epoch:
            train_losses = train_one_epoch(model, loaders['train'], loss_fn, optimizer, scaler, device,
                                            amp=False, epoch=epoch, grad_clip_norm=cfg['training']['grad_clip_norm'],
                                            freeze_encoder_bn=cfg['training'].get('freeze_encoder_bn', False))
            if scheduler: scheduler.step()
        metrics, _, _ = evaluate_model(model, loaders['val'], loss_fn, device, amp=False, description=f'pilot val {epoch}')
        records.append({'epoch': epoch, 'validation': metrics['mean'], 'train': train_losses})
        save_json({'config': args.config, 'records': records, 'test_evaluated': False}, args.output / 'pilot_results.json')
        print(f'Pilot epoch {epoch}: validation Dice={metrics["mean"]["dice"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
