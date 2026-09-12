"""Compare historical logged validation loss against current and legacy BCE."""
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocu_net.data import _read_manifest, PolypSegDataset, JointTransform
from ocu_net.losses import build_loss
from ocu_net.model import build_model
from ocu_net.utils import save_json


def legacy_bce(p, y, epsilon=1e-7):
    return F.binary_cross_entropy_with_logits(p.clamp(epsilon, 1-epsilon), y)


def legacy_structure(p, y, kernel_size=31, edge_weight=5, smooth=1):
    p = p.clamp(1e-7, 1-1e-7)
    w = 1 + edge_weight * (F.avg_pool2d(y, kernel_size, stride=1, padding=kernel_size//2) - y).abs()
    bce = (w * F.binary_cross_entropy_with_logits(p, y, reduction='none')).sum((2,3)) / w.sum((2,3))
    intersection = (p*y*w).sum((2,3))
    union = ((p+y)*w).sum((2,3))
    return (bce + 1 - (intersection+smooth)/(union-intersection+smooth)).mean()


def main():
    torch.set_num_threads(2)
    results = []
    for run_name, dataset in [('crs_ocu_kvasir_seed42','Kvasir-SEG'), ('crs_ocu_cvc_clinicdb_seed42','CVC-ClinicDB'), ('rocu_cvc_colondb_seed42','CVC-ColonDB')]:
        run = Path('runs') / run_name
        checkpoint = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        cfg['model']['pretrained'] = False
        model = build_model(cfg).eval()
        model.load_state_dict(checkpoint['model'])
        pairs = _read_manifest(run/'splits/val.csv', Path('dataset')/dataset)
        loader = DataLoader(PolypSegDataset(pairs, JointTransform(tuple(cfg['data']['image_size']))), batch_size=4)
        criterion = build_loss(cfg)
        sums = {'current': 0., 'legacy': 0.}
        with torch.inference_mode():
            for batch in loader:
                output = model(batch['image'])
                current, _ = criterion(output, batch['mask'])
                with patch('ocu_net.losses.probability_bce', legacy_bce), patch('ocu_net.losses.weighted_structure_loss', legacy_structure):
                    legacy, _ = criterion(output, batch['mask'])
                sums['current'] += float(current) * len(batch['image'])
                sums['legacy'] += float(legacy) * len(batch['image'])
        history = pd.read_csv(run/'history.csv')
        result = {'run': run_name, 'epoch': checkpoint['epoch'],
                  'logged_val_loss': float(history.loc[history.epoch==checkpoint['epoch'], 'val_loss'].iloc[0]),
                  **{k: value/len(pairs) for k,value in sums.items()}}
        print(result, flush=True)
        results.append(result)
    save_json({'validation_loss_comparison_cpu_fp32': results}, 'reports/colondb_audit/loss_version.json')


if __name__ == '__main__':
    main()
