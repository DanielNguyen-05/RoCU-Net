"""Audit saved splits, rectangular rotation damage and checkpoint predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocu_net.config import resolve_project_path
from ocu_net.data import _read_manifest, JointTransform, PolypSegDataset
from ocu_net.metrics import batch_metrics
from ocu_net.model import build_model
from ocu_net.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/rocu_cvc_colondb_seed42')
    parser.add_argument('--output', default='reports/colondb_audit')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    config = ck['config']
    root = resolve_project_path(config['data']['root'])
    splits = {s: _read_manifest(run / 'splits' / f'{s}.csv', root) for s in ('train', 'val', 'test')}
    ids = {s: {p.sample_id for p in pairs} for s, pairs in splits.items()}
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
    assert sum(map(len, ids.values())) == len(list((root / 'images').glob('*.png')))
    data_rows = []
    duplicate_groups = defaultdict(list)
    for split, pairs in splits.items():
        for p in pairs:
            with Image.open(p.image) as image, Image.open(p.mask) as source:
                mask = source.convert('L')
                assert image.size == mask.size
                a = np.asarray(mask) > 127
                image_hash = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
                duplicate_groups[image_hash].append({'split': split, 'id': p.sample_id,
                                                     'mask_hash': hashlib.sha256(a.tobytes()).hexdigest()})
                old_rotation = np.asarray(mask.rotate(90, resample=Image.Resampling.NEAREST)) > 127
                data_rows.append({'split': split, 'sample_id': p.sample_id,
                                  'foreground_fraction': float(a.mean()),
                                  'width': image.width, 'height': image.height,
                                  'rotation_90_retained_fraction': float(old_rotation.sum() / max(a.sum(), 1))})
    data = pd.DataFrame(data_rows)
    duplicates = [g for g in duplicate_groups.values() if len(g) > 1]
    save_json({'unique_decoded_images': len(duplicate_groups), 'duplicate_groups': duplicates,
               'cross_split_groups': [g for g in duplicates if len({p['split'] for p in g}) > 1],
               'conflicting_mask_groups': [g for g in duplicates if len({p['mask_hash'] for p in g}) > 1]},
              out / 'duplicates.json')
    data.to_csv(out / 'dataset_audit.csv', index=False)
    config['model']['pretrained'] = False
    model = build_model(config).eval()
    model.load_state_dict(ck['model'])
    records = []
    # Alternative inference settings are investigated on validation only.
    # Test rows describe the existing checkpoint, without tuning to the test set.
    with torch.inference_mode():
        for split, pairs in splits.items():
            loader = DataLoader(PolypSegDataset(pairs, JointTransform(tuple(config['data']['image_size']))), batch_size=4)
            for batch in tqdm(loader, desc=split):
                x, y = batch['image'], batch['mask']
                output = model(x)
                variants = {'baseline': output}
                if split == 'val':
                    for block in (model.crs1, model.crs2): block.hard_routing_inference = False
                    variants['dense_solver'] = model(x)
                    for block in (model.crs1, model.crs2): block.hard_routing_inference = config['model']['hard_routing_inference']
                for variant, result in variants.items():
                    metrics = batch_metrics(result['p352'].float(), y)
                    for i, sample_id in enumerate(batch['id']):
                        gt = y[i, 0] > .5
                        p = result['p352'][i, 0].float()
                        coarse = result['p88'][i:i+1].float()
                        coarse_gt = torch.nn.functional.adaptive_max_pool2d(y[i:i+1], coarse.shape[-2:]) > .5
                        records.append({'split': split, 'variant': variant, 'sample_id': sample_id,
                                        **{k: float(v[i]) for k, v in metrics.items()},
                                        'gt_probability_mean': float(p[gt].mean()),
                                        'prediction_max': float(p.max()),
                                        'predicted_pixels': int((p >= .5).sum()),
                                        'coarse_max_on_gt': float(coarse[coarse_gt].max()),
                                        'coarse_below_0125_on_gt': float((coarse[coarse_gt] < .125).float().mean())})
    frame = pd.DataFrame(records).merge(data, on=['split', 'sample_id'])
    frame.to_csv(out / 'checkpoint_audit.csv', index=False)
    summary = {'checkpoint_epoch': ck['epoch'], 'split_sizes': {s: len(p) for s,p in splits.items()},
               'rotation_90': {'train_masks_losing_area': int(((data.split == 'train') & (data.rotation_90_retained_fraction < .99)).sum()),
                               'train_masks_losing_over_half': int(((data.split == 'train') & (data.rotation_90_retained_fraction < .5)).sum())},
               'evaluation': {f'{split}/{variant}': g[['dice','iou','recall','precision','boundary_f1']].mean().to_dict()
                              for (split,variant),g in frame.groupby(['split','variant'])}}
    save_json(summary, out / 'audit_summary.json')
    print(summary)


if __name__ == '__main__':
    main()
