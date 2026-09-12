"""Compare per-image and LV-UNet-style batch pooling on unchanged binary masks."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocu_net.config import load_config, resolve_project_path
from ocu_net.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.run / 'config_resolved.yaml')
    split = pd.read_csv(args.run / 'splits/test.csv', dtype=str)
    counts = []
    for row in split.itertuples():
        with Image.open(args.run / 'test_predictions/binary' / f'{row.sample_id}.png') as im:
            p = np.asarray(im) > 127
        with Image.open(args.data_root / row.mask) as im:
            g = np.asarray(im.convert('L').resize((p.shape[1], p.shape[0]), Image.Resampling.NEAREST)) > 127
        counts.append([int((p & g).sum()), int((p | g).sum())])
    counts = np.asarray(counts)
    macro_iou = (counts[:,0] + 1e-5) / (counts[:,1] + 1e-5)
    batch_size = int(config['training'].get('eval_batch_size', config['training']['batch_size']))
    batches = []
    for start in range(0, len(counts), batch_size):
        chunk = counts[start:start + batch_size]
        iou = (chunk[:,0].sum() + 1e-5) / (chunk[:,1].sum() + 1e-5)
        batches.append([len(chunk), iou, 2 * iou / (1 + iou)])
    batches = np.asarray(batches)
    pooled = np.average(batches[:,1:], axis=0, weights=batches[:,0])
    result = {'run': str(args.run), 'n_images': len(counts), 'batch_size': batch_size,
              'order': 'test.csv order; batch-pooling results depend on this order',
              'macro_per_image': {'iou': float(macro_iou.mean()), 'dice': float((2 * macro_iou / (1 + macro_iou)).mean())},
              'batch_pooled': {'iou': float(pooled[0]), 'dice': float(pooled[1])},
              'note': 'Same saved predictions. Metric aggregation difference, not a model improvement or a matched LV-UNet benchmark.'}
    save_json(result, args.output)
    print(result)


if __name__ == '__main__':
    main()
