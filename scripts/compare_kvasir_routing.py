"""Compare dense vs hard routing on one trained A0 checkpoint, without retraining."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocu_net.compat import load_checkpoint
from rocu_net.config import resolve_project_path
from rocu_net.data import JointTransform, PolypSegDataset, _read_manifest
from rocu_net.engine import evaluate_model, save_per_image_records
from rocu_net.inference import inference_model
from rocu_net.losses import build_loss
from rocu_net.model import build_model
from rocu_net.profiler import profile_model
from rocu_net.utils import get_device, save_json, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument("--warmup-iterations", type=int, default=30)
    args = parser.parse_args()
    if args.benchmark_iterations < 1 or args.warmup_iterations < 0:
        parser.error("benchmark-iterations must be positive; warmup-iterations nonnegative")
    path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(path)
    config = checkpoint["config"]
    cfg_model = config["model"]
    if not all(cfg_model.get(k, True) for k in ("occupancy_constraint", "routing_enabled", "use_semantic_carrier")):
        raise ValueError("Routing comparison requires the full A0 architecture")
    root = resolve_project_path(args.data_root or config["data"]["root"])
    if not (root / config['data']['image_dir']).is_dir() and (root / 'Kvasir-SEG').is_dir():
        root = root / 'Kvasir-SEG'
    manifest = path.parent / "splits" / f"{args.split}.csv"
    pairs = _read_manifest(manifest, root)
    if not pairs or len({p.sample_id for p in pairs}) != len(pairs):
        raise ValueError("Evaluation requires a nonempty original split with unique IDs")
    device = get_device(args.device)
    output = (args.output or path.parent / f"routing_{args.split}").resolve()
    dataset = PolypSegDataset(pairs, JointTransform(tuple(config["data"]["image_size"])))
    loader = DataLoader(dataset, batch_size=int(config["training"].get("eval_batch_size", 8)),
                        num_workers=int(config["data"].get("num_workers", 0)), shuffle=False)
    report = {"checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "epoch": checkpoint["epoch"], "split": args.split,
              "split_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "routing_threshold": cfg_model.get("routing_uncertainty_threshold", .2),
              "threshold": config["training"]["threshold"], "tta": "none", "amp": False,
              "note": "Routing threshold is fixed from checkpoint config; no test-based threshold search.",
              "results": {}}
    rows = []
    for mode, hard in (("dense", False), ("hard", True)):
        model_config = deepcopy(config)
        model_config["model"].update(pretrained=False, hard_routing_inference=hard)
        model = build_model(model_config).to(device)
        model.load_state_dict(checkpoint["model"])
        model = inference_model(model, model_config, "none")
        metrics, records, _ = evaluate_model(model, loader, build_loss(config), device,
                                            threshold=config["training"]["threshold"], amp=False,
                                            description=f"{args.split}/{mode}")
        save_json(metrics, output / f"{mode}_metrics.json")
        save_per_image_records(records, output / f"{mode}_per_image.csv")
        # Use identical profiling input and device settings for the two modes.
        set_seed(int(config['experiment']['seed']))
        profile = profile_model(model, device, image_size=tuple(config["data"]["image_size"]),
                                batch_size=1, warmup_iterations=args.warmup_iterations,
                                benchmark_iterations=args.benchmark_iterations, checkpoint_path=path)
        report['results'][mode] = {"metrics": metrics, "profile": profile}
        routing = metrics['confidence_routing']
        image_h, image_w = config['data']['image_size']
        cells = ((image_h // 4) * (image_w // 4), (image_h // 2) * (image_w // 2))
        skipped = 1 - sum(c * routing[f'stage{i+1}_active_fraction'] for i, c in enumerate(cells)) / sum(cells)
        residual = metrics['occupancy_conservation']
        rows.append({"Mode": mode, "Split": args.split, "Dice": metrics['mean']['dice'],
                     "IoU": metrics['mean']['iou'], "Boundary F1": metrics['mean']['boundary_f1'],
                     "Occupancy MAE": (residual['p352_to_p176_mae'] + residual['p176_to_p88_mae']) / 2,
                     "Solver skip fraction (split)": skipped,
                     "Latency ms (batch=1)": profile['runtime']['latency_mean_ms_per_batch'],
                     "FPS (batch=1)": profile['runtime']['images_per_second']})
        del model
    save_json(report, output / 'routing_comparison.json')
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'routing_comparison.csv', index=False)
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in row) + ' |')
    (output / 'routing_comparison.md').write_text('\n'.join(lines) + '\n')
    print(frame.to_string(index=False))
    print(f'Saved routing comparison to {output}')


if __name__ == '__main__':
    main()
