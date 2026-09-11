"""Export real checkpoint predictions and training history as PNG/PDF paper figures."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from ocu_net.config import load_config, resolve_project_path
from ocu_net.data import JointTransform, discover_pairs, _read_manifest
from ocu_net.metrics import batch_metrics
from ocu_net.model import build_model
from ocu_net.utils import autocast_context, get_device, save_json, set_seed
from ocu_net.visualization import diagnostics, metric_figures, qualitative, select_samples, training_figure


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--history", type=Path, help="history.csv or captured training .log; can be used alone")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", help="Relocate the checkpoint's dataset")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--external-config", help="Evaluate ALL images of a different dataset; no splits are created")
    parser.add_argument("--selection", choices=["quantiles", "random", "best", "worst"], default="quantiles")
    parser.add_argument("--sample-ids", nargs="+", help="Explicit sample IDs override selection")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    if not args.checkpoint and not args.history: parser.error("Provide --checkpoint and/or --history")
    if args.num_samples < 1 or args.dpi < 72: parser.error("num-samples must be positive and dpi >= 72")
    if args.external_config and not args.checkpoint: parser.error("--external-config requires --checkpoint")
    return args


def load_evaluation_pairs(config, run_dir, split, external_config=None, data_root=None):
    data = load_config(external_config)["data"] if external_config else config["data"]
    root = resolve_project_path(data_root or data["root"])
    if external_config:
        return discover_pairs(root, data.get("image_dir", "images"), data.get("mask_dir", "masks"))
    manifest = run_dir / "splits" / f"{split}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing original evaluation manifest: {manifest}. Copy the run's splits directory; paper evaluation must not silently recreate a split.")
    image_dir = data.get("image_dir", "images")
    if not (root / image_dir).is_dir() and (root / "Kvasir-SEG" / image_dir).is_dir():
        root = root / "Kvasir-SEG"
    pairs = _read_manifest(manifest, root)
    ids = [p.sample_id for p in pairs]
    if not pairs or len(ids) != len(set(ids)):
        raise ValueError("Evaluation manifest must contain nonempty, unique sample IDs")
    return pairs


def export_checkpoint(args):
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    pairs = load_evaluation_pairs(config, checkpoint_path.parent, args.split, args.external_config, args.data_root)
    model_config = {**config, "model": {**config["model"], "pretrained": False}}
    device = get_device(args.device)
    set_seed(args.seed)
    model = build_model(model_config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    transform = JointTransform(tuple(config["data"]["image_size"]), train=False)
    threshold = float(config["training"].get("threshold", 0.5))
    amp = bool(config["training"].get("amp", True) and device.type == "cuda")
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rocu-figures-") as cache:
        cache_paths = {}
        with torch.inference_mode():
            for index, pair in enumerate(tqdm(pairs, desc="Paper evaluation")):
                with Image.open(pair.image) as handle: image = handle.convert("RGB")
                with Image.open(pair.mask) as handle: mask = handle.convert("L")
                x, y = transform(image, mask)
                target = y.unsqueeze(0).to(device)
                with autocast_context(amp): outputs = model(x.unsqueeze(0).to(device))
                probability = outputs["p352"].float()
                metrics = batch_metrics(probability, target, threshold=threshold)
                residual = (F.avg_pool2d(probability, 2) - outputs["p176"].float()).abs()
                row = {"sample_id": pair.sample_id, **{k: float(v[0]) for k,v in metrics.items()},
                       "foreground_fraction": float(y.mean()),
                       "occupancy_mae_stage2": float(residual.mean()),
                       "occupancy_mae_stage1": float((F.avg_pool2d(outputs['p176'].float(), 2)-outputs['p88'].float()).abs().mean())}
                h, w = y.shape[-2:]
                sample = {"image": np.asarray(image.resize((w,h), Image.Resampling.BILINEAR)),
                          "target": y[0].numpy().astype(bool), "probability": probability[0,0].cpu().numpy(),
                          "residual": residual[0,0].cpu().numpy()}
                if "routing_gate_2" in outputs:
                    block = model.crs2
                    active = (outputs["routing_uncertainty_2"] >= block.uncertainty_threshold
                              if block.routing_enabled and block.hard_routing_inference
                              else torch.ones_like(outputs["routing_uncertainty_2"], dtype=torch.bool))
                    sample.update(routing=outputs["routing_gate_2"][0,0].float().cpu().numpy(),
                                  boundary=outputs["boundary_full"][0,0].float().cpu().numpy(),
                                  active=active[0,0].cpu().numpy())
                    row["solver_active_fraction_stage2"] = float(active.float().mean())
                records.append(row)
                cache_paths[pair.sample_id] = Path(cache) / f"{index}.npz"
                np.savez_compressed(cache_paths[pair.sample_id], **sample)
        frame = pd.DataFrame(records)
        frame.to_csv(args.output / "per_image_metrics.csv", index=False)
        selected_ids = args.sample_ids or select_samples(frame, args.num_samples, args.selection, args.seed)
        if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) - set(cache_paths):
            raise ValueError("Requested sample IDs must be unique and present in the evaluated split")
        selected = []
        indexed = frame.set_index("sample_id")
        for sample_id in selected_ids:
            with np.load(cache_paths[sample_id]) as arrays: sample = dict(arrays)
            sample.update(id=sample_id, threshold=threshold, dice=float(indexed.loc[sample_id,'dice']), iou=float(indexed.loc[sample_id,'iou']))
            selected.append(sample)
        selection = "explicit IDs" if args.sample_ids else args.selection
        protocol = "external dataset (all images)" if args.external_config else f"{args.split} split"
        # Paginate long requests to keep text readable in a two-column paper.
        for start in range(0, len(selected), 6):
            page = selected[start:start+6]
            suffix = f"_{start//6+1:02d}"
            qualitative(page, args.output / f"qualitative{suffix}", f"RoCU-Net | {protocol} | {selection} selection", args.dpi)
            diagnostics(page, args.output / f"mechanism{suffix}", args.dpi)
        metric_figures(frame, args.output, args.dpi)
        numeric = frame.drop(columns="sample_id")
        with checkpoint_path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
        save_json({"checkpoint": str(checkpoint_path), "checkpoint_sha256": digest,
                   "checkpoint_epoch": checkpoint.get("epoch"), "protocol": protocol,
                   "external_config": args.external_config, "selection": selection,
                   "selected_ids": selected_ids, "selection_seed": args.seed,
                   "image_size": config['data']['image_size'], "threshold": threshold,
                   "amp": amp, "device": str(device), "n_images": len(frame),
                   "mean": numeric.mean().to_dict(), "std": numeric.std(ddof=0).to_dict()}, args.output / "figure_metadata.json")
        frame.set_index('sample_id').loc[selected_ids].to_csv(args.output / "selected_samples.csv")
        (args.output / "captions.txt").write_text(
            f"Qualitative segmentation on {protocol}. Rows selected using {selection} (seed {args.seed}); sample IDs and per-image metrics accompany this figure. "
            f"Input and masks are shown at the evaluation resolution {config['data']['image_size']}; threshold={threshold}. "
            "Columns show input, ground truth, RoCU-Net prediction, contour detail cropped to the union of target and prediction, TP/FP/FN errors, and foreground probability. "
            "The rectangle in the input defines the contour crop. TP green, FP orange, FN purple, TN black; GT contour blue, prediction contour orange.\n\n"
            "Mechanism: auxiliary boundary probability, soft routing gate, solver-active parent cells and absolute |avgpool(P_full)-P_half|. "
            "Routing maps are internal diagnostics, not calibrated confidence or an explanation of causality. Solver-active cells depend on the checkpoint's inference routing settings. "
            "These maps illustrate routing and conservation; speed gains require timed comparisons and ablations.\n\n"
            "Metric distribution: all evaluated images, including failure cases; no filtering by quality. Foreground area is measured from resized ground truth. "
            "Boxplots show median, quartiles, whiskers/outliers and mean triangles. No baseline superiority claim is established by these figures alone.\n")
    history = checkpoint_path.parent / "history.csv"
    if history.is_file() and not args.history: training_figure(history, args.output, args.dpi)
    print(f"Exported {len(pairs)} image metrics and {len(selected)} visualized cases to {args.output}")


def main():
    args = parse_args()
    if args.history: training_figure(args.history, args.output, args.dpi)
    if args.checkpoint: export_checkpoint(args)


if __name__ == "__main__":
    main()
