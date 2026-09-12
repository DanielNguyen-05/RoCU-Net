from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rocu_net.compat import load_checkpoint
from rocu_net.config import resolve_project_path
from rocu_net.data import build_dataloaders, create_or_load_splits
from rocu_net.engine import evaluate_model, save_per_image_records
from rocu_net.losses import build_loss
from rocu_net.model import build_model
from rocu_net.profiler import profile_model
from rocu_net.utils import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a RoCU-Net checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    device = get_device(args.device)
    seed = int(config["experiment"]["seed"])
    set_seed(seed, deterministic=bool(config["experiment"].get("deterministic", False)))
    run_dir = checkpoint_path.parent
    data_cfg = config["data"]
    dataset_root = resolve_project_path(data_cfg["root"])
    splits = create_or_load_splits(
        dataset_root,
        run_dir / "splits",
        image_dir=data_cfg.get("image_dir", "images"),
        mask_dir=data_cfg.get("mask_dir", "masks"),
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        test_ratio=float(data_cfg.get("test_ratio", 0.1)),
        seed=seed,
    )
    loaders = build_dataloaders(config, splits, training=False)
    # The checkpoint already contains encoder weights; avoid an ImageNet download.
    model_config = {**config, "model": {**config["model"], "pretrained": False}}
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    loss_fn = build_loss(config)
    threshold = float(config["training"].get("threshold", 0.5))
    prediction_dir = run_dir / f"{args.split}_predictions" if args.save_predictions else None
    metrics, records, losses = evaluate_model(
        model,
        loaders[args.split],
        loss_fn,
        device,
        threshold=threshold,
        amp=bool(config["training"].get("amp", True)),
        description=args.split,
        prediction_dir=prediction_dir,
    )
    metrics["loss"] = losses
    save_json(metrics, run_dir / f"{args.split}_metrics.json")
    save_per_image_records(records, run_dir / f"{args.split}_per_image.csv")
    print(f"{args.split}: Dice={metrics['mean']['dice']:.4f}, IoU={metrics['mean']['iou']:.4f}")

    if args.profile:
        profile_cfg = config.get("profiling", {})
        lightweight = profile_model(
            model,
            device,
            image_size=tuple(int(v) for v in data_cfg["image_size"]),
            batch_size=int(profile_cfg.get("batch_size", 1)),
            in_channels=int(config["model"].get("in_channels", 3)),
            warmup_iterations=int(profile_cfg.get("warmup_iterations", 30)),
            benchmark_iterations=int(profile_cfg.get("benchmark_iterations", 100)),
            checkpoint_path=checkpoint_path,
        )
        save_json(lightweight, run_dir / "lightweight_metrics.json")
        print(
            f"Params={lightweight['parameters']['millions']:.3f}M, "
            f"GMACs={lightweight['computation']['gmacs_per_image']:.3f}, "
            f"FPS={lightweight['runtime']['images_per_second']:.2f}"
        )


if __name__ == "__main__":
    main()
