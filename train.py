from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from rocu_net.compat import load_checkpoint
from rocu_net.config import apply_common_overrides, load_config, resolve_project_path, save_config
from rocu_net.data import build_dataloaders, create_or_load_splits
from rocu_net.engine import (
    evaluate_model,
    plot_history,
    save_history,
    save_per_image_records,
    train_one_epoch,
)
from rocu_net.losses import build_loss
from rocu_net.model import build_model
from rocu_net.inference import inference_model
from rocu_net.profiler import profile_model
from rocu_net.utils import (
    atomic_torch_save,
    build_grad_scaler,
    get_device,
    implementation_fingerprint,
    save_json,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RoCU-Net on paired polyp segmentation datasets")
    parser.add_argument("--config", default="configs/kvasir.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", default=None, help="Path to last.pt or another training checkpoint")
    initialization.add_argument("--init-checkpoint", default=None, help="Load model weights only into a new run; reset optimizer/epoch")
    parser.add_argument("--no-test", action="store_true", help="Do not evaluate the held-out test set")
    return parser.parse_args()


def make_optimizer(model: torch.nn.Module, config: dict):
    cfg = config["training"]
    name = str(cfg.get("optimizer", "adamw")).lower()
    learning_rate = float(cfg["learning_rate"])
    kwargs = {"lr": learning_rate, "weight_decay": float(cfg.get("weight_decay", 0.0))}
    parameters: object = model.parameters()
    backbone_multiplier = float(cfg.get("backbone_lr_multiplier", 1.0))
    if backbone_multiplier != 1.0 and hasattr(model, "encoder"):
        backbone_parameters: list[torch.nn.Parameter] = []
        task_parameters: list[torch.nn.Parameter] = []
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter_name.startswith("encoder."):
                backbone_parameters.append(parameter)
            else:
                task_parameters.append(parameter)
        parameters = [
            {"params": backbone_parameters, "lr": learning_rate * backbone_multiplier},
            {"params": task_parameters, "lr": learning_rate},
        ]
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def make_scheduler(optimizer, config: dict):
    cfg = config["training"]
    name = str(cfg.get("scheduler", "cosine")).lower()
    if name == "cosine":
        epochs = int(cfg["epochs"])
        warmup_epochs = int(cfg.get("warmup_epochs", 0))
        minimum = float(cfg.get("min_learning_rate", 1e-6))
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=float(cfg.get("warmup_start_factor", 0.1)),
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=minimum,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=minimum
        )
    if name in {"none", "off"}:
        return None
    raise ValueError(f"Unsupported scheduler: {name}")


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    config: dict,
    epoch: int,
    best_dice: float,
    epochs_without_improvement: int,
) -> dict:
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "best_val_dice": float(best_dice),
        "epochs_without_improvement": int(epochs_without_improvement),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "implementation": implementation_fingerprint(),
    }


def main() -> None:
    args = parse_args()
    config = apply_common_overrides(
        load_config(args.config), data_root=args.data_root, name=args.name, seed=args.seed
    )
    if args.init_checkpoint is not None:
        config["training"]["init_checkpoint"] = args.init_checkpoint
    device = get_device(args.device)
    seed = int(config["experiment"]["seed"])
    set_seed(seed, deterministic=bool(config["experiment"].get("deterministic", False)))

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        run_dir = resume_path.parent
    else:
        output_root = resolve_project_path(config["experiment"].get("output_dir", "runs"))
        run_dir = output_root / str(config["experiment"]["name"])
        resume_path = None
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume_path is None and any((run_dir / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError(f"Run already contains checkpoints: {run_dir}. Use --resume or a new --name.")
    logger = setup_logger(run_dir / "train.log")
    save_config(config, run_dir / "config_resolved.yaml")

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
        source_split_dir=(resolve_project_path(data_cfg["split_source"]) if data_cfg.get("split_source") else None),
        group_identical_images=bool(data_cfg.get("group_identical_images", False)),
    )
    loaders = build_dataloaders(config, splits)
    logger.info(
        "Device=%s | train=%d val=%d test=%d | run=%s",
        device,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        run_dir,
    )

    init_path = config["training"].get("init_checkpoint")
    model_config = config
    if resume_path is not None or init_path:
        model_config = {**config, "model": {**config["model"], "pretrained": False}}
    model = build_model(model_config).to(device)
    if init_path and resume_path is None:
        init_path = resolve_project_path(init_path)
        model.load_state_dict(load_checkpoint(init_path)["model"], strict=True)
        logger.info("Initialized model weights from %s; optimizer and epoch start fresh", init_path)
    evaluation_model = inference_model(model, config)
    loss_fn = build_loss(config)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    train_cfg = config["training"]
    amp_enabled = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = build_grad_scaler(amp_enabled)

    start_epoch = 1
    best_dice = float("-inf")
    stale_epochs = 0
    history: list[dict] = []
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_val_dice", best_dice))
        stale_epochs = int(checkpoint.get("epochs_without_improvement", 0))
        history_path = run_dir / "history.csv"
        if history_path.is_file():
            history = pd.read_csv(history_path).to_dict("records")
        logger.info("Resumed %s at epoch %d", resume_path, start_epoch)

    if init_path and resume_path is None:
        initial_metrics, _, _ = evaluate_model(
            evaluation_model, loaders["val"], loss_fn, device,
            threshold=float(train_cfg.get("threshold", 0.5)), amp=amp_enabled,
            description="initial validation",
        )
        best_dice = float(initial_metrics["mean"]["dice"])
        atomic_torch_save(
            checkpoint_payload(model, optimizer, scheduler, scaler, config, 0, best_dice, 0),
            run_dir / "best.pt",
        )
        save_json(initial_metrics, run_dir / "initial_val_metrics.json")
        logger.info("Initial validation Dice %.4f; keep initialization unless fine-tuning improves it", best_dice)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(run_dir / "tensorboard")
    except ImportError:
        logger.warning("tensorboard is unavailable; CSV logging remains enabled")

    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg.get("early_stopping_patience", epochs))
    threshold = float(train_cfg.get("threshold", 0.5))
    for epoch in range(start_epoch, epochs + 1):
        train_losses = train_one_epoch(
            model,
            loaders["train"],
            loss_fn,
            optimizer,
            scaler,
            device,
            amp=amp_enabled,
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 0.0)),
            epoch=epoch,
            freeze_encoder_bn=bool(train_cfg.get("freeze_encoder_bn", False)),
            multi_scale_factors=tuple(train_cfg.get("multi_scale_factors", [1.0])),
        )
        val_metrics, _, val_losses = evaluate_model(
            evaluation_model,
            loaders["val"],
            loss_fn,
            device,
            threshold=threshold,
            amp=amp_enabled,
            description=f"val {epoch:03d}",
        )
        current_lr = float(max(group["lr"] for group in optimizer.param_groups))
        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_losses["loss"],
            "val_loss": val_losses["loss"],
            "val_dice": val_metrics["mean"]["dice"],
            "val_iou": val_metrics["mean"]["iou"],
            "val_mae": val_metrics["mean"]["mae"],
            "val_boundary_f1": val_metrics["mean"]["boundary_f1"],
        }
        history.append(row)
        save_history(history, run_dir / "history.csv")
        plot_history(history, run_dir / "training_curves.png")
        if writer is not None:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)

        val_dice = float(val_metrics["mean"]["dice"])
        improved = val_dice > best_dice
        if improved:
            best_dice = val_dice
            stale_epochs = 0
        else:
            stale_epochs += 1
        if scheduler is not None:
            scheduler.step()
        payload = checkpoint_payload(
            model, optimizer, scheduler, scaler, config, epoch, best_dice, stale_epochs
        )
        atomic_torch_save(payload, run_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, run_dir / "best.pt")
            save_json(val_metrics, run_dir / "val_metrics.json")
        logger.info(
            "Epoch %03d | train %.4f | val %.4f | Dice %.4f | IoU %.4f | best %.4f",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_dice"],
            row["val_iou"],
            best_dice,
        )
        if stale_epochs >= patience:
            logger.info("Early stopping after %d epochs without improvement", stale_epochs)
            break

    if writer is not None:
        writer.close()

    best_path = run_dir / "best.pt"
    best_checkpoint = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    val_metrics, val_records, _ = evaluate_model(
        evaluation_model,
        loaders["val"],
        loss_fn,
        device,
        threshold=threshold,
        amp=amp_enabled,
        description="best validation",
    )
    save_json(val_metrics, run_dir / "val_metrics.json")
    save_per_image_records(val_records, run_dir / "val_per_image.csv")
    logger.info(
        "Best validation (epoch %d) | Dice %.4f | IoU %.4f",
        int(best_checkpoint["epoch"]),
        val_metrics["mean"]["dice"],
        val_metrics["mean"]["iou"],
    )

    test_metrics = None
    should_test = bool(train_cfg.get("test_after_training", True)) and not args.no_test
    if should_test:
        prediction_dir = (
            run_dir / "test_predictions"
            if bool(train_cfg.get("save_test_predictions", True))
            else None
        )
        test_metrics, test_records, _ = evaluate_model(
            evaluation_model,
            loaders["test"],
            loss_fn,
            device,
            threshold=threshold,
            amp=amp_enabled,
            description="held-out test",
            prediction_dir=prediction_dir,
        )
        save_json(test_metrics, run_dir / "test_metrics.json")
        save_per_image_records(test_records, run_dir / "test_per_image.csv")
        logger.info(
            "Test | Dice %.4f | IoU %.4f | MAE %.4f",
            test_metrics["mean"]["dice"],
            test_metrics["mean"]["iou"],
            test_metrics["mean"]["mae"],
        )
    else:
        reason = "--no-test" if args.no_test else "training.test_after_training=false"
        logger.info("Held-out test skipped (%s). No test metrics were computed.", reason)
        logger.info('Run test separately: python evaluate.py --checkpoint "%s" --split test', best_path)

    profile_cfg = config.get("profiling", {})
    lightweight = profile_model(
        evaluation_model,
        device,
        image_size=tuple(int(v) for v in data_cfg["image_size"]),
        batch_size=int(profile_cfg.get("batch_size", 1)),
        in_channels=int(config["model"].get("in_channels", 3)),
        warmup_iterations=int(profile_cfg.get("warmup_iterations", 30)),
        benchmark_iterations=int(profile_cfg.get("benchmark_iterations", 100)),
        checkpoint_path=best_path,
    )
    save_json(lightweight, run_dir / "lightweight_metrics.json")
    summary = {
        "experiment": config["experiment"]["name"],
        "implementation": implementation_fingerprint(),
        "model": {
            "architecture": config["model"].get("architecture", "rocu_net"),
            "backbone": config["model"].get("backbone"),
            "pretrained": config["model"].get("pretrained", False),
        },
        "protocol": {
            "seed": seed,
            "image_size": data_cfg["image_size"],
            "train_images": len(splits["train"]),
            "validation_images": len(splits["val"]),
            "test_images": len(splits["test"]),
            "threshold": threshold,
            "tta": evaluation_model.tta,
        },
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation": val_metrics,
        "held_out_test": test_metrics,
        "lightweight": lightweight,
    }
    save_json(summary, run_dir / "summary.json")
    logger.info("Finished. Results saved to %s", run_dir)


if __name__ == "__main__":
    main()
