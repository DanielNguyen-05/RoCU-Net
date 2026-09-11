from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm.auto import tqdm

from .metrics import BinarySegmentationMeter
from .utils import autocast_context


def _average_components(totals: dict[str, float], n_samples: int) -> dict[str, float]:
    return {key: value / max(n_samples, 1) for key, value in totals.items()}


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    loss_fn,
    optimizer,
    scaler,
    device: torch.device,
    *,
    amp: bool = True,
    grad_clip_norm: float = 0.0,
    epoch: int = 0,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    n_samples = 0
    amp_enabled = bool(amp and device.type == "cuda")
    progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        batch_size = image.shape[0]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(amp_enabled):
            outputs = model(image)
            loss, components = loss_fn(outputs, target)
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        n_samples += batch_size
        for key, value in components.items():
            totals[key] += float(value) * batch_size
        progress.set_postfix(loss=f"{components['loss']:.4f}")
    return _average_components(totals, n_samples)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    loss_fn,
    device: torch.device,
    *,
    threshold: float = 0.5,
    amp: bool = True,
    description: str = "eval",
    prediction_dir: str | Path | None = None,
) -> tuple[dict, list[dict[str, float | str]], dict[str, float]]:
    model.eval()
    meter = BinarySegmentationMeter(threshold=threshold)
    totals: dict[str, float] = defaultdict(float)
    conservation_totals: dict[str, float] = defaultdict(float)
    routing_totals: dict[str, float] = defaultdict(float)
    n_samples = 0
    amp_enabled = bool(amp and device.type == "cuda")
    output_dir = Path(prediction_dir) if prediction_dir is not None else None
    if output_dir is not None:
        (output_dir / "probability").mkdir(parents=True, exist_ok=True)
        (output_dir / "binary").mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc=description, leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        sample_ids = [str(value) for value in batch["id"]]
        with autocast_context(amp_enabled):
            outputs = model(image)
            _, components = loss_fn(outputs, target)
        probability = outputs["p352"].float()
        meter.update(probability, target.float(), sample_ids)
        batch_size = image.shape[0]
        n_samples += batch_size
        high_residual = torch.abs(F.avg_pool2d(outputs["p352"].float(), 2) - outputs["p176"].float()).mean()
        mid_residual = torch.abs(F.avg_pool2d(outputs["p176"].float(), 2) - outputs["p88"].float()).mean()
        conservation_totals["p352_to_p176_mae"] += float(high_residual) * batch_size
        conservation_totals["p176_to_p88_mae"] += float(mid_residual) * batch_size
        for key, output_key in (
            ("stage1_active_fraction", "routing_fraction_1"),
            ("stage2_active_fraction", "routing_fraction_2"),
            ("stage1_gate_mean", "routing_gate_1"),
            ("stage2_gate_mean", "routing_gate_2"),
        ):
            if output_key in outputs:
                routing_totals[key] += float(outputs[output_key].float().mean()) * batch_size
        for key, value in components.items():
            totals[key] += float(value) * batch_size
        if output_dir is not None:
            probability_cpu = probability.detach().cpu().numpy()
            for index, sample_id in enumerate(sample_ids):
                prob = np.clip(probability_cpu[index, 0] * 255.0, 0, 255).astype(np.uint8)
                binary = (probability_cpu[index, 0] >= threshold).astype(np.uint8) * 255
                Image.fromarray(prob, mode="L").save(output_dir / "probability" / f"{sample_id}.png")
                Image.fromarray(binary, mode="L").save(output_dir / "binary" / f"{sample_id}.png")
    summary = meter.summary()
    summary["occupancy_conservation"] = _average_components(conservation_totals, n_samples)
    if routing_totals:
        summary["confidence_routing"] = _average_components(routing_totals, n_samples)
    return summary, meter.records, _average_components(totals, n_samples)


def save_per_image_records(records: list[dict[str, float | str]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(destination, index=False)


def save_history(history: list[dict], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(history).to_csv(destination, index=False)


def plot_history(history: list[dict], path: str | Path) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame.from_records(history)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0, 0].plot(frame["epoch"], frame["val_loss"], label="validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 1].plot(frame["epoch"], frame["val_dice"], color="tab:green")
    axes[0, 1].set_title("Validation Dice")
    axes[1, 0].plot(frame["epoch"], frame["val_iou"], color="tab:blue")
    axes[1, 0].set_title("Validation IoU")
    axes[1, 1].plot(frame["epoch"], frame["learning_rate"], color="tab:orange")
    axes[1, 1].set_title("Learning rate")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
