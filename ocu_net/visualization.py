"""Reproducible paper figures. All metric plots use the full evaluation split."""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})


def save_figure(fig, output: Path, dpi: int = 300):
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def select_samples(records: pd.DataFrame, count: int, selection: str, seed: int) -> list[str]:
    count = min(count, len(records))
    if count < 1:
        raise ValueError("At least one sample is required")
    ordered = records.sort_values(["dice", "sample_id"], kind="stable")
    if selection == "quantiles":
        positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
        return ordered.iloc[positions].sample_id.tolist()
    if selection == "best":
        return ordered.tail(count).iloc[::-1].sample_id.tolist()
    if selection == "worst":
        return ordered.head(count).sample_id.tolist()
    return records.sample(n=count, random_state=seed).sample_id.tolist()


def error_map(prediction, target):
    rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
    rgb[prediction & target] = (0, 158, 115)  # TP: green
    rgb[prediction & ~target] = (230, 159, 0)  # FP: orange
    rgb[~prediction & target] = (204, 121, 167)  # FN: purple
    return rgb


def draw_contours(ax, image, target, prediction):
    ax.imshow(image)
    if target.any() and not target.all():
        ax.contour(target, levels=[0.5], colors=["#56B4E9"], linewidths=1.1)
    if prediction.any() and not prediction.all():
        ax.contour(prediction, levels=[0.5], colors=["#E69F00"], linewidths=1.1)


def qualitative(samples, output: Path, title: str, dpi: int):
    titles = ["Input", "Ground truth", "RoCU-Net", "Contours / detail", "Pixel errors", "Probability"]
    fig, axes = plt.subplots(len(samples), 6, figsize=(13, 2.15 * len(samples) + 0.6), squeeze=False)
    for row, sample in enumerate(samples):
        image, target, prob = sample["image"], sample["target"], sample["probability"]
        pred = prob >= sample["threshold"]
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        draw_contours(axes[row, 3], image, target, pred)
        # A reproducible crop around the union of GT and prediction.
        ys, xs = np.where(target | pred)
        if len(xs):
            h, w = target.shape
            x0, x1 = max(0, xs.min()-8), min(w, xs.max()+9)
            y0, y1 = max(0, ys.min()-8), min(h, ys.max()+9)
            axes[row, 3].set_xlim(x0, x1)
            axes[row, 3].set_ylim(y1, y0)
            axes[row, 0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#56B4E9", linewidth=1))
        axes[row, 4].imshow(error_map(pred, target))
        im = axes[row, 5].imshow(prob, cmap="viridis", vmin=0, vmax=1)
        for col, ax in enumerate(axes[row]):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(titles[col])
        axes[row, 0].set_ylabel(f"{sample['id'][:22]}\nDice {sample['dice']:.3f}\nIoU {sample['iou']:.3f}", fontsize=8)
    fig.suptitle(title, y=1.01)
    fig.legend(handles=[Patch(color="#56B4E9", label="GT contour"), Patch(color="#E69F00", label="Prediction contour / FP"), Patch(color="#009E73", label="TP"), Patch(color="#CC79A7", label="FN")], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.5 / fig.get_figheight(), 1, 1))
    fig.colorbar(im, ax=axes[:, 5].tolist(), fraction=0.025, pad=0.02, label="P(polyp)")
    save_figure(fig, output, dpi)


def diagnostics(samples, output: Path, dpi: int):
    if not all("routing" in s for s in samples):
        return
    fig, axes = plt.subplots(len(samples), 4, figsize=(9, len(samples)*2.2), squeeze=False)
    for row, s in enumerate(samples):
        for col, (key, title, cmap, vmax) in enumerate([
            ("boundary", "Auxiliary boundary", "viridis", 1),
            ("routing", "Soft routing gate (stage 2)", "viridis", 1),
            ("active", "Solver-active cells (stage 2)", "gray", 1),
            ("residual", "Occupancy residual (stage 2)", "magma", max(1e-6, max(float(x['residual'].max()) for x in samples))),
        ]):
            im = axes[row, col].imshow(s[key], cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0: axes[row, col].set_title(title)
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.02)
        axes[row, 0].set_ylabel(s['id'][:22], fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, dpi)


def metric_figures(frame: pd.DataFrame, output: Path, dpi: int):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    keys = ["dice", "iou", "boundary_f1"]
    axes[0].boxplot([frame[k] for k in keys], showmeans=True)
    axes[0].set_xticks([1, 2, 3], ["Dice", "IoU", "Boundary F1"])
    axes[0].set_ylim(-0.03, 1.03); axes[0].set_title(f"All {len(frame)} evaluated images")
    axes[1].scatter(frame.foreground_fraction * 100, frame.dice, s=15, alpha=0.65, color="#0072B2")
    axes[1].set(xlabel="Ground-truth area (% of image)", ylabel="Dice", ylim=(-0.03, 1.03))
    axes[2].hist(frame.dice, bins=np.linspace(0, 1, 21), color="#009E73", edgecolor="white")
    axes[2].set(xlabel="Dice", ylabel="Number of images", xlim=(0, 1))
    fig.tight_layout(); save_figure(fig, output / "metric_distribution", dpi)


def training_figure(source: Path, output: Path, dpi: int = 300):
    if source.suffix == ".csv":
        frame = pd.read_csv(source)
    else:
        pattern = r"Epoch\s+(\d+)\s*\|\s*train\s+([\d.]+)\s*\|\s*val\s+([\d.]+)\s*\|\s*Dice\s+([\d.]+)\s*\|\s*IoU\s+([\d.]+)"
        rows = re.findall(pattern, source.read_text(errors="replace"))
        if not rows: raise ValueError(f"No epoch records in {source}")
        frame = pd.DataFrame(rows, columns=["epoch", "train_loss", "val_loss", "val_dice", "val_iou"]).astype(float)
    if frame.epoch.duplicated().any():
        raise ValueError("Duplicate epochs: provide the history.csv of one run")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "training_data.csv", index=False)
    best = frame.loc[frame.val_dice.idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(frame.epoch, frame.train_loss, label="Train", color="#0072B2")
    axes[0].plot(frame.epoch, frame.val_loss, label="Validation", color="#D55E00")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[1].plot(frame.epoch, frame.val_dice, label="Validation Dice", color="#009E73")
    axes[1].plot(frame.epoch, frame.val_iou, label="Validation IoU", color="#0072B2")
    axes[1].axvline(best.epoch, color="0.5", linestyle="--", linewidth=1, label=f"Best val. Dice: epoch {int(best.epoch)}")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=0.15)
    fig.tight_layout(); save_figure(fig, output / "training_curves", dpi)
    (output / "training_caption.txt").write_text(f"Training and validation curves from {source.name}; best validation Dice at epoch {int(best.epoch)}. Values extracted from {'rounded console logs' if source.suffix != '.csv' else 'history CSV'}. No smoothing. This figure does not measure test performance.\n")
