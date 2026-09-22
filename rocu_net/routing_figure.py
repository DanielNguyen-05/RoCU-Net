"""Real-image routing measurements and Figure 3; no synthetic latency inputs."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .data import IMAGENET_MEAN, IMAGENET_STD
from .metrics import batch_metrics
from .utils import synchronize


def set_routing_mode(model, tau: float | None) -> None:
    """Dense disables only the hard bypass; preserve weights and soft blending."""
    for stage in (1, 2):
        block = getattr(model, f"rocu{stage}", None)
        if block is None or not (block.occupancy_constraint and block.routing_enabled):
            raise ValueError("Figure 3 requires RoCU blocks with occupancy constraint and soft gate enabled")
        block.hard_routing_inference = tau is not None
        if tau is not None:
            if not np.isfinite(tau) or not 0 <= tau <= 1:
                raise ValueError("Routing thresholds must be finite and in [0, 1]")
            block.uncertainty_threshold = float(tau)
    model.eval()


def extract_routing_maps(outputs, tau: float | None, index: int = 0) -> dict[str, np.ndarray]:
    """Save actual block inputs/decisions, and verify the routing equation."""
    maps = {"probability": outputs["p_full"][index, 0].float().cpu().numpy().copy()}
    for stage, key in ((1, "p_quarter"), (2, "p_half")):
        parent = outputs[key][index:index + 1]
        ambiguity = (4 * parent * (1 - parent)).clamp(0, 1)
        actual = outputs[f"routing_active_{stage}"][index:index + 1]
        expected = torch.ones_like(parent, dtype=torch.bool) if tau is None else ambiguity >= tau
        if not torch.equal(actual.bool(), expected):
            raise RuntimeError(f"Stage {stage} solver decisions do not match the recorded parent occupancy")
        if not torch.allclose(ambiguity, outputs[f"routing_uncertainty_{stage}"][index:index + 1]):
            raise RuntimeError(f"Stage {stage} ambiguity does not match the block diagnostic")
        maps[f"parent_{stage}"] = parent[0, 0].float().cpu().numpy().copy()
        maps[f"ambiguity_{stage}"] = ambiguity[0, 0].float().cpu().numpy().copy()
        maps[f"active_{stage}"] = actual[0, 0].bool().cpu().numpy().copy()
        maps[f"active_overlay_{stage}"] = F.interpolate(
            actual.float(), size=outputs["p_full"].shape[-2:], mode="nearest"
        )[0, 0].bool().cpu().numpy().copy()
    return maps


def balanced_mode_orders(n_modes: int, repeats: int, seed: int) -> list[list[int]]:
    """Each mode occupies every position once in each complete n_modes cycle."""
    if n_modes < 1 or repeats < 1:
        raise ValueError("Need positive mode and repeat counts")
    rng = np.random.default_rng(seed)
    orders = []
    while len(orders) < repeats:
        base = rng.permutation(n_modes)
        for shift in range(n_modes):
            orders.append(np.roll(base, -shift).tolist())
            if len(orders) == repeats:
                break
    return orders


@torch.inference_mode()
def measure_routing(model, dataset, device, modes, *, batch_size=1, warmup=30,
                    repeats=3, seed=42, threshold=.5, sample_id: str):
    """Time synchronized forward only. Metrics/maps consume the timed output.

    Decode/resize once on CPU. Transfer each batch before synchronization and
    timing. Balance mode positions using seeded cyclic orders to reduce order bias;
    every mode sees the same ordered images and the same warmup images.
    """
    if batch_size < 1 or repeats < 1 or warmup < 0:
        raise ValueError("batch_size/repeats must be positive and warmup nonnegative")
    if not len(dataset) or len(dataset) % batch_size:
        raise ValueError("Split size must be divisible by batch size; use --batch-size 1")
    items = [dataset[i] for i in tqdm(range(len(dataset)), desc="Decode real images")]
    ids = [str(item["id"]) for item in items]
    if len(set(ids)) != len(ids) or sample_id not in ids:
        raise ValueError("Unique sample IDs and an illustration from this split are required")
    batches = []
    for start in range(0, len(items), batch_size):
        group = items[start:start + batch_size]
        batches.append((torch.stack([x["image"] for x in group]),
                        torch.stack([x["mask"] for x in group]),
                        [str(x["id"]) for x in group]))
    timing_rows, metric_rows, samples, orders = [], [], {}, []
    for repeat, order in enumerate(balanced_mode_orders(len(modes), repeats, seed)):
        orders.append([modes[i][0] for i in order])
        for mode_index in order:
            label, tau = modes[mode_index]
            set_routing_mode(model, tau)
            for step in range(warmup):
                image = batches[step % len(batches)][0].to(device)
                model(image)
            synchronize(device)
            for batch_index, (image_cpu, mask_cpu, batch_ids) in enumerate(
                    tqdm(batches, desc=f"Repeat {repeat + 1}/{repeats} {label}")):
                image = image_cpu.to(device)
                synchronize(device)
                start = time.perf_counter_ns()
                outputs = model(image)
                synchronize(device)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                timing_rows.append({"mode": label, "repeat": repeat, "batch_index": batch_index,
                                    "sample_ids": "|".join(batch_ids), "n_images": len(batch_ids),
                                    "latency_ms_batch": elapsed, "latency_ms_image": elapsed / len(batch_ids)})
                # Everything below, including GPU metrics and CPU copies, is outside timing.
                if repeat == 0:
                    metrics = batch_metrics(outputs["p_full"], mask_cpu.to(device), threshold=threshold)
                    for i, sid in enumerate(batch_ids):
                        row = {"mode": label, "sample_id": sid,
                               **{k: float(v[i].cpu()) for k, v in metrics.items()}}
                        for stage in (1, 2):
                            active = outputs[f"routing_active_{stage}"][i]
                            row[f"stage{stage}_active_cells"] = int(active.sum().cpu())
                            row[f"stage{stage}_total_cells"] = active.numel()
                        metric_rows.append(row)
                        if sid == sample_id:
                            sample = extract_routing_maps(outputs, tau, i)
                            rgb = image_cpu[i].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
                            sample.update(image=np.clip(rgb, 0, 1), target=mask_cpu[i, 0].numpy().astype(bool))
                            samples[label] = sample
                del outputs
    return pd.DataFrame(timing_rows), pd.DataFrame(metric_rows), samples, orders


def repeat_measurements(timings: pd.DataFrame) -> pd.DataFrame:
    """Pair each mode with dense within the same complete dataset repeat."""
    frame = timings.groupby(["mode", "repeat"], as_index=False)[["latency_ms_batch", "n_images"]].sum()
    frame["latency_mean_ms_image"] = frame.latency_ms_batch / frame.n_images
    dense = frame.loc[frame["mode"] == "dense"].set_index("repeat").latency_mean_ms_image
    frame["dense_latency_ms_image"] = frame["repeat"].map(dense)
    if frame.dense_latency_ms_image.isna().any():
        raise ValueError("Every repeat must include a dense measurement")
    frame["latency_saved_ms_image"] = frame.dense_latency_ms_image - frame.latency_mean_ms_image
    frame["speedup_vs_dense"] = frame.dense_latency_ms_image / frame.latency_mean_ms_image
    return frame


def summarize_measurements(timings: pd.DataFrame, metrics: pd.DataFrame, modes) -> pd.DataFrame:
    rows = []
    repeat_frame = repeat_measurements(timings)
    for label, tau in modes:
        times = timings[timings["mode"] == label]
        scores = metrics[metrics["mode"] == label]
        grouped = times.groupby("repeat")[["latency_ms_batch", "n_images"]].sum()
        per_repeat = grouped.latency_ms_batch / grouped.n_images
        paired = repeat_frame[repeat_frame["mode"] == label]
        active = sum(scores[f"stage{s}_active_cells"].sum() for s in (1, 2))
        total = sum(scores[f"stage{s}_total_cells"].sum() for s in (1, 2))
        rows.append({"mode": label, "tau_r": tau, "n_images": len(scores),
                     "dice": scores.dice.mean(), "iou": scores.iou.mean(),
                     "boundary_f1": scores.boundary_f1.mean(),
                     "latency_mean_ms_image": times.latency_ms_batch.sum() / times.n_images.sum(),
                     "latency_median_ms_image": times.latency_ms_image.median(),
                     "latency_p95_ms_image": times.latency_ms_image.quantile(.95),
                     "n_repeats": len(per_repeat),
                     "latency_repeat_sd_ms": per_repeat.std(ddof=1) if len(per_repeat) > 1 else 0.,
                     "paired_latency_saved_mean_ms": paired.latency_saved_ms_image.mean(),
                     "paired_latency_saved_sd_ms": paired.latency_saved_ms_image.std(ddof=1) if len(paired) > 1 else 0.,
                     "repeats_faster_than_dense": int((paired.latency_saved_ms_image > 0).sum()),
                     "solver_skip_fraction": 1 - active / total,
                     **{f"stage{s}_active_fraction": scores[f"stage{s}_active_cells"].sum()
                        / scores[f"stage{s}_total_cells"].sum() for s in (1, 2)}})
    frame = pd.DataFrame(rows)
    dense = float(frame.loc[frame["mode"] == "dense", "latency_mean_ms_image"].iloc[0])
    frame["speedup_vs_dense"] = dense / frame.latency_mean_ms_image
    frame["fps"] = 1000 / frame.latency_mean_ms_image
    frame["latency_reduction_percent"] = 100 * (1 - frame.latency_mean_ms_image / dense)
    frame["dice_delta_vs_dense"] = frame.dice - float(frame.loc[frame["mode"] == "dense", "dice"].iloc[0])
    return frame


def save_paper_table(summary: pd.DataFrame, output: Path):
    """Export every operating point; no best-run or best-image filtering."""
    rows = []
    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Mode & Dice & Latency (ms) & FPS & Solver skipped (\%) \\", r"\midrule"]
    for row in summary.itertuples(index=False):
        name = "Dense" if row.mode == "dense" else f"tau={row.tau_r:.2f}"
        latex_name = "Dense" if row.mode == "dense" else rf"$\tau_r={row.tau_r:.2f}$"
        values = [name, f"{row.dice:.6f}",
                  f"{row.latency_mean_ms_image:.3f} ± {row.latency_repeat_sd_ms:.3f}",
                  f"{row.fps:.2f}", f"{100 * row.solver_skip_fraction:.2f}"]
        rows.append(values)
        tex.append(f"{latex_name} & {values[1]} & "
                   rf"${row.latency_mean_ms_image:.3f} \pm {row.latency_repeat_sd_ms:.3f}$"
                   + f" & {values[3]} & {values[4]} " + r"\\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "routing_summary.tex").write_text("\n".join(tex) + "\n")
    lines = ["| Mode | Dice | Latency (ms/image) | FPS | Solver skipped (%) |",
             "|---|---:|---:|---:|---:|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines += ["", "Latency: mean ± sample SD of complete-repeat means; FPS = 1000 / mean latency.",
              "SD describes timing repeatability on this fixed split, not variability over training seeds."]
    (output / "routing_summary.md").write_text("\n".join(lines) + "\n")


def save_dense_routed_table(summary: pd.DataFrame, output: Path, tau: float = .10) -> pd.DataFrame:
    """Export the compact same-checkpoint dense-vs-routed evidence table."""
    dense = summary.loc[summary["mode"] == "dense"]
    routed = summary.loc[summary["tau_r"].notna() & np.isclose(summary["tau_r"], tau)]
    if len(dense) != 1 or len(routed) != 1:
        raise ValueError(f"Need exactly one dense row and one tau_r={tau:.2f} row")
    selected = pd.concat((dense, routed), ignore_index=True).copy()
    selected["display_mode"] = ["Dense solver", f"Routed, tau_r={tau:.2f}"]
    selected["active_cell_ratio"] = 1 - selected["solver_skip_fraction"]
    columns = ["display_mode", "dice", "active_cell_ratio", "latency_mean_ms_image",
               "latency_repeat_sd_ms", "fps", "latency_reduction_percent", "dice_delta_vs_dense"]
    compact = selected[columns]
    compact.to_csv(output / "routing_dense_vs_routed.csv", index=False)

    tex = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Mode & Dice $\uparrow$ & Active cells (\%) $\downarrow$ & Latency (ms/image) $\downarrow$ \\",
           r"\midrule"]
    markdown = ["| Mode | Dice | Active cells (%) | Latency (ms/image) |",
                "|---|---:|---:|---:|"]
    for row in compact.itertuples(index=False):
        latex_mode = "Dense solver" if row.display_mode == "Dense solver" else rf"Routed, $\tau_r={tau:.2f}$"
        latency = f"{row.latency_mean_ms_image:.3f} ± {row.latency_repeat_sd_ms:.3f}"
        tex.append(f"{latex_mode} & {row.dice:.6f} & {100 * row.active_cell_ratio:.2f} & "
                   rf"${row.latency_mean_ms_image:.3f} \pm {row.latency_repeat_sd_ms:.3f}$" + r" \\")
        markdown.append(f"| {row.display_mode} | {row.dice:.6f} | "
                        f"{100 * row.active_cell_ratio:.2f} | {latency} |")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    routed_row = compact.iloc[1]
    note = (f"Batch 1; latency is mean ± sample SD across complete test-split repeats. "
            f"At tau_r={tau:.2f}, latency changed by {routed_row['latency_reduction_percent']:.2f}% "
            f"and Dice by {routed_row['dice_delta_vs_dense']:+.2e} relative to dense.")
    (output / "routing_dense_vs_routed.tex").write_text("\n".join(tex) + "\n")
    (output / "routing_dense_vs_routed.md").write_text("\n".join(markdown) + "\n\n" + note + "\n")
    return compact


def draw_routing_figure(summary, sample, *, label, sample_id, split, hardware,
                        batch_size, threshold, output: Path, dpi=300):
    """Render measured data, with editable vector exports and no invented points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    output.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
             "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
             "axes.spines.right": False}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(12.5, 7.8), layout="constrained")
        grid = fig.add_gridspec(2, 5, height_ratios=[1.65, 1])
        ax = fig.add_subplot(grid[0, :3])
        colors = ["#333333", "#0072B2", "#009E73", "#E69F00", "#CC79A7"]
        # Stagger coincident point labels rather than altering measured coordinates.
        offsets = [(10, 22), (10, -32), (-65, 55), (-65, -45), (12, -80)]
        for i, row in enumerate(summary.itertuples(index=False)):
            dense = row.mode == "dense"
            point_label = "Dense" if dense else rf"$\tau_r={row.tau_r:.2f}$"
            ax.scatter(row.latency_mean_ms_image, row.dice, color=colors[i % len(colors)],
                       marker=["D", "o", "s", "^", "v"][i % 5], s=65, zorder=3)
            ax.annotate(point_label, (row.latency_mean_ms_image, row.dice),
                        xytext=offsets[i % len(offsets)], textcoords="offset points", fontsize=10,
                        color=colors[i % len(colors)])
        ax.set(xlabel=f"Mean forward latency (ms/image; batch = {batch_size})", ylabel="Mean Dice")
        ax.set_title("(a) Routing quality–latency trade-off", loc="left", fontweight="bold")
        ax.grid(alpha=.2)
        ax.margins(x=.35)
        # Do not magnify numerical-scale Dice differences into apparent gains.
        low, high = float(summary.dice.min()), float(summary.dice.max())
        pad = max(.005, (high - low) * .7)
        ax.set_ylim(max(0, low - pad), min(1, high + pad))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        info = fig.add_subplot(grid[0, 3:])
        info.axis("off")
        info.text(.02, .98, f"RoCU-Net · {split} ({int(summary.n_images.iloc[0])} images)\n"
                  f"{hardware}\n320 × 320 · FP32 · no TTA · threshold {threshold:g}",
                  va="top", fontsize=10, linespacing=1.5)
        table_rows = [["Dense" if r.mode == "dense" else f"τr = {r.tau_r:.2f}",
                       f"{r.latency_mean_ms_image:.2f} ± {r.latency_repeat_sd_ms:.2f}",
                       f"{100 * r.solver_skip_fraction:.2f}%"] for r in summary.itertuples(index=False)]
        table = info.table(cellText=table_rows, colLabels=["Mode", "Latency (ms)", "Solver skipped"],
                           cellLoc="center", colWidths=[.25, .39, .36], bbox=[.01, .32, .99, .43])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row_index, _), cell in table.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if row_index == 0:
                cell.set_facecolor("#f3f3f3")
        repeats = int(summary.n_repeats.iloc[0]) if "n_repeats" in summary else None
        repetition = f"{repeats} full repeats; " if repeats is not None else ""
        info.text(.02, .25, repetition + "table: mean ± SD\n"
                  "Synchronized forward only; transfers excluded.\n"
                  "Orange cells: solver executed.\n"
                  "Convolutions and prediction heads remain dense.",
                  va="top", fontsize=9, linespacing=1.5)
        titles = ["Input", "Ground truth", "Prediction", "Stage 1 active cells", "Stage 2 active cells"]
        for col, title in enumerate(titles):
            panel = fig.add_subplot(grid[1, col])
            if col == 0:
                panel.imshow(sample["image"], interpolation="nearest")
            elif col in (1, 2):
                array = sample["target"] if col == 1 else sample["probability"] >= threshold
                panel.imshow(array, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                stage = col - 2
                panel.imshow(sample["image"], interpolation="nearest")
                active = sample[f"active_overlay_{stage}"]
                rgba = np.zeros((*active.shape, 4))
                rgba[..., :3] = (230 / 255, 159 / 255, 0)
                rgba[..., 3] = active * .65
                panel.imshow(rgba, interpolation="nearest")
                shape = sample[f"active_{stage}"].shape
                title += f"\n{shape[0]} × {shape[1]} · {sample[f'active_{stage}'].mean():.1%} active"
            panel.set_title(title, fontsize=10)
            panel.set_xticks([])
            panel.set_yticks([])
        fig.supxlabel(f"(b) {sample_id} · {label}   |   Orange = solver-active parent cells", fontsize=10)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output / f"figure3.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # Supplementary occupancy/ambiguity views use the same actual forward pass.
        fig, axes = plt.subplots(2, 3, figsize=(9, 5.8), layout="constrained")
        for stage in (1, 2):
            for col, (key, title) in enumerate((("parent", "Parent occupancy"),
                                               ("ambiguity", "Ambiguity: 4P(1−P)"),
                                               ("active", "Solver-active cells"))):
                im = axes[stage - 1, col].imshow(sample[f"{key}_{stage}"], vmin=0, vmax=1,
                    cmap="gray" if key == "active" else "viridis", interpolation="nearest")
                axes[stage - 1, col].set_title(f"Stage {stage}: {title}")
                axes[stage - 1, col].set_axis_off()
                fig.colorbar(im, ax=axes[stage - 1, col], fraction=.045, pad=.02)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output / f"figure3_occupancy.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def save_sample_arrays(sample, destination: Path, threshold: float):
    from PIL import Image

    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "maps.npz", **sample)
    Image.fromarray(np.round(sample["image"] * 255).astype(np.uint8)).save(destination / "input.png")
    Image.fromarray(sample["target"].astype(np.uint8) * 255).save(destination / "ground_truth.png")
    Image.fromarray((sample["probability"] >= threshold).astype(np.uint8) * 255).save(destination / "prediction.png")
    for stage in (1, 2):
        for name in (f"active_{stage}", f"active_overlay_{stage}"):
            Image.fromarray(sample[name].astype(np.uint8) * 255).save(destination / f"{name}.png")
