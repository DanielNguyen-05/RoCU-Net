"""Measure routing on real split images and export the paper's Figure 3."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocu_net.compat import load_checkpoint
from rocu_net.config import resolve_project_path
from rocu_net.data import JointTransform, PolypSegDataset, _read_manifest
from rocu_net.model import build_model
from rocu_net.routing_figure import (balanced_mode_orders, draw_routing_figure, measure_routing,
                                     repeat_measurements, save_dense_routed_table, save_paper_table,
                                     save_sample_arrays, set_routing_mode, summarize_measurements)
from rocu_net.utils import get_device, implementation_fingerprint, save_json, set_seed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def figure_caption(report: dict, illustration: dict) -> str:
    return (
        f"Figure 3. Real-image routing evaluation with one RoCU-Net checkpoint "
        f"(epoch {report['checkpoint_epoch']}) on the same {report['n_images']} {report['split']} images. "
        f"(a) Mean Dice versus mean forward latency for dense and tau_r = "
        f"{', '.join(f'{t:.2f}' for t in report['thresholds'])}. Dense retains soft gating and disables "
        f"only hard solver routing. All modes use 320 × 320 inputs, batch size {report['batch_size']}, "
        f"FP32 without TF32 or TTA, segmentation threshold {report['segmentation_threshold']:g}, "
        f"and {report['hardware']['name']}. Each mode has "
        f"{report['warmup_forwards_per_mode_per_repeat']} real-image warmup forwards per repeat and "
        f"{report['repeats']} timed passes over the full split. "
        f"Mode order: {report.get('mode_order_design', 'seeded random permutation per repeat')}. "
        "Scatter points are shown without connecting lines or error bars; table uncertainties show "
        "the SD of complete-repeat mean latencies, not confidence intervals. "
        "Device-synchronized wall-clock forward timing excludes loading, preprocessing, transfers, "
        "metrics and figure export. Dice uses first-repeat predictions. "
        f"(b) Image {illustration['sample_id']} ({illustration['selection']}) at "
        f"tau_r = {illustration['tau_r']:.2f}: input, ground truth, prediction and actual solver-active "
        "parent cells. Stage 1 (80 × 80) uses A1 = 1[4P0(1−P0) >= tau_r]; stage 2 (160 × 160) "
        "uses A2 = 1[4P1(1−P1) >= tau_r], where P0 and P1 are the actual inputs to the respective "
        "blocks in this routed forward pass. Active maps are enlarged with nearest-neighbor "
        "interpolation. Colored cells denote occupancy solver execution; convolutions and prediction "
        "heads remain dense. Table skip percentages aggregate both stages and all images, whereas "
        "panel (b) percentages refer to the illustrated image.\n"
    )


def dense_routed_caption(report: dict, tau: float) -> str:
    return (
        "Dense-versus-routed inference using the same RoCU-Net checkpoint on all "
        f"{report['n_images']} real {report['split']} images. Inputs are 320 x 320, batch size "
        f"{report['batch_size']}, FP32 without TTA, on {report['hardware']['name']}. Active-cell ratio "
        "is the fraction of parent cells that execute the occupancy solver, aggregated across both "
        f"stages and all images. Routed inference uses tau_r={tau:.2f}. Latency is synchronized "
        f"forward time reported as mean +/- sample SD over {report['repeats']} complete split repeats; "
        "loading, preprocessing, transfers, metrics and export are excluded. Convolutions and "
        "prediction heads remain dense.\n"
    )


def render(output: Path, dpi: int, illustration_threshold: float | None = None):
    report = json.loads((output / "measurement.json").read_text())
    for name, expected in report["artifact_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"Recorded experimental artifact changed: {name}")
    summary = pd.read_csv(output / "routing_summary.csv")
    settings_path = output / "figure_settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    illustration = dict(settings.get("illustration", report["illustration"]))
    if illustration_threshold is not None:
        if illustration_threshold not in report["thresholds"]:
            raise ValueError("Illustration threshold must be one of the measured thresholds")
        illustration.update(tau_r=illustration_threshold, mode=f"tau_{illustration_threshold:g}")
    save_dense_routed_table(summary, output, illustration["tau_r"])
    (output / "routing_dense_vs_routed_caption.txt").write_text(
        dense_routed_caption(report, illustration["tau_r"]), encoding="utf-8")
    with np.load(output / "samples" / illustration["mode"] / "maps.npz") as archive:
        sample = {name: archive[name] for name in archive.files}
    draw_routing_figure(summary, sample, label=f"τr = {illustration['tau_r']:.2f}",
                        sample_id=illustration["sample_id"], split=report["split"],
                        hardware=report["hardware"]["name"], batch_size=report["batch_size"],
                        threshold=report["segmentation_threshold"], output=output, dpi=dpi)
    save_json({"illustration": illustration, "plot": "scatter", "error_bars": False,
               "measurement_sha256": sha256(output / "measurement.json")}, settings_path)
    (output / "caption.txt").write_text(figure_caption(report, illustration), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[.05, .10, .20, .30])
    parser.add_argument("--illustration-threshold", type=float,
                        help="Panel (b) threshold; new measurements default to 0.10; also works with --render-only")
    parser.add_argument("--sample-id", help="Image ID from the selected split; default: seeded random, independent of scores")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30, help="Real-image warmup forwards per mode per repeat")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Complete timed passes; must be a multiple of the number of measured modes")
    parser.add_argument("--cpu-threads", type=int, help="Fix CPU intra-op threads for every mode, including CUDA host work")
    parser.add_argument("--seed", type=int, default=42, help="Illustration selection and mode-order seed")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--table-only", action="store_true",
                        help="Measure and export tables/raw data without rendering or saving illustration panels")
    parser.add_argument("--render-only", type=Path, metavar="RESULT_DIR", help="Redraw saved measurements without inference")
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("dpi must be positive")
    if args.render_only:
        if args.checkpoint or args.output or args.table_only:
            parser.error("--render-only takes a completed result directory; omit --checkpoint/--output")
        render(args.render_only.expanduser().resolve(), args.dpi, args.illustration_threshold)
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required for measurement")
    if args.illustration_threshold is None:
        args.illustration_threshold = .10
    if args.batch_size < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("batch-size/repeats must be positive and warmup nonnegative")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    if (len(set(args.thresholds)) != len(args.thresholds)
            or any(not np.isfinite(t) or not 0 <= t <= 1 for t in args.thresholds)):
        parser.error("thresholds must be unique, finite values in [0, 1]")
    if args.illustration_threshold not in args.thresholds:
        parser.error("illustration-threshold must be among --thresholds")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = deepcopy(checkpoint["config"])
    if tuple(config["data"]["image_size"]) != (320, 320):
        raise ValueError("Figure 3 uses 320 × 320; this checkpoint has a different evaluation resolution")
    cfg_model = config["model"]
    if not all(cfg_model.get(k, True) for k in ("occupancy_constraint", "routing_enabled", "use_semantic_carrier")):
        raise ValueError("Use a full RoCU-Net checkpoint, e.g. A0, for the routing-only comparison")
    root = resolve_project_path(args.data_root or config["data"]["root"])
    if not (root / config["data"].get("image_dir", "images")).is_dir() and (root / "Kvasir-SEG").is_dir():
        root = root / "Kvasir-SEG"
    manifest = checkpoint_path.parent / "splits" / f"{args.split}.csv"
    pairs = _read_manifest(manifest, root)  # Never create or change an evaluation split.
    ids = [p.sample_id for p in pairs]
    if not ids or len(set(ids)) != len(ids) or len(ids) % args.batch_size:
        raise ValueError("Need a nonempty original split with unique IDs, divisible by batch size")
    sample_id = args.sample_id or str(np.random.default_rng(args.seed).choice(sorted(ids)))
    if sample_id not in ids:
        raise ValueError(f"Illustration ID {sample_id!r} is not in {args.split}")
    output = (args.output or checkpoint_path.parent / f"figure3_{args.split}").expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is nonempty: {output}. Use a new --output or --render-only.")
    device = get_device(args.device)
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed, deterministic=True)
    # Use explicit FP32, including matrix operations on CUDA, for every mode.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg_model["pretrained"] = False
    model = build_model(config).float().to(device)
    model.load_state_dict(checkpoint["model"])
    set_routing_mode(model, None)
    modes = [("dense", None)] + [(f"tau_{t:g}", t) for t in args.thresholds]
    if args.repeats % len(modes):
        parser.error(f"--repeats must be a multiple of {len(modes)} so every mode occupies every timing position equally")
    hardware_name = (torch.cuda.get_device_name(device) if device.type == "cuda"
                     else f"{platform.processor() or platform.machine()} ({device.type.upper()})")
    if device.type == "cpu":
        hardware_name += f", {torch.get_num_threads()} threads"
    dataset = PolypSegDataset(pairs, JointTransform((320, 320)))
    threshold = float(config["training"].get("threshold", .5))
    source = implementation_fingerprint()
    for file in (Path(__file__), Path(__file__).resolve().parents[1] / "rocu_net/routing_figure.py"):
        source["source_sha256"][file.name] = sha256(file)
    report = {
        "measurement_started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_config": checkpoint["config"],
        "implementation": source, "split": args.split, "manifest": str(manifest),
        "manifest_sha256": sha256(manifest), "n_images": len(pairs), "data_root": str(root),
        "data_files": [{"sample_id": p.sample_id, "image_sha256": sha256(p.image),
                        "mask_sha256": sha256(p.mask)} for p in pairs],
        "image_size": [320, 320], "batch_size": args.batch_size, "precision": "fp32",
        "autocast": False, "tf32": False, "tta": "none", "segmentation_threshold": threshold,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "thresholds": args.thresholds, "warmup_forwards_per_mode_per_repeat": args.warmup,
        "repeats": args.repeats, "seed": args.seed,
        "mode_order_design": "Seeded permutation, then cyclic rotation; each mode occupies every position "
                             f"once per complete cycle of {len(modes)} repeats",
        "planned_mode_order_per_repeat": [[modes[i][0] for i in order] for order in
                                           balanced_mode_orders(len(modes), args.repeats, args.seed)],
        "latency_sd_definition": "Sample SD (ddof=1) across complete-repeat mean latencies; not a confidence interval",
        "timing_scope": "perf_counter_ns around model(image), device synchronization before/after; "
                        "includes Python dispatch and solver; excludes decoding, preprocessing, transfers, "
                        "metrics, drawing and saving; all real images, no outlier removal",
        "dice_scope": "Unweighted mean over all images; computed from first timed repeat outputs",
        "hardware": {"name": hardware_name, "device": str(device), "platform": platform.platform(),
                     "torch": str(torch.__version__), "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version(), "cpu_threads": torch.get_num_threads(),
                     "python": platform.python_version(), "numpy": np.__version__},
        "illustration": {"sample_id": sample_id, "tau_r": args.illustration_threshold,
                         "mode": f"tau_{args.illustration_threshold:g}",
                         "selection": "explicit ID" if args.sample_id else "seeded random from sorted split IDs"},
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(report, output / "measurement_plan.json")
    print(f"Measuring {len(pairs)} real {args.split} images on {hardware_name}; illustration: {sample_id}", flush=True)
    timings, metrics, samples, orders = measure_routing(
        model, dataset, device, modes, batch_size=args.batch_size, warmup=args.warmup,
        repeats=args.repeats, seed=args.seed, threshold=threshold, sample_id=sample_id)
    summary = summarize_measurements(timings, metrics, modes)
    timings.to_csv(output / "latency_raw.csv", index=False)
    metrics.to_csv(output / "metrics_per_image.csv", index=False)
    repeat_measurements(timings).to_csv(output / "latency_per_repeat.csv", index=False)
    summary.to_csv(output / "routing_summary.csv", index=False)
    save_paper_table(summary, output)
    save_dense_routed_table(summary, output, args.illustration_threshold)
    (output / "routing_dense_vs_routed_caption.txt").write_text(
        dense_routed_caption(report, args.illustration_threshold), encoding="utf-8")
    if not args.table_only:
        for label, sample in samples.items():
            save_sample_arrays(sample, output / "samples" / label, threshold)
    report["mode_order_per_repeat"] = orders
    report["measurement_finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["artifact_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    save_json(report, output / "measurement.json")
    if not args.table_only:
        render(output, args.dpi)
    print(summary.to_string(index=False))
    print(f"Routing tables and raw measurements saved to {output}")


if __name__ == "__main__":
    main()
