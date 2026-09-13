"""Run the four predefined Kvasir ablations, then export separate val/test tables."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocu_net.compat import load_checkpoint
from rocu_net.config import PROJECT_ROOT, load_config, resolve_project_path, save_config
from rocu_net.data import create_or_load_splits, discover_pairs, image_content_hash
from rocu_net.utils import save_json


VARIANTS = (
    ("A0", "Full (dense)", "kvasir_a0_full.yaml"),
    ("A1", "Without occupancy constraint", "kvasir_a1_no_constraint.yaml"),
    ("A2", "Without propagated carrier", "kvasir_a2_no_carrier.yaml"),
    ("A3", "Without adaptive blending", "kvasir_a3_no_blending.yaml"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_config(config: dict) -> dict:
    return {k: v for k, v in config.items() if not k.startswith("_")}


def portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def validate_variants(configs: list[dict]) -> None:
    """Reject accidental changes to a second factor or the shared protocol."""
    base = clean_config(configs[0])
    for index, config in enumerate(configs):
        expected = deepcopy(base)
        expected["experiment"]["name"] = config["experiment"]["name"]
        if index == 1:
            expected["model"]["occupancy_constraint"] = False
        elif index == 2:
            expected["model"]["use_semantic_carrier"] = False
        elif index == 3:
            expected["model"]["routing_enabled"] = False
            expected["loss"]["routing_weight"] = 0.0
        if clean_config(config) != expected:
            raise ValueError(f"{VARIANTS[index][0]} differs from A0 beyond its intended ablation")
    model, training = base["model"], base["training"]
    if not all(model[k] for k in ("occupancy_constraint", "use_semantic_carrier", "routing_enabled")):
        raise ValueError("A0 must enable all three components")
    if model["hard_routing_inference"] or base["inference"]["tta"] != "none":
        raise ValueError("Ablations require dense inference and no TTA")
    if training.get("init_checkpoint") or training["test_after_training"] or training["eval_amp"]:
        raise ValueError("Ablations require fresh training, deferred test and FP32 evaluation")
    if base["experiment"]["seed"] != 42 or not base["data"]["group_identical_images"]:
        raise ValueError("This suite uses seed 42 and pixel-grouped splits")
    if len({c["experiment"]["name"] for c in configs}) != 4:
        raise ValueError("Each ablation needs a separate run directory")


def suite_configs(output: Path, data_root: Path | None = None, epochs: int | None = None,
                  smoke: bool = False) -> list[dict]:
    configs = []
    for _, _, filename in VARIANTS:
        cfg = load_config(PROJECT_ROOT / "configs/ablations" / filename)
        cfg["experiment"]["output_dir"] = portable_path(output)
        cfg["data"]["root"] = portable_path(data_root or resolve_project_path(cfg["data"]["root"]))
        cfg["data"]["split_source"] = portable_path(output / "splits")
        if epochs is not None:
            if epochs < 1:
                raise ValueError("epochs must be positive")
            cfg["training"]["epochs"] = epochs
            cfg["training"]["warmup_epochs"] = min(cfg["training"]["warmup_epochs"], epochs - 1)
        if smoke:
            cfg["data"].update(image_size=[64, 64], num_workers=0, pin_memory=False, persistent_workers=False)
            cfg["model"].update(backbone="efficient", pretrained=False, backbone_channels=[8, 12, 16, 24],
                                backbone_depths=[1, 1, 1, 1], decoder_channels=[20, 16],
                                carrier_channels=8, shallow_guide_channels=8)
            cfg["training"].update(epochs=1, warmup_epochs=0, amp=False, batch_size=3, eval_batch_size=3)
            cfg["profiling"].update(warmup_iterations=0, benchmark_iterations=1)
        configs.append(cfg)
    validate_variants(configs)
    return configs


def prepare(output: Path, configs: list[dict], smoke: bool) -> dict:
    data = configs[0]["data"]
    root = resolve_project_path(data["root"])
    splits = create_or_load_splits(
        root, output / "splits", image_dir=data["image_dir"], mask_dir=data["mask_dir"],
        train_ratio=data["train_ratio"], val_ratio=data["val_ratio"], test_ratio=data["test_ratio"],
        seed=42, group_identical_images=True,
    )
    pairs = discover_pairs(root, data["image_dir"], data["mask_dir"])
    ids = [p.sample_id for values in splits.values() for p in values]
    if len(ids) != len(set(ids)) or set(ids) != {p.sample_id for p in pairs}:
        raise ValueError("Shared manifests do not cover the current dataset exactly once")
    # Fingerprint actual pixels and masks; filenames alone cannot detect changed data.
    dataset_digest = hashlib.sha256()
    hashes = set()
    for pair in pairs:
        image_hash = image_content_hash(pair.image)
        hashes.add(image_hash)
        dataset_digest.update(json.dumps([pair.sample_id, image_hash, image_content_hash(pair.mask)]).encode())
    protocol = {
        "smoke_test": smoke, "dataset_root": portable_path(root), "seed": 42,
        "dataset_sha256": dataset_digest.hexdigest(), "unique_decoded_images": len(hashes),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_sha256": {k: sha256(output / "splits" / f"{k}.csv") for k in splits},
        "configs": [clean_config(c) for c in configs],
        "note": "Exact-pixel grouping does not establish patient/video independence. One seed only.",
    }
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Dataset, split or config changed. Use a new output root; existing suite is preserved.")
    # Check all existing run configs before writing any generated configs.
    for cfg in configs:
        run = output / cfg["experiment"]["name"]
        for name in ("config_resolved.yaml", "ablation_config.yaml"):
            old = run / name
            if old.exists() and clean_config(load_config(old)) != clean_config(cfg):
                raise ValueError(f"Existing config does not match this suite: {old}")
    save_json(protocol, path)
    for cfg in configs:
        save_config(cfg, output / cfg["experiment"]["name"] / "ablation_config.yaml")
    return protocol


def verify_run(run: Path, cfg: dict, protocol: dict) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    checkpoint = load_checkpoint(run / "best.pt")
    if clean_config(checkpoint["config"]) != clean_config(cfg) or checkpoint["epoch"] != summary["best_epoch"]:
        raise ValueError(f"Checkpoint/config/summary mismatch in {run}")
    for split, digest in protocol["split_sha256"].items():
        if sha256(run / "splits" / f"{split}.csv") != digest:
            raise ValueError(f"{run}: {split} differs from the shared split")
    return summary


def collect(output: Path, configs: list[dict], protocol: dict, split: str) -> pd.DataFrame:
    rows = []
    for (variant, label, _), cfg in zip(VARIANTS, configs):
        run = output / cfg["experiment"]["name"]
        summary = verify_run(run, cfg, protocol)
        metrics = json.loads((run / f"{split}_metrics.json").read_text())
        if metrics["n_images"] != protocol["split_sizes"][split] or metrics["threshold"] != cfg["training"]["threshold"]:
            raise ValueError(f"Metric protocol mismatch in {run}")
        if metrics["inference"]["tta"] != "none":
            raise ValueError(f"Unexpected TTA in {run}")
        if split == "test":
            provenance = json.loads((run / "test_provenance.json").read_text())
            if provenance != {"checkpoint_sha256": sha256(run / "best.pt"),
                              "split_sha256": protocol["split_sha256"]["test"],
                              "metrics_sha256": sha256(run / "test_metrics.json")}:
                raise ValueError(f"Stale test results in {run}; run --evaluate-only")
        elif metrics != summary["best_validation"]:
            raise ValueError(f"Validation metrics do not match the completed training summary: {run}")
        residual = metrics["occupancy_conservation"]
        full, half = residual["p352_to_p176_mae"], residual["p176_to_p88_mae"]
        mean = metrics["mean"]
        rows.append({"Variant": variant, "Description": label, "Split": split, "Best epoch": summary["best_epoch"],
                     "Dice": mean["dice"], "IoU": mean["iou"], "Boundary F1": mean["boundary_f1"],
                     "Occupancy MAE": (full + half) / 2,
                     "Occupancy MAE full-half": full, "Occupancy MAE half-quarter": half,
                     "Checkpoint SHA256": sha256(run / "best.pt")})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"ablation_{split}.csv", index=False)
    heading = "SOFTWARE SMOKE TEST — NOT PAPER RESULTS" if protocol["smoke_test"] else f"Kvasir ablation — {split.upper()}"
    lines = [f"# {heading}", "", "Seed 42; no TTA; no hard routing; FP32 evaluation.", "",
             "| Variant | Dice ↑ | IoU ↑ | Boundary F1 ↑ | Occupancy MAE ↓ |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['Variant']} — {row['Description']} | {row['Dice']:.4f} | {row['IoU']:.4f} | "
                     f"{row['Boundary F1']:.4f} | {row['Occupancy MAE']:.3e} |")
    (output / f"ablation_{split}.md").write_text("\n".join(lines) + "\n")
    latex = ["% " + heading, r"\begin{tabular}{lrrrr}", r"\hline",
             r"Variant & Dice $\uparrow$ & IoU $\uparrow$ & Boundary F1 $\uparrow$ & Occupancy MAE $\downarrow$ \\", r"\hline"]
    for row in rows:
        mantissa, exponent = f"{row['Occupancy MAE']:.3e}".split("e")
        latex.append(f"{row['Variant']} & {row['Dice']:.4f} & {row['IoU']:.4f} & {row['Boundary F1']:.4f} & "
                     + f"${mantissa} \\times 10^{{{int(exponent)}}}$" + r" \\")
    latex += [r"\hline", r"\end{tabular}"]
    (output / f"ablation_{split}.tex").write_text("\n".join(latex) + "\n")
    print(frame[["Variant", "Split", "Dice", "IoU", "Boundary F1", "Occupancy MAE"]].to_string(index=False), flush=True)
    return frame


def execute(command: list[str], log: Path) -> None:
    print("Running: " + " ".join(command) + f"\nLog: {log}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--epochs", type=int, help="Override epochs identically for all four variants")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs; resume unfinished runs from last.pt")
    parser.add_argument("--train-only", action="store_true", help="Train all variants and export validation only")
    parser.add_argument("--smoke-test", action="store_true", help="Four one-epoch offline runs on generated toy data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true", help="Evaluate all four completed checkpoints on test")
    mode.add_argument("--collect-only", action="store_true", help="Rebuild tables from saved, verified metrics")
    args = parser.parse_args()
    output = (args.output_root or PROJECT_ROOT / "runs" / ("rocu_kvasir_ablations_smoke" if args.smoke_test else "rocu_kvasir_ablations_seed42")).resolve()
    if args.smoke_test:
        if args.data_root is not None:
            parser.error("--smoke-test uses generated toy data; omit --data-root")
        args.data_root = output / "toy_data"
        if not args.data_root.exists():
            execute([sys.executable, str(PROJECT_ROOT / "scripts/create_toy_kvasir.py"),
                     "--output", str(args.data_root), "--count", "20", "--size", "64"], output / "logs/toy_data.log")
    configs = suite_configs(output, args.data_root, args.epochs, args.smoke_test)
    protocol = prepare(output, configs, args.smoke_test)
    print(f"Shared split sizes: {protocol['split_sizes']}; output: {output}", flush=True)
    if args.prepare_only:
        return
    if not (args.collect_only or args.evaluate_only):
        # Preflight all destinations before spending GPU time on the first run.
        for cfg in configs:
            run = output / cfg["experiment"]["name"]
            if not args.resume and any((run / f).exists() for f in ("best.pt", "last.pt", "summary.json")):
                raise FileExistsError(f"Existing run: {run}. Use --resume or a new --output-root.")
            if args.resume and (run / "best.pt").exists() and not (run / "summary.json").exists() and not (run / "last.pt").exists():
                raise FileNotFoundError(f"Missing last.pt for resume: {run}")
        for cfg in configs:
            run = output / cfg["experiment"]["name"]
            if args.resume and (run / "summary.json").exists():
                verify_run(run, cfg, protocol)
                print(f"Completed, skipping training: {run.name}", flush=True)
                continue
            command = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", str(run / "ablation_config.yaml"), "--device", args.device]
            if args.resume and (run / "last.pt").exists():
                command += ["--resume", str(run / "last.pt")]
            execute(command, output / "logs" / f"{run.name}.log")
    collect(output, configs, protocol, "val")
    if args.train_only:
        return
    if not args.collect_only:
        # No test results are computed until every training run has completed.
        for cfg in configs:
            run = output / cfg["experiment"]["name"]
            execute([sys.executable, str(PROJECT_ROOT / "evaluate.py"), "--checkpoint", str(run / "best.pt"),
                     "--split", "test", "--device", args.device, "--save-predictions"],
                    output / "logs" / f"{run.name}_test.log")
            save_json({"checkpoint_sha256": sha256(run / "best.pt"),
                       "split_sha256": protocol["split_sha256"]["test"],
                       "metrics_sha256": sha256(run / "test_metrics.json")}, run / "test_provenance.json")
            summary = json.loads((run / "summary.json").read_text())
            summary["held_out_test"] = json.loads((run / "test_metrics.json").read_text())
            summary["test_provenance"] = json.loads((run / "test_provenance.json").read_text())
            save_json(summary, run / "summary.json")
    collect(output, configs, protocol, "test")
    print(f"Saved ablation_val/test.csv, .md and .tex under {output}", flush=True)


if __name__ == "__main__":
    main()
