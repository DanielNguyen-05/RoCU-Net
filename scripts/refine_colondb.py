"""Fine-tune one existing ColonDB checkpoint, keeping its original split and model."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocu_net.compat import load_checkpoint
from rocu_net.config import PROJECT_ROOT, save_config
from rocu_net.data import create_or_load_splits


def refinement_config(checkpoint: Path, output: Path, epochs: int, data_root=None) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    checkpoint, output = checkpoint.resolve(), output.resolve()
    if output == checkpoint.parent:
        raise ValueError("Refinement output must differ from the source run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refinement output must be empty: {output}")
    config = load_checkpoint(checkpoint)["config"]
    source_splits = checkpoint.parent / "splits"
    for split in ("train", "val", "test"):
        if not (source_splits / f"{split}.csv").is_file():
            raise FileNotFoundError(f"Missing original {split} manifest in {source_splits}")
    if data_root is not None:
        config["data"]["root"] = str(Path(data_root).resolve())
    config["data"]["split_source"] = str(source_splits)
    config["experiment"].update(name=output.name, output_dir=str(output.parent))
    config["model"]["pretrained"] = False
    config["training"].update(
        init_checkpoint=str(checkpoint), epochs=epochs, learning_rate=5e-5,
        backbone_lr_multiplier=0.1, warmup_epochs=min(3, epochs - 1),
        early_stopping_patience=15, freeze_encoder_bn=True,
        multi_scale_factors=[0.9, 1.0, 1.1, 1.2], test_after_training=True,
    )
    # Use the inference policy supported by the validation-only comparison.
    config["inference"] = {"tta": "flip"}
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/rocu_cvc_colondb_refined_seed42")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    config = refinement_config(args.checkpoint, args.output, args.epochs, args.data_root)
    # Validate and copy manifests before launching training; never create a new split.
    data = config["data"]
    root = Path(data["root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    create_or_load_splits(root, args.output.resolve() / "splits",
                         source_split_dir=data["split_source"], seed=config["experiment"]["seed"])
    config_path = args.output.resolve() / "refinement_config.yaml"
    save_config(config, config_path)
    print(f"Fine-tuning {args.checkpoint}; config: {config_path}", flush=True)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "train.py"),
                    "--config", str(config_path), "--device", args.device],
                   cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
