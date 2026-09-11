from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = (
    "configs/kvasir_crs.yaml",
    "configs/ablations/crs_no_routing.yaml",
    "configs/ablations/crs_no_semantic_carrier.yaml",
    "configs/ablations/crs_no_boundary_supervision.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CRS-OCU-Net ablation suite")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    for config in args.configs:
        command = [
            sys.executable,
            str(project_root / "train.py"),
            "--config",
            str((project_root / config).resolve()),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        if args.data_root:
            command.extend(["--data-root", str(Path(args.data_root).expanduser().resolve())])
        subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()

