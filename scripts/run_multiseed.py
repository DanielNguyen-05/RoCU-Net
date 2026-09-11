from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CRS-OCU-Net with several random seeds")
    parser.add_argument("--config", default="configs/kvasir_crs.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 87])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--prefix", default="crs_ocu_kvasir")
    parser.add_argument("--no-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path
    run_directories: list[str] = []
    for seed in args.seeds:
        name = f"{args.prefix}_seed{seed}"
        command = [
            sys.executable,
            str(project_root / "train.py"),
            "--config",
            str(config_path.resolve()),
            "--seed",
            str(seed),
            "--name",
            name,
            "--device",
            args.device,
        ]
        if args.data_root:
            command.extend(["--data-root", str(Path(args.data_root).expanduser().resolve())])
        if args.no_test:
            command.append("--no-test")
        subprocess.run(command, cwd=project_root, check=True)
        run_directories.append(str(project_root / "runs" / name))
    if not args.no_test:
        output = project_root / "runs" / f"{args.prefix}_3seed_summary"
        subprocess.run(
            [
                sys.executable,
                str(project_root / "aggregate_results.py"),
                *run_directories,
                "--output",
                str(output),
            ],
            cwd=project_root,
            check=True,
        )


if __name__ == "__main__":
    main()
