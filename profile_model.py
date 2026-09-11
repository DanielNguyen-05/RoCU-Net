from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ocu_net.config import load_config
from ocu_net.model import build_model
from ocu_net.profiler import profile_model
from ocu_net.utils import get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile RoCU efficiency")
    parser.add_argument("--config", default="configs/kvasir.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = build_model(config)
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        checkpoint = load_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
    device = get_device(args.device)
    data_cfg = config["data"]
    profile_cfg = config.get("profiling", {})
    result = profile_model(
        model,
        device,
        image_size=tuple(int(v) for v in data_cfg["image_size"]),
        batch_size=int(profile_cfg.get("batch_size", 1)),
        in_channels=int(config["model"].get("in_channels", 3)),
        warmup_iterations=int(profile_cfg.get("warmup_iterations", 30)),
        benchmark_iterations=int(profile_cfg.get("benchmark_iterations", 100)),
        checkpoint_path=checkpoint_path,
    )
    if args.output:
        save_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
