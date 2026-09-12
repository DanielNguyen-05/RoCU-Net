from __future__ import annotations

import argparse
import json
from pathlib import Path

from rocu_net.compat import load_checkpoint
from rocu_net.config import load_config
from rocu_net.model import build_model
from rocu_net.inference import inference_model
from rocu_net.profiler import profile_model
from rocu_net.utils import get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile RoCU-Net efficiency")
    parser.add_argument("--config", default="configs/kvasir.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--tta", choices=["none", "flip"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        checkpoint = load_checkpoint(checkpoint_path)
        config = checkpoint["config"]
        config["model"]["pretrained"] = False
    model = build_model(config)
    if checkpoint_path is not None:
        model.load_state_dict(checkpoint["model"])
    model = inference_model(model, config, args.tta)
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
