from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rocu_net.config import load_config, resolve_project_path
from rocu_net.data import discover_pairs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paired polyp images and masks")
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_config(args.config)["data"] if args.config else {}
    root = resolve_project_path(args.root or data.get("root", "dataset/Kvasir-SEG"))
    pairs = discover_pairs(
        root, args.image_dir or data.get("image_dir", "images"),
        args.mask_dir or data.get("mask_dir", "masks"),
    )
    image_sizes: Counter[tuple[int, int]] = Counter()
    foreground_ratios: list[float] = []
    for pair in pairs:
        with Image.open(pair.image) as image:
            image.verify()
        with Image.open(pair.image) as image:
            image_sizes[image.size] += 1
            image_size = image.size
        with Image.open(pair.mask) as mask:
            if mask.size != image_size:
                raise ValueError(f"Image/mask size mismatch for {pair.sample_id}: {image_size} vs {mask.size}")
            mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
        foreground_ratios.append(float((mask_array > 127).mean()))
    print(f"root={root}")
    print(f"pairs={len(pairs)} unique_sizes={len(image_sizes)}")
    print(
        "foreground_ratio: "
        f"min={min(foreground_ratios):.4f} "
        f"mean={np.mean(foreground_ratios):.4f} "
        f"max={max(foreground_ratios):.4f}"
    )


if __name__ == "__main__":
    main()
