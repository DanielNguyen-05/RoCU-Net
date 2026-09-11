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

from ocu_net.data import discover_pairs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Kvasir-SEG directory")
    parser.add_argument("--root", default="dataset/Kvasir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = discover_pairs(args.root)
    image_sizes: Counter[tuple[int, int]] = Counter()
    foreground_ratios: list[float] = []
    for pair in pairs:
        with Image.open(pair.image) as image:
            image.verify()
        with Image.open(pair.image) as image:
            image_sizes[image.size] += 1
        with Image.open(pair.mask) as mask:
            mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
        foreground_ratios.append(float((mask_array > 127).mean()))
    print(f"root={Path(args.root).expanduser().resolve()}")
    print(f"pairs={len(pairs)} unique_sizes={len(image_sizes)}")
    print(
        "foreground_ratio: "
        f"min={min(foreground_ratios):.4f} "
        f"mean={np.mean(foreground_ratios):.4f} "
        f"max={max(foreground_ratios):.4f}"
    )


if __name__ == "__main__":
    main()
