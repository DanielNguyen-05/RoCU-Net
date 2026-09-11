from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic Kvasir-like smoke dataset")
    parser.add_argument("--output", default="dataset/ToyKvasir")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_sample(size: int, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    yy, xx = np.mgrid[:size, :size]
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 85 + 35 * xx / max(size - 1, 1)
    base[..., 1] = 28 + 18 * yy / max(size - 1, 1)
    base[..., 2] = 25
    noise = np.asarray(
        [rng.gauss(0.0, 7.0) for _ in range(size * size * 3)], dtype=np.float32
    ).reshape(size, size, 3)
    image = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), mode="RGB")
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    cx = rng.randint(size // 4, 3 * size // 4)
    cy = rng.randint(size // 4, 3 * size // 4)
    rx = rng.randint(max(5, size // 10), max(7, size // 4))
    ry = rng.randint(max(5, size // 12), max(7, size // 4))
    points = []
    for index in range(48):
        angle = 2.0 * math.pi * index / 48.0
        radius = 1.0 + rng.uniform(-0.12, 0.12)
        points.append(
            (
                cx + radius * rx * math.cos(angle),
                cy + radius * ry * math.sin(angle),
            )
        )
    draw.polygon(points, fill=255)
    polyp = Image.new("RGB", (size, size), (178, 85, 75)).filter(ImageFilter.GaussianBlur(2.0))
    image = Image.composite(polyp, image, mask)
    return image, mask


def main() -> None:
    args = parse_args()
    if args.count < 3:
        raise ValueError("count must be at least 3")
    if args.size < 32 or args.size % 16 != 0:
        raise ValueError("size must be >=32 and divisible by 16")
    root = Path(args.output).expanduser().resolve()
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for index in range(args.count):
        image, mask = make_sample(args.size, rng)
        sample_id = f"toy_{index:04d}"
        image.save(image_dir / f"{sample_id}.jpg", quality=92)
        mask.save(mask_dir / f"{sample_id}.png")
    print(f"Created {args.count} paired samples under {root}")


if __name__ == "__main__":
    main()

