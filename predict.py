from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rocu_net.compat import load_checkpoint
from rocu_net.data import IMAGE_EXTENSIONS, JointTransform
from rocu_net.model import build_model
from rocu_net.inference import inference_model
from rocu_net.utils import autocast_context, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict polyp masks with RoCU-Net")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", default="predictions")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tta", choices=["none", "flip"], default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return parser.parse_args()


def discover_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    images = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise RuntimeError(f"No supported images found in {source}")
    return images


def save_outputs(
    image: Image.Image,
    probability: np.ndarray,
    destination: Path,
    sample_name: str,
    threshold: float,
) -> None:
    destination.joinpath("probability").mkdir(parents=True, exist_ok=True)
    destination.joinpath("binary").mkdir(parents=True, exist_ok=True)
    destination.joinpath("overlay").mkdir(parents=True, exist_ok=True)
    probability_image = Image.fromarray(
        np.clip(probability * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).resize(image.size, Image.Resampling.BILINEAR)
    probability_array = np.asarray(probability_image, dtype=np.float32) / 255.0
    binary_array = probability_array >= threshold
    binary_image = Image.fromarray(binary_array.astype(np.uint8) * 255, mode="L")

    original = np.asarray(image, dtype=np.float32)
    color = np.zeros_like(original)
    color[..., 0] = 255.0
    alpha = (0.42 * binary_array.astype(np.float32))[..., None]
    overlay = np.clip(original * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)

    probability_image.save(destination / "probability" / f"{sample_name}.png")
    binary_image.save(destination / "binary" / f"{sample_name}.png")
    Image.fromarray(overlay, mode="RGB").save(destination / "overlay" / f"{sample_name}.png")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    device = get_device(args.device)
    # The checkpoint already contains encoder weights; avoid an ImageNet download.
    model_config = {**config, "model": {**config["model"], "pretrained": False}}
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model = inference_model(model, config, args.tta)
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    transform = JointTransform(image_size, train=False)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(config["training"].get("threshold", 0.5))
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    images = discover_images(source)
    amp_enabled = bool(config["training"].get("eval_amp", config["training"].get("amp", True)) and device.type == "cuda")

    with torch.inference_mode():
        for index, path in enumerate(images, start=1):
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            dummy_mask = Image.new("L", image.size, 0)
            image_tensor, _ = transform(image, dummy_mask)
            with autocast_context(amp_enabled):
                probability = model(image_tensor.unsqueeze(0).to(device))["p352"]
            probability_array = probability[0, 0].float().cpu().numpy()
            save_outputs(image, probability_array, destination, path.stem, threshold)
            print(f"[{index}/{len(images)}] {path.name}")
    print(f"Saved predictions to {destination}")


if __name__ == "__main__":
    main()
