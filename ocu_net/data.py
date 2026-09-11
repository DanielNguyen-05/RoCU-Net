from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from .utils import seed_worker


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class SamplePair:
    sample_id: str
    image: Path
    mask: Path


def _indexed_files(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem
        if key in files:
            raise ValueError(f"Duplicate filename stem '{key}' in {directory}")
        files[key] = path
    return files


def discover_pairs(root: str | Path, image_dir: str = "images", mask_dir: str = "masks") -> list[SamplePair]:
    root = Path(root).expanduser().resolve()
    candidates = [root, root / "Kvasir-SEG"]
    selected: tuple[Path, Path] | None = None
    for base in candidates:
        images = base / image_dir
        masks = base / mask_dir
        if images.is_dir() and masks.is_dir():
            selected = images, masks
            break
    if selected is None:
        raise FileNotFoundError(
            "Kvasir-SEG was not found. Expected '<root>/images' and '<root>/masks' "
            f"under: {root}"
        )
    images, masks = selected
    image_index = _indexed_files(images)
    mask_index = _indexed_files(masks)
    common = sorted(image_index.keys() & mask_index.keys())
    if not common:
        raise RuntimeError(f"No image/mask pairs with matching filename stems in {images} and {masks}")
    missing_masks = sorted(image_index.keys() - mask_index.keys())
    missing_images = sorted(mask_index.keys() - image_index.keys())
    if missing_masks or missing_images:
        raise RuntimeError(
            f"Unpaired Kvasir files: {len(missing_masks)} image(s) without masks and "
            f"{len(missing_images)} mask(s) without images."
        )
    return [SamplePair(key, image_index[key], mask_index[key]) for key in common]


def _write_manifest(path: Path, pairs: Iterable[SamplePair], dataset_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "image", "mask"])
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "sample_id": pair.sample_id,
                    "image": str(pair.image.relative_to(dataset_root)),
                    "mask": str(pair.mask.relative_to(dataset_root)),
                }
            )


def _read_manifest(path: Path, dataset_root: Path) -> list[SamplePair]:
    pairs: list[SamplePair] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pairs.append(
                SamplePair(
                    sample_id=row["sample_id"],
                    image=(dataset_root / row["image"]).resolve(),
                    mask=(dataset_root / row["mask"]).resolve(),
                )
            )
    missing = [str(p.image) for p in pairs if not p.image.is_file()]
    missing += [str(p.mask) for p in pairs if not p.mask.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest points to {len(missing)} missing file(s); first: {missing[0]}")
    return pairs


def create_or_load_splits(
    dataset_root: str | Path,
    split_dir: str | Path,
    *,
    image_dir: str = "images",
    mask_dir: str = "masks",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[SamplePair]]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    # If the data are nested, use the actual common parent to make manifests portable.
    if not (dataset_root / image_dir).is_dir() and (dataset_root / "Kvasir-SEG" / image_dir).is_dir():
        dataset_root = dataset_root / "Kvasir-SEG"
    split_dir = Path(split_dir)
    paths = {name: split_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if all(path.is_file() for path in paths.values()):
        return {name: _read_manifest(path, dataset_root) for name, path in paths.items()}
    if any(path.exists() for path in paths.values()):
        raise RuntimeError(
            f"Incomplete split manifests in {split_dir}. Keep all three CSV files or remove the set."
        )
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    pairs = discover_pairs(dataset_root, image_dir=image_dir, mask_dir=mask_dir)
    if len(pairs) < 3:
        raise ValueError("At least three paired samples are required for train/val/test splits.")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_train = n_total - 2
        n_val = 1
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    for name, items in splits.items():
        _write_manifest(paths[name], items, dataset_root)
    return splits


class JointTransform:
    def __init__(self, image_size: tuple[int, int], augmentation: dict | None = None, train: bool = False):
        self.height, self.width = image_size
        self.augmentation = augmentation or {}
        self.train = train

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            if random.random() < self.augmentation.get("horizontal_flip_probability", 0.0):
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < self.augmentation.get("vertical_flip_probability", 0.0):
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if random.random() < self.augmentation.get("rotate_90_probability", 0.0):
                angle = random.choice((90, 180, 270))
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
                mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
            if random.random() < self.augmentation.get("small_rotation_probability", 0.0):
                limit = float(self.augmentation.get("max_rotation_degrees", 15))
                angle = random.uniform(-limit, limit)
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
            if random.random() < self.augmentation.get("color_jitter_probability", 0.0):
                brightness = float(self.augmentation.get("brightness", 0.0))
                contrast = float(self.augmentation.get("contrast", 0.0))
                saturation = float(self.augmentation.get("saturation", 0.0))
                image = ImageEnhance.Brightness(image).enhance(random.uniform(1 - brightness, 1 + brightness))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(1 - contrast, 1 + contrast))
                image = ImageEnhance.Color(image).enhance(random.uniform(1 - saturation, 1 + saturation))

        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        mask = mask.resize((self.width, self.height), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
        mask_array = (np.asarray(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask_array[None, ...].copy())
        return image_tensor, mask_tensor


class KvasirSegDataset(Dataset):
    def __init__(self, pairs: list[SamplePair], transform: JointTransform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        with Image.open(pair.image) as handle:
            image = handle.convert("RGB")
        with Image.open(pair.mask) as handle:
            mask = handle.convert("L")
        image_tensor, mask_tensor = self.transform(image, mask)
        return {"image": image_tensor, "mask": mask_tensor, "id": pair.sample_id}


def build_dataloaders(
    config: dict,
    splits: dict[str, list[SamplePair]],
    *,
    training: bool = True,
) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    image_size = tuple(int(v) for v in data_cfg["image_size"])
    train_dataset = KvasirSegDataset(
        splits["train"],
        JointTransform(image_size, config.get("augmentation"), train=training),
    )
    eval_transform = JointTransform(image_size, train=False)
    datasets = {
        "train": train_dataset,
        "val": KvasirSegDataset(splits["val"], eval_transform),
        "test": KvasirSegDataset(splits["test"], eval_transform),
    }
    workers = int(data_cfg.get("num_workers", 4))
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    common = {
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "persistent_workers": bool(data_cfg.get("persistent_workers", True)) and workers > 0,
        "worker_init_fn": seed_worker,
    }
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(train_cfg["batch_size"]),
            shuffle=training,
            drop_last=False,
            generator=generator,
            **common,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
            shuffle=False,
            drop_last=False,
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
            shuffle=False,
            drop_last=False,
            **common,
        ),
    }
