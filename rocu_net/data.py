from __future__ import annotations

import csv
import hashlib
import random
import shutil
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


def image_content_hash(path: Path) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        digest = hashlib.sha256(str(image.size).encode())
        digest.update(image.tobytes())
    return digest.hexdigest()


def _validate_image_group_splits(splits: dict[str, list[SamplePair]]) -> None:
    owners: dict[str, str] = {}
    for split, pairs in splits.items():
        for pair in pairs:
            key = image_content_hash(pair.image)
            if key in owners and owners[key] != split:
                raise ValueError("Identical image pixels cross existing splits; use a new run for grouped splitting")
            owners[key] = split


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
            f"Dataset was not found. Expected {image_dir!r} and {mask_dir!r} "
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
            f"Unpaired dataset files: {len(missing_masks)} image(s) without masks and "
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
    source_split_dir: str | Path | None = None,
    group_identical_images: bool = False,
) -> dict[str, list[SamplePair]]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    # If the data are nested, use the actual common parent to make manifests portable.
    if not (dataset_root / image_dir).is_dir() and (dataset_root / "Kvasir-SEG" / image_dir).is_dir():
        dataset_root = dataset_root / "Kvasir-SEG"
    split_dir = Path(split_dir)
    paths = {name: split_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if source_split_dir is not None:
        source = Path(source_split_dir).expanduser().resolve()
        source_paths = {name: source / f"{name}.csv" for name in paths}
        if not all(p.is_file() for p in source_paths.values()):
            raise FileNotFoundError(f"Expected all three existing split manifests in {source}")
        original = {name: _read_manifest(p, dataset_root) for name, p in source_paths.items()}
        if group_identical_images:
            _validate_image_group_splits(original)
        seen: set[str] = set()
        for name, samples in original.items():
            ids = [p.sample_id for p in samples]
            if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError(f"Empty, duplicate or overlapping {name} IDs in {source}")
            seen.update(ids)
        if any(p.exists() for p in paths.values()):
            if not all(p.is_file() for p in paths.values()) or any(
                _read_manifest(paths[name], dataset_root) != original[name] for name in paths
            ):
                raise ValueError(f"Existing splits in {split_dir} differ from {source}")
        else:
            split_dir.mkdir(parents=True, exist_ok=True)
            for name, p in paths.items():
                shutil.copy2(source_paths[name], p)
        return original
    if all(path.is_file() for path in paths.values()):
        loaded = {name: _read_manifest(path, dataset_root) for name, path in paths.items()}
        if group_identical_images:
            _validate_image_group_splits(loaded)
        return loaded
    if any(path.exists() for path in paths.values()):
        raise RuntimeError(
            f"Incomplete split manifests in {split_dir}. Keep all three CSV files or remove the set."
        )
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    pairs = discover_pairs(dataset_root, image_dir=image_dir, mask_dir=mask_dir)
    if len(pairs) < 3:
        raise ValueError("At least three paired samples are required for train/val/test splits.")
    if group_identical_images:
        groups: dict[str, list[SamplePair]] = {}
        for pair in pairs:
            groups.setdefault(image_content_hash(pair.image), []).append(pair)
        shuffled = list(groups.values())
        if len(shuffled) < 3:
            raise ValueError("At least three distinct images are required for grouped splits")
    else:
        shuffled = [[pair] for pair in pairs]
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_train = n_total - 2
        n_val = 1
    splits = {
        "train": [p for group in shuffled[:n_train] for p in group],
        "val": [p for group in shuffled[n_train : n_train + n_val] for p in group],
        "test": [p for group in shuffled[n_train + n_val :] for p in group],
    }
    for name, items in splits.items():
        _write_manifest(paths[name], items, dataset_root)
    return splits


class JointTransform:
    def __init__(self, image_size: tuple[int, int], augmentation: dict | None = None, train: bool = False):
        self.height, self.width = image_size
        self.augmentation = augmentation or {}
        self.train = train

    def _random_crop(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Scale augmentation with a minimum retained foreground fraction.

        Coordinates come exclusively from this training image/mask. Failed
        attempts fall back to the full image, including for edge-touching polyps.
        """
        minimum, maximum = self.augmentation.get("crop_scale", (0.5, 1.0))
        retain = float(self.augmentation.get("crop_min_foreground_retained", 0.75))
        if not 0 < minimum <= maximum <= 1 or not 0 <= retain <= 1:
            raise ValueError("crop_scale must lie in (0, 1] and foreground retention in [0, 1]")
        width, height = image.size
        foreground = np.asarray(mask) > 127
        total = int(foreground.sum())
        for _ in range(10):
            scale = random.uniform(minimum, maximum) ** 0.5
            crop_w, crop_h = max(1, round(width * scale)), max(1, round(height * scale))
            left, top = random.randint(0, width - crop_w), random.randint(0, height - crop_h)
            box = (left, top, left + crop_w, top + crop_h)
            kept = foreground[top:top + crop_h, left:left + crop_w].sum()
            if not total or kept >= retain * total:
                return image.crop(box), mask.crop(box)
        return image, mask

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            if self.augmentation.get("random_crop_probability", 0.0) > 0 and random.random() < self.augmentation["random_crop_probability"]:
                image, mask = self._random_crop(image, mask)
            if random.random() < self.augmentation.get("horizontal_flip_probability", 0.0):
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < self.augmentation.get("vertical_flip_probability", 0.0):
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if random.random() < self.augmentation.get("rotate_90_probability", 0.0):
                rotation = random.choice((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270))
                image = image.transpose(rotation)
                mask = mask.transpose(rotation)
            if random.random() < self.augmentation.get("small_rotation_probability", 0.0):
                limit = float(self.augmentation.get("max_rotation_degrees", 15))
                angle = random.uniform(-limit, limit)
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)
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


class PolypSegDataset(Dataset):
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


# Backward-compatible import for existing Kvasir experiments.
KvasirSegDataset = PolypSegDataset


def build_dataloaders(
    config: dict,
    splits: dict[str, list[SamplePair]],
    *,
    training: bool = True,
) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    image_size = tuple(int(v) for v in data_cfg["image_size"])
    train_dataset = PolypSegDataset(
        splits["train"],
        JointTransform(image_size, config.get("augmentation"), train=training),
    )
    eval_transform = JointTransform(image_size, train=False)
    datasets = {
        "train": train_dataset,
        "val": PolypSegDataset(splits["val"], eval_transform),
        "test": PolypSegDataset(splits["test"], eval_transform),
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
