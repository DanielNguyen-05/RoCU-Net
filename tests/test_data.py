from pathlib import Path

import numpy as np
from PIL import Image

from ocu_net.data import JointTransform, create_or_load_splits, discover_pairs
from ocu_net.config import load_config


def _make_tiny_kvasir(root: Path, count: int = 10):
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    for index in range(count):
        image = np.full((20, 24, 3), index * 10, dtype=np.uint8)
        mask = np.zeros((20, 24), dtype=np.uint8)
        mask[4:14, 5:17] = 255
        Image.fromarray(image).save(root / "images" / f"case_{index:02d}.jpg")
        Image.fromarray(mask).save(root / "masks" / f"case_{index:02d}.png")


def test_discovery_split_reuse_and_transform(tmp_path):
    data_root = tmp_path / "Kvasir"
    _make_tiny_kvasir(data_root)
    assert len(discover_pairs(data_root)) == 10
    split_dir = tmp_path / "splits"
    first = create_or_load_splits(data_root, split_dir, seed=42)
    second = create_or_load_splits(data_root, split_dir, seed=999)
    assert {key: [p.sample_id for p in value] for key, value in first.items()} == {
        key: [p.sample_id for p in value] for key, value in second.items()
    }
    assert [len(first[name]) for name in ("train", "val", "test")] == [8, 1, 1]
    with Image.open(first["train"][0].image) as handle:
        image = handle.convert("RGB")
    with Image.open(first["train"][0].mask) as handle:
        mask = handle.convert("L")
    image_tensor, mask_tensor = JointTransform((32, 32))(image, mask)
    assert image_tensor.shape == (3, 32, 32)
    assert mask_tensor.shape == (1, 32, 32)
    assert set(mask_tensor.unique().tolist()).issubset({0.0, 1.0})


def test_ablation_config_inherits_base_settings():
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs/ablations/crs_no_routing.yaml")
    assert config["model"]["architecture"] == "crs_ocu_net"
    assert config["model"]["routing_enabled"] is False
    assert config["data"]["image_size"] == [256, 256]
