from pathlib import Path

import numpy as np
from PIL import Image

from rocu_net.data import JointTransform, create_or_load_splits, discover_pairs
from rocu_net.config import load_config
from unittest.mock import patch


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


def test_colondb_config_inherits_selected_rotation_fix_settings():
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs/cvc_colondb.yaml")
    assert config["model"]["architecture"] == "rocu_net"
    assert config["model"]["routing_enabled"] is True
    assert config["experiment"]["name"] == "rocu_cvc_colondb_rotation_fix_seed42"
    assert config["data"]["split_source"] == "runs/rocu_cvc_colondb_seed42/splits"
    assert config["augmentation"].get("random_crop_probability", 0) == 0
    assert config["loss"].get("tversky_weight", 0) == 0
    assert config["data"]["image_size"] == load_config(project_root / "configs/kvasir.yaml")["data"]["image_size"]


def test_clinicdb_png_and_tif(tmp_path):
    from rocu_net.data import PolypSegDataset

    for folder, extension in (("PNG", ".png"), ("TIF", ".tif")):
        root = tmp_path / folder
        (root / "Original").mkdir(parents=True)
        (root / "Ground Truth").mkdir()
        for index in range(10):
            Image.new("RGB", (48, 32), (120, 80, 60)).save(root / "Original" / f"{index}{extension}")
            mask = np.zeros((32, 48), dtype=np.uint8)
            mask[8:24, 12:36] = 255
            Image.fromarray(mask).save(root / "Ground Truth" / f"{index}{extension}")
        kwargs = dict(image_dir="Original", mask_dir="Ground Truth")
        splits = create_or_load_splits(root, tmp_path / f"splits_{folder}", **kwargs)
        ids = [{p.sample_id for p in splits[name]} for name in ("train", "val", "test")]
        assert list(map(len, ids)) == [8, 1, 1]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        reused = create_or_load_splits(root, tmp_path / f"splits_{folder}", seed=99, **kwargs)
        assert splits == reused
        sample = PolypSegDataset(splits["train"], JointTransform((32, 32)))[0]
        assert sample["image"].shape == (3, 32, 32)
        assert set(sample["mask"].unique().tolist()) == {0.0, 1.0}


def test_right_angle_rotation_preserves_edge_polyp():
    mask = Image.new("L", (60, 20), 0)
    mask.paste(255, (0, 0, 8, 8))
    image = mask.convert("RGB")
    transform = JointTransform((60, 20), {"rotate_90_probability": 1.0}, train=True)
    with patch("rocu_net.data.random.choice", return_value=Image.Transpose.ROTATE_90):
        _, actual = transform(image, mask)
    expected = np.asarray(mask.transpose(Image.Transpose.ROTATE_90)) > 127
    assert np.array_equal(actual[0].numpy(), expected)
    assert actual.sum() == 64
    assert not (np.asarray(mask.rotate(90)) > 127).any()  # Original bug loses this polyp.


def test_foreground_crop_alignment_and_retention():
    mask = Image.new("L", (60, 40), 0)
    mask.paste(255, (25, 15, 35, 25))
    transform = JointTransform((32, 32), {"crop_scale": [0.5, 0.8], "crop_min_foreground_retained": 0.75}, train=True)
    for _ in range(10):
        image, cropped = transform._random_crop(mask.convert("RGB"), mask)
        actual = np.asarray(cropped) > 127
        assert actual.sum() >= 75
        assert np.array_equal(np.asarray(image)[:, :, 0], np.asarray(cropped))


def test_finetune_reuses_original_splits_and_rejects_overlap(tmp_path):
    root = tmp_path / "data"
    _make_tiny_kvasir(root)
    source = tmp_path / "original"
    original = create_or_load_splits(root, source, seed=42)
    reused = create_or_load_splits(root, tmp_path / "new", seed=13, source_split_dir=source)
    assert original == reused
    import shutil
    shutil.copyfile(source / "val.csv", source / "test.csv")
    try:
        create_or_load_splits(root, tmp_path / "bad", source_split_dir=source)
    except ValueError as error:
        assert "overlapping" in str(error)
    else:
        raise AssertionError("Overlapping source splits accepted")


def test_identical_images_stay_in_one_split(tmp_path):
    import shutil
    from rocu_net.data import image_content_hash
    root = tmp_path / "data"
    _make_tiny_kvasir(root, count=10)
    shutil.copyfile(root / "images/case_00.jpg", root / "images/duplicate.jpg")
    shutil.copyfile(root / "masks/case_00.png", root / "masks/duplicate.png")
    splits = create_or_load_splits(root, tmp_path / "splits", group_identical_images=True)
    owners = {}
    for name, pairs in splits.items():
        for pair in pairs:
            key = image_content_hash(pair.image)
            assert key not in owners or owners[key] == name
            owners[key] = name
    assert sum(map(len, splits.values())) == 11
    assert create_or_load_splits(root, tmp_path / "splits", group_identical_images=True) == splits
