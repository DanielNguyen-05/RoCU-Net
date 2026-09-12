from collections import OrderedDict
from copy import deepcopy

import torch
import yaml

from rocu_net.compat import load_checkpoint, normalize_config
from rocu_net.config import load_config, save_config
from rocu_net.model import RoCUNet, build_model
from scripts.migrate_rocu_names import migrate


def test_legacy_checkpoint_predictions_and_new_state_keys(tmp_path):
    config = {"experiment": {"name": "crs_ocu_example_seed42"}, "model": {
        "architecture": "crs_ocu_net", "backbone": "efficient", "pretrained": False,
        "backbone_channels": [8, 12, 16, 24], "backbone_depths": [1, 1, 1, 1],
        "decoder_channels": [20, 16], "carrier_channels": 8, "shallow_guide_channels": 8,
        "ocu_solver_iterations": 28, "ocu_epsilon": 1e-5,
    }}
    original_config = deepcopy(config)
    model = build_model(config).eval()
    assert isinstance(model, RoCUNet)
    assert model.rocu1.solver_iterations == 28
    legacy_state = OrderedDict((k.replace("rocu1.", "crs1.").replace("rocu2.", "crs2."), v)
                               for k, v in model.state_dict().items())
    checkpoint_path = tmp_path / "old.pt"
    torch.save({"config": config, "model": legacy_state}, checkpoint_path)
    checkpoint_bytes = checkpoint_path.read_bytes()
    loaded = load_checkpoint(checkpoint_path)
    restored = build_model(loaded["config"]).eval()
    restored.load_state_dict(loaded["model"], strict=True)
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected, actual = model(image), restored(image)
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key
    assert loaded["config"]["model"]["architecture"] == "rocu_net"
    assert loaded["config"]["experiment"]["name"] == "rocu_example_seed42"
    assert loaded["config"]["model"]["epsilon"] == 1e-5
    assert config == original_config
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    assert all(not k.startswith("crs") for k in restored.state_dict())
    restored.load_state_dict(model.state_dict(), strict=True)
    saved_config = tmp_path / "config.yaml"
    save_config(config, saved_config)
    assert "crs" not in saved_config.read_text()


def test_legacy_config_inheritance_preserves_solver_override(tmp_path):
    base = {"model": {"architecture": "crs_ocu_net", "ocu_solver_iterations": 24}}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({
        "base_config": "base.yaml", "model": {"ocu_solver_iterations": 31}}))
    result = load_config(tmp_path / "child.yaml")
    assert result["model"] == {"architecture": "rocu_net", "solver_iterations": 31}
    assert normalize_config({"model": {"architecture": "RoCU-Net"}})["model"]["architecture"] == "rocu_net"


def test_output_migration_preserves_checkpoint_and_refuses_collisions(tmp_path):
    source = tmp_path / "runs" / "crs_ocu_example_seed42"
    source.mkdir(parents=True)
    (source / "best.pt").write_bytes(b"checkpoint must remain unchanged")
    (source / "config_resolved.yaml").write_text(yaml.safe_dump({
        "experiment": {"name": source.name}, "model": {"architecture": "crs_ocu_net"}}))
    destination = source.with_name("rocu_example_seed42")
    destination.mkdir()
    original_config = (source / "config_resolved.yaml").read_bytes()
    try:
        migrate(tmp_path, apply=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Migration must refuse to overwrite a run")
    assert (source / "config_resolved.yaml").read_bytes() == original_config
    destination.rmdir()
    assert len(migrate(tmp_path)) == 2
    assert source.is_dir() and not destination.exists()
    migrate(tmp_path, apply=True)
    assert not source.exists()
    assert (destination / "best.pt").read_bytes() == b"checkpoint must remain unchanged"
    assert "crs" not in (destination / "config_resolved.yaml").read_text()
    assert migrate(tmp_path, apply=True) == []
