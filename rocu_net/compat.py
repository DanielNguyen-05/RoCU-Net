"""Read historical experiment names and checkpoints using the current API.

Legacy spellings live here only; new configs and state dictionaries use RoCU names.
Checkpoint files are never rewritten by the loader.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn


def canonical_run_name(name: str) -> str:
    for old in ("crs_ocu_", "crs-ocu-", "CRS-OCU-"):
        if name.startswith(old):
            return "rocu_" + name[len(old):]
    return name


def normalize_config(config: dict) -> dict:
    """Return a copy with canonical identifiers, preserving all numeric settings."""
    config = deepcopy(config)
    if "model" in config:
        model = config["model"]
        architecture = str(model.get("architecture", "rocu_net")).lower().replace("-", "_")
        if architecture in {"rocu", "rocunet", "crs_ocu", "crs_ocu_net", "crsocunet"}:
            architecture = "rocu_net"
        elif architecture in {"ocu", "ocu_net", "ocunet"}:
            architecture = "occupancy_unet"
        if "architecture" in model:
            model["architecture"] = architecture
        for old, new in (("ocu_solver_iterations", "solver_iterations"), ("ocu_epsilon", "epsilon")):
            if old in model:
                if new in model and model[new] != model[old]:
                    raise ValueError(f"Conflicting model settings: {old} and {new}")
                model[new] = model.pop(old)
    experiment = config.get("experiment", {})
    if "name" in experiment:
        experiment["name"] = canonical_run_name(str(experiment["name"]))
    return config


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if "config" in checkpoint:
        checkpoint["config"] = normalize_config(checkpoint["config"])
    return checkpoint


class CheckpointCompatibleModule(nn.Module):
    """Translate old stage prefixes while keeping strict weight validation."""

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        stages = {"crs1": "rocu1", "crs2": "rocu2", "ocu1": "occupancy1", "ocu2": "occupancy2"}

        def rename(key):
            stage, separator, tail = key.partition(".")
            return stages.get(stage, stage) + separator + tail

        translated = OrderedDict()
        for key, value in state_dict.items():
            new_key = rename(key)
            if new_key in translated:
                raise ValueError(f"Duplicate checkpoint parameter after migration: {new_key}")
            translated[new_key] = value
        if hasattr(state_dict, "_metadata"):
            translated._metadata = {rename(k): v for k, v in state_dict._metadata.items()}
        return super().load_state_dict(translated, strict=strict, **kwargs)
