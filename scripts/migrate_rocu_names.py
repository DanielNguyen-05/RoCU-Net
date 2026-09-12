"""Rename existing experiment outputs to RoCU-Net without changing checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocu_net.compat import canonical_run_name, normalize_config
from rocu_net.config import save_config


def migrate(root: Path, *, apply: bool = False) -> list[str]:
    root = Path(root).resolve()
    renames = []
    for folder in ("runs", "logs", "figures", "reports"):
        directory = root / folder
        if directory.is_dir():
            for source in sorted(directory.iterdir()):
                destination = source.with_name(canonical_run_name(source.name))
                if source != destination:
                    renames.append((source, destination))
    # Check the whole plan before writing anything, including saved configs.
    destinations = set()
    for source, destination in renames:
        if destination.exists() or destination.is_symlink() or destination in destinations:
            raise FileExistsError(f"Cannot rename {source}: destination already exists: {destination}")
        destinations.add(destination)
    updates = []
    for path in sorted((root / "runs").glob("*/config_resolved.yaml")):
        with path.open(encoding="utf-8") as handle:
            original = yaml.safe_load(handle)
        normalized = normalize_config(original)
        if original != normalized:
            updates.append((path, normalized))
    actions = [f"Normalize {p.relative_to(root)}" for p, _ in updates]
    actions += [f"Rename {s.relative_to(root)} -> {d.relative_to(root)}" for s, d in renames]
    if apply:
        for path, config in updates:
            save_config(config, path)
        for source, destination in renames:
            source.rename(destination)
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true", help="Apply changes; default prints the plan")
    args = parser.parse_args()
    actions = migrate(args.root, apply=args.apply)
    for action in actions:
        print(action)
    print(f"{'Applied' if args.apply else 'Planned'} {len(actions)} changes.")


if __name__ == "__main__":
    main()
