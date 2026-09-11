from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ocu_net.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate OCU-Net results across random seeds")
    parser.add_argument("runs", nargs="+", help="Run directories or summary.json files")
    parser.add_argument("--output", default="runs/kvasir_multiseed")
    return parser.parse_args()


def resolve_summary(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    return candidate / "summary.json" if candidate.is_dir() else candidate


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for source in args.runs:
        path = resolve_summary(source)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        test = payload.get("held_out_test")
        if not test:
            raise ValueError(f"No held-out test results in {path}")
        lightweight = payload["lightweight"]
        model_info = payload.get("model", {})
        routing = lightweight.get("confidence_routing", {})
        row = {
            "run": str(path.parent),
            "experiment": payload["experiment"],
            "architecture": model_info.get("architecture", "unknown"),
            "backbone": model_info.get("backbone"),
            "best_epoch": payload["best_epoch"],
            **{f"test_{key}": value for key, value in test["mean"].items()},
            "params_m": lightweight["parameters"]["millions"],
            "gmacs": lightweight["computation"]["gmacs_per_image"],
            "latency_ms": lightweight["runtime"]["latency_mean_ms_per_batch"],
            "fps": lightweight["runtime"]["images_per_second"],
            "peak_cuda_mb": lightweight["memory"]["peak_cuda_allocated_mb"],
            "routing_stage1_active": routing.get("stage1_active_fraction"),
            "routing_stage2_active": routing.get("stage2_active_fraction"),
            "solver_skip_fraction": routing.get("solver_skip_fraction_estimate"),
        }
        rows.append(row)
    frame = pd.DataFrame.from_records(rows)
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_seed.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number])
    aggregate = {
        "n_runs": len(frame),
        "runs": rows,
        "mean": {key: float(value) for key, value in numeric.mean().items()},
        "std": {key: float(value) for key, value in numeric.std(ddof=0).items()},
    }
    save_json(aggregate, output_dir / "aggregate.json")
    print(frame.to_string(index=False))
    print(f"Saved aggregate report to {output_dir}")


if __name__ == "__main__":
    main()
