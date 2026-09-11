from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-ready table from run summaries")
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run directory, summary.json, or label=path",
    )
    parser.add_argument("--output", default="runs/model_comparison")
    return parser.parse_args()


def parse_source(source: str) -> tuple[str | None, Path]:
    if "=" in source:
        label, raw_path = source.split("=", 1)
    else:
        label, raw_path = None, source
    path = Path(raw_path).expanduser().resolve()
    if path.is_dir():
        path = path / "summary.json"
    return label, path


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for source in args.runs:
        label, path = parse_source(source)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        test = payload.get("held_out_test")
        if not test:
            raise ValueError(f"Missing held_out_test in {path}")
        quality = test["mean"]
        lightweight = payload["lightweight"]
        routing = lightweight.get("confidence_routing", {})
        conservation = test.get("occupancy_conservation", {})
        rows.append(
            {
                "Method": label or payload.get("experiment", path.parent.name),
                "Dice": quality.get("dice"),
                "IoU": quality.get("iou"),
                "Precision": quality.get("precision"),
                "Recall": quality.get("recall"),
                "Boundary F1": quality.get("boundary_f1"),
                "MAE": quality.get("mae"),
                "Params (M)": lightweight["parameters"].get("millions"),
                "GMACs": lightweight["computation"].get("gmacs_per_image"),
                "GFLOPs (2xMAC)": lightweight["computation"].get("gflops_per_image"),
                "Latency (ms)": lightweight["runtime"].get("latency_mean_ms_per_batch"),
                "FPS": lightweight["runtime"].get("images_per_second"),
                "Peak CUDA (MB)": lightweight["memory"].get("peak_cuda_allocated_mb"),
                "Solver skip": routing.get("solver_skip_fraction_estimate"),
                "Conservation MAE full": conservation.get("p352_to_p176_mae"),
                "Conservation MAE half": conservation.get("p176_to_p88_mae"),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "comparison.csv", index=False)
    with (output / "comparison.md").open("w", encoding="utf-8") as handle:
        columns = list(frame.columns)
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in frame.itertuples(index=False, name=None):
            formatted = []
            for value in row:
                if pd.isna(value):
                    formatted.append("")
                elif isinstance(value, float):
                    formatted.append(
                        f"{value:.3e}" if 0.0 < abs(value) < 1e-3 else f"{value:.4f}"
                    )
                else:
                    formatted.append(str(value))
            handle.write("| " + " | ".join(formatted) + " |\n")
    print(frame.to_string(index=False))
    print(f"Saved tables to {output}")


if __name__ == "__main__":
    main()
