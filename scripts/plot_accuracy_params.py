"""Plot the user-supplied paper tables; no model inference or estimated metrics.

Run from any directory: python scripts/plot_accuracy_params.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("clinicdb", "colondb", "kvasir")
# Literal values from the two tables supplied by the user, including rounded
# parameter counts. Each metric tuple is (ClinicDB, ColonDB, Kvasir).
TABLE = [
    ("UNet", 7.8, (0.8638, 0.8387, 0.8440), (0.7649, 0.7425, 0.7327)),
    ("Attention-UNet", 8.7, (0.8839, 0.8487, 0.8512), (0.7955, 0.7523, 0.7449)),
    ("UNet++", 9.2, (0.8832, 0.8294, 0.8598), (0.7951, 0.7246, 0.7565)),
    ("UNeXt", 1.5, (0.8660, 0.7774, 0.8115), (0.7709, 0.6613, 0.6880)),
    ("MALUNet", 0.2, (0.6565, 0.6860, 0.7219), (0.4994, 0.5361, 0.5724)),
    ("EGEUNet", 0.0534, (0.6774, 0.7098, 0.6928), (0.5190, 0.5744, 0.5342)),
    ("U-KAN", 25.4, (0.9017, 0.8353, 0.8438), (0.8243, 0.7367, 0.7338)),
    ("MobileNetv3-Large-UNet", 2.0, (0.9087, 0.8700, 0.8747), (0.8347, 0.7802, 0.7804)),
    ("LV-UNet", 0.9, (0.9183, 0.8808, 0.8917), (0.8514, 0.7964, 0.8069)),
    ("RoCU-Net", 0.3, (0.9102, 0.8969, 0.9028), (0.8506, 0.8301, 0.8358)),
]
# Familiar U-Net references plus compact/recent comparators. Keep the strongest
# competing mean Dice (LV-UNet); the full source table is also exported to CSV.
SELECTED = ("UNet", "UNet++", "UNeXt", "U-KAN", "MobileNetv3-Large-UNet", "LV-UNet", "RoCU-Net")
STYLES = {
    "UNet": ("o", "#ed8080", (0, -14), "center"),
    "UNet++": ("s", "#166534", (12, 12), "right"),
    "UNeXt": ("D", "#527A91", (9, 4), "left"),
    "U-KAN": ("P", "#1d4ed8", (0, 12), "center"),
    "MobileNetv3-Large-UNet": ("v", "#57534e", (10, -1), "left"),
    "LV-UNet": ("^", "#B77729", (1, 7), "left"),
    "RoCU-Net": ("*", "#C43C39", (-14, 13), "left"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "accuracy_params")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, params, dice, iou in TABLE:
        row = {"model": name, "params_m": params, "included_in_figure": name in SELECTED}
        for metric, values in (("dice", dice), ("iou", iou)):
            row.update({f"{dataset}_{metric}": value for dataset, value in zip(DATASETS, values)})
            row[f"mean_{metric}"] = sum(values) / len(values)
        rows.append(row)
    with (args.output / "source_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    fig.subplots_adjust(left=0.115, right=0.965, bottom=0.16, top=0.96)
    ax.set_xscale("log")
    ax.set_xlim(0.22, 38)
    ax.set_ylim(79.8, 92.2)
    ax.set_xlabel("Parameters (M)", labelpad=9)
    ax.set_ylabel("Mean Dice (%)", labelpad=9)
    ax.xaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 5, 10, 25]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E2E6EA", linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9AA4AE")
    ax.tick_params(colors="#394551", length=3)

    for row in rows:
        name = row["model"]
        if name not in SELECTED:
            continue
        marker, color, offset, alignment = STYLES[name]
        ours = name == "RoCU-Net"
        x, y = row["params_m"], row["mean_dice"] * 100
        ax.scatter(x, y, s=245 if ours else 78, marker=marker, color=color,
                   edgecolors="white", linewidths=0.9, zorder=4)
        label = "MobileNetV3-Large\nUNet" if name == "MobileNetv3-Large-UNet" else name
        if ours:
            label = "RoCU-Net (ours)"
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                    ha=alignment, va="center", fontsize=10,
                    fontweight="bold" if ours else "normal",
                    color=color,
                    linespacing=1.35)

    for suffix in ("pdf", "svg", "png"):
        fig.savefig(args.output / f"accuracy_vs_params.{suffix}", dpi=args.dpi)
    plt.close(fig)

    caption = (
        "Parameter efficiency of selected segmentation models. The vertical axis reports "
        "the unweighted mean Dice (%) across CVC-ClinicDB, CVC-ColonDB, and Kvasir-SEG; "
        "the horizontal axis shows parameters in millions on a logarithmic scale. "
        "Points toward the upper left combine higher mean Dice with fewer parameters. "
        "RoCU-Net is highlighted in red. Values are taken from the supplied comparison "
        "and model-cost tables; parameter counts retain the rounding of those tables. "
        "The figure shows seven selected methods, not all methods in the tables.\n"
    )
    (args.output / "caption.txt").write_text(caption, encoding="utf-8")
    (args.output / "README.md").write_text(
        "# Accuracy–parameter figure\n\n"
        "Regenerate: `python scripts/plot_accuracy_params.py`\n\n"
        "Use the vector PDF for LaTeX, SVG for editing, or the 600-dpi PNG. "
        "The y-axis is mean Dice, not pixel accuracy. Each dataset contributes equally; "
        "this is not a pooled per-image average. Values come exclusively from the two "
        "user-supplied tables, without rerunning models or modifying measurements.\n\n"
        "Selected models: UNet, UNet++, UNeXt, U-KAN, MobileNetV3-Large-UNet, "
        "LV-UNet, and RoCU-Net. Attention-UNet is omitted to reduce crowding among "
        "U-Net variants; MALUNet and EGEUNet remain in the exported full source CSV. "
        "RoCU-Net has the highest mean Dice in the supplied table, but is not the "
        "smallest model in that full table and is not best on ClinicDB individually.\n\n"
        "These aggregate values do not provide seed uncertainty or establish matched "
        "training protocols. This figure compares parameters and Dice, not FLOPs or latency.\n\n"
        "## Suggested caption\n\n" + caption,
        encoding="utf-8",
    )
    for row in rows:
        if row["included_in_figure"]:
            print(f"{row['model']:<26} {row['params_m']:>5g} M  mean Dice {row['mean_dice'] * 100:.4f}%")
    print(f"Exported PDF, SVG, PNG, source CSV and caption to {args.output}")


if __name__ == "__main__":
    main()
