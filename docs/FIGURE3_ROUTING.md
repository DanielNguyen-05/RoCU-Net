# Figure 3: routing on real images

This experiment requires no retraining. It evaluates **one full RoCU-Net
checkpoint** on the same original validation or test manifest in five modes:
dense, and routing thresholds 0.05, 0.10, 0.20, 0.30. Dense disables only
`hard_routing_inference`; soft gating remains enabled. A1/A2/A3 ablation
checkpoints are rejected.

For the compact dense-versus-routed table on the fixed test split, without
recreating the figure, run this command on the same GPU used for the throughput
report:

```bash
python scripts/figure3_routing.py \
  --checkpoint runs/rocu_kvasir_seed42/best.pt \
  --data-root dataset/Kvasir-SEG \
  --split test --device cuda --cpu-threads 6 \
  --thresholds 0.10 --illustration-threshold 0.10 \
  --warmup 30 --repeats 10 --batch-size 1 --seed 42 \
  --table-only \
  --output figures/kvasir_routing_test_gpu
```

Ten repeats form five complete two-mode timing cycles, so dense and routed each
occupy both timing positions equally. The command produces
`routing_dense_vs_routed.{csv,md,tex}`, a matching caption text file, the full
summary, every raw timing and per-image metric, plus checkpoint/data/source
hashes. It does not train, select a threshold, render a figure, or include
preprocessing and transfers in timing.

Run on the server from the project directory, with the device otherwise idle:

```bash
python scripts/figure3_routing.py \
  --checkpoint runs/rocu_kvasir_seed42/best.pt \
  --data-root dataset/Kvasir-SEG \
  --split val --device cuda \
  --warmup 30 --repeats 5 --batch-size 1 \
  --output figures/kvasir_routing
```

The original Kvasir checkpoint is supported. After A0 training completes, replace
`--checkpoint` with
`runs/rocu_kvasir_ablations_seed42/rocu_kvasir_a0_full_seed42/best.pt`
to use the dense-trained A0 weights. The checkpoint and epoch are recorded in
the experiment metadata; do not mix different checkpoints in this plot.

Use `--split test` for a final fixed-threshold sweep and report all the points.
If choosing an operating threshold, choose it on validation first. The script
neither selects a winner nor changes the segmentation threshold. It loads the
manifest next to the checkpoint and never creates a new split.

## Timing and metrics

- Every input is a real image from the selected split, evaluated at 320 × 320.
  A checkpoint configured for another resolution is rejected.
- All modes share the same weights, hardware, batch size, image order and FP32
  precision. Autocast, TF32 and TTA are disabled explicitly.
- Images are decoded, resized and normalized once on CPU. Each batch is moved
  to the device **before timing**. Each mode is warmed up on real images before
  each complete timed repeat. A seed-42 permutation is cyclically rotated:
  in each five-repeat cycle, each of the five modes occupies every position
  exactly once. The planned settings and order are saved before measurement.
- `perf_counter_ns` measures wall-clock model forward, with CUDA/MPS
  synchronization immediately before and after. This includes Python dispatch
  and any synchronization required by the sparse solver. It excludes loading,
  preprocessing, host-to-device transfer, metrics, plotting and saving.
- Dice is the unweighted mean over all images, calculated from the **first
  timed repeat's outputs**. Additional repeats measure latency, not extra samples
  for Dice. No observations or latency outliers are removed.
- X is total timed milliseconds / total timed images. Batch size defaults to 1;
  larger batch sizes must divide the split size exactly. At larger batches,
  ms/image is amortized batch latency, not single-image response latency.
- CSV also records median/p95 latency, variation between repeat means, actual
  solver-active fractions at each stage, and measured speedup relative to dense.
  Table uncertainties are ±1 sample SD (ddof=1) across complete-repeat means.
  The chart uses separate scatter points without connecting lines or error bars.
  The per-repeat CSV pairs each mode with dense from the same repeat and records
  the latency difference. These are descriptive measurements, not confidence
  intervals or variability over training seeds. All repeats are retained.
- `--cpu-threads N` fixes intra-op threads for every mode. Set it before the
  experiment; do not select a thread count separately for dense and routing.
  CPU and GPU results must be reported as separate hardware experiments.

Routing is not guaranteed to be faster on every device. Boolean indexing and
synchronization have overhead; the figure reports whatever the measured results
show. The old `compare_kvasir_routing.py` uses a synthetic latency input; use this
new script for the real-image Figure 3 requested here.

## Illustration and exact active cells

Panel (b) contains Input, Ground Truth, Prediction, Stage 1 Active Cells and
Stage 2 Active Cells. Its default operating threshold is 0.10, matching the paper.
The image is chosen by seeded random selection from sorted split IDs, before
inference and independently of scores. Repeated commands with the same seed and
split choose the same illustration; hardware latency itself will vary.

To select a specific image and an already measured operating point:

```bash
python scripts/figure3_routing.py \
  --checkpoint runs/rocu_kvasir_seed42/best.pt \
  --data-root dataset/Kvasir-SEG --split val --device cuda \
  --sample-id YOUR_IMAGE_ID --illustration-threshold 0.10 \
  --output figures/kvasir_routing_selected_image
```

The model now exposes the **actual solver mask** in `routing_active_1/2`.
The exporter checks it against each block's actual parent occupancy:

```text
Stage 1: P0 = p_quarter, A1 = [4 P0 (1 − P0) >= tau_r], grid 80 × 80
Stage 2: P1 = p_half,    A2 = [4 P1 (1 − P1) >= tau_r], grid 160 × 160
```

P1 is taken from that threshold's routed forward pass; it is not copied from
dense or reconstructed from the final mask. Binary active grids are expanded to
320 × 320 with nearest-neighbor interpolation. Dense's saved masks are all ones.
Raw parent occupancy, ambiguity, active cells and final probabilities are saved
for the illustration in **every measured mode**, as are native and enlarged
binary active PNGs. No inference hooks run inside the timed forward.

The figure and generated caption explicitly state: **active cells are cells
running the occupancy solver; convolutions and prediction heads remain dense.**
Orange overlays indicate those solver cells, not sparse convolution regions.

## Outputs and editing

```text
figures/kvasir_routing/
  figure3.png / .pdf / .svg             # Main Dice–latency + five-image panels
  figure3_occupancy.png / .pdf / .svg    # Actual occupancy/ambiguity/active maps
  caption.txt                          # Caption with this experiment's settings
  routing_summary.csv                  # One measured point per mode
  routing_summary.md / .tex            # Paper table, mean ± repeat SD, FPS, skip fraction
  latency_raw.csv                      # Every batch in every timed repeat
  latency_per_repeat.csv               # Paired comparisons with dense per repeat
  metrics_per_image.csv                # Every image in every mode
  measurement_plan.json                # Settings and mode order recorded before timing
  measurement.json                     # Checkpoint/data/source hashes, settings, device
  figure_settings.json                 # Displayed operating point; measurement metadata preserved
  samples/dense/                       # Selected image under dense inference
  samples/tau_0.05/                     # Similarly for all four thresholds
    maps.npz                           # Float occupancy/probability and exact active grids
    input.png / ground_truth.png / prediction.png
    active_1.png / active_2.png
    active_overlay_1.png / active_overlay_2.png
```

The CSV and NumPy arrays preserve full numerical precision. PDF and SVG allow
layout/font/color editing without rerunning measurements; SVG text stays text.
After changing plot styling in `rocu_net/routing_figure.py`, redraw with:

```bash
python scripts/figure3_routing.py --render-only figures/kvasir_routing --dpi 600
```

To use the saved predictions and active maps at the paper's threshold without
rerunning inference:

```bash
python scripts/figure3_routing.py --render-only figures/kvasir_routing \
  --illustration-threshold 0.10 --dpi 600
```

The selected display threshold is saved in `figure_settings.json` and reused by
later redraws. Caption and both main/supplementary maps follow that setting.

Redrawing verifies the saved experimental artifact hashes. It does not rerun
inference or change measurements. A new experiment needs a new output directory
so earlier measurements are not silently overwritten.
