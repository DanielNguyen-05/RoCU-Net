# Figure 3: completed real-image CPU experiment

The five-mode experiment completed on the original 100-image Kvasir validation
split using the existing full RoCU-Net checkpoint, epoch 197. No retraining or
checkpoint selection was performed. The SHA-256 is
`53598bb81bd07f668c5533371724b05033a5e6d2086db92e58eafeb0f97d4d61`.

Hardware available in this session was an ARM CPU on macOS, with two PyTorch
intra-op threads. CUDA and MPS were unavailable. **These results are CPU
measurements and must not be labeled as RTX 4060 Ti results.**

## Measured results

| Mode | Mean Dice | Latency, ms/image | FPS | Solver cells skipped | Latency reduction |
|---|---:|---:|---:|---:|---:|
| Dense | 0.888750 | 104.434 ± 0.896 | 9.58 | 0.00% | — |
| τr = 0.05 | 0.888750 | 101.351 ± 1.047 | 9.87 | 97.99% | 2.95% |
| τr = 0.10 | 0.888750 | 101.359 ± 0.887 | 9.87 | 98.38% | 2.95% |
| τr = 0.20 | 0.888749 | 101.439 ± 0.681 | 9.86 | 98.80% | 2.87% |
| τr = 0.30 | 0.888750 | 102.413 ± 3.128 | 9.76 | 99.18% | 1.94% |

Latency is mean ± sample SD of five complete-repeat means, not an interval over
training seeds or a statistical confidence interval. FPS is 1000 / mean latency.
The skip percentage is aggregated over both stages and all 100 images; the
active percentages in panel (b) refer only to that illustration.

At the paper's operating threshold, τr = 0.10, the mean Dice difference from
dense was +1.70e−7. Routing was faster than dense in all
five paired repeats at 0.05, 0.10 and 0.20. At 0.30 it was faster in four of five
repeats and had greater timing variation. No repeat or operating point was
discarded. Do not infer a statistically significant difference among routing
thresholds from their close mean latencies.

The evidence supports **a large reduction in solver-processed cells, with nearly
unchanged segmentation and a modest CPU latency improvement**. It does not
support a claim of a 98–99% reduction in total computation or latency: convolution
and prediction heads still run densely. No new GPU speedup has been measured.

## Experimental protocol

- Same checkpoint, complete original validation manifest, and image order.
- Fixed 320 × 320 input, batch size 1, segmentation threshold 0.5, FP32, no TTA
  or TF32. The dense reference retains soft gating.
- All four thresholds were specified before measurement; none was chosen from
  this run's test performance. This is a validation experiment.
- Five complete timed passes per mode, with 30 real-image warmup forwards before
  each pass: 2,500 timed forwards and 750 warmup forwards overall.
- Seeded cyclic mode order gives every mode each order position exactly once.
  `measurement_plan.json` records the settings and planned order before timing.
- Timing includes synchronized model forward, including solver and Python
  dispatch. It excludes loading, normalization, transfers, metrics and export.
- Dice comes from first-repeat timed outputs for all 100 images. The illustrative
  image was selected by seed 42 independently of scores. Its actual parent
  occupancy and solver-active decisions were checked at both stages.

## Files

All measured artifacts are in `figures/kvasir_routing_paper_cpu/`:

- `figure3.pdf`, `figure3.svg`, `figure3.png`: combined figure, 600-DPI raster export.
- `figure3_occupancy.*`: actual occupancy, ambiguity and solver maps.
- `routing_summary.csv`, `.md`, `.tex`: every operating point, ready for inspection
  and a LaTeX table (`booktabs` is required).
- `latency_raw.csv`, `latency_per_repeat.csv`, `metrics_per_image.csv`: all data.
- `measurement_plan.json`, `measurement.json`: configuration, hardware, file
  hashes and the order of measurement.
- `samples/`: actual inputs, ground truth, predictions and native/expanded maps
  for the selected image under every mode.
- `caption.txt`: detailed caption and experimental settings.
- `figure_settings.json`: the displayed operating point (0.10) and scatter style;
  original measurement settings and raw numerical data are retained.

Audit: 2,500 timing records, 500 image/mode metric rows, matching planned and
actual orders, native 80 × 80 and 160 × 160 parent grids, nearest-neighbor
overlays, and all recorded artifact hashes were verified after the run.

## Suggested paper wording

Results paragraph:

> On the 100-image Kvasir validation split, confidence-based solver routing
> skipped 98.38% of parent cells at τr = 0.10 while changing mean Dice by less
> than 10−6 relative to dense inference. With identical weights and FP32 inputs
> on an ARM CPU using two threads, mean forward latency decreased from
> 104.43 ± 0.90 to 101.36 ± 0.89 ms per image over five complete repeats,
> corresponding to a 2.95% reduction. The improvement is modest because
> convolutional layers and prediction heads remain dense.

Concise caption:

> Figure 3. Real-image routing evaluation using one RoCU-Net checkpoint on all
> 100 Kvasir validation images. (a) Mean Dice versus forward latency for dense
> inference and four routing thresholds. Inputs are 320 × 320, batch size is 1,
> precision is FP32, and TTA is disabled. Points are shown without connecting
> lines; table uncertainties show one sample SD of five complete-repeat mean
> latencies on an ARM CPU with two threads. Timing
> excludes loading, preprocessing, transfers and visualization. (b) A seeded
> example at τr = 0.10 with input, ground truth, prediction and actual solver-active
> parent cells at stages 1 and 2. Native 80 × 80 and 160 × 160 active grids are
> enlarged with nearest-neighbor interpolation. Colored cells denote occupancy
> solver execution; convolutions and prediction heads remain dense.

If the paper retains the original synthetic-input RTX 4060 Ti latency/FPS table,
replace the blanket limitation with a qualified statement:

> The original GPU throughput benchmark uses a synthetic input, while Fig. 3
> additionally evaluates routing on real Kvasir images using an ARM CPU. These
> forward-only measurements exclude preprocessing and data-transfer overhead;
> the real-image routing speedup has not yet been validated on the GPU used in
> the throughput table.

Do not delete the GPU limitation merely because a CPU figure was added. The
original saved GPU profile reports zero active solver cells on its synthetic
input; real-image routing needs its own measurement on that GPU.

## Run the matching experiment on the server

With the RTX 4060 Ti otherwise idle, run:

```bash
python scripts/figure3_routing.py \
  --checkpoint runs/rocu_kvasir_seed42/best.pt \
  --data-root dataset/Kvasir-SEG \
  --split val --device cuda --cpu-threads 6 \
  --illustration-threshold 0.10 \
  --warmup 30 --repeats 5 --batch-size 1 --seed 42 --dpi 600 \
  --output figures/kvasir_routing_paper_gpu
```

The script records the actual GPU identity; the command does not assume it is
the same device. Report the generated GPU measurements as a separate experiment
or use that figure in the paper. This command requires no retraining.

The completed CPU run used the same command with `--device cpu --cpu-threads 2`
and `--output figures/kvasir_routing_paper_cpu` (OMP and MKL thread limits were
also set to 2). Repeat the CPU command with a fresh output path to preserve the
completed measurements.
