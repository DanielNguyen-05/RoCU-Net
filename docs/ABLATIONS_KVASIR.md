# Kvasir A0–A3 ablations

Run the predefined four variants sequentially with one seed, 42:

```bash
mkdir -p logs
nohup python scripts/run_kvasir_ablations.py --device cuda \
  > logs/RoCUNet_Kvasir_ablations.log 2>&1 &
```

The suite prepares one shared split, trains all four variants, then evaluates
their validation-selected checkpoints on test. It does not select a winning
variant using test. Training jobs have their own logs under
`runs/rocu_kvasir_ablations_seed42/logs/`; the top-level log records phase changes
and the final tables. Follow the A0 training log with:

```bash
tail -f runs/rocu_kvasir_ablations_seed42/logs/rocu_kvasir_a0_full_seed42.log
```

## The four configurations

| ID | Config in `configs/ablations/` | Difference from A0 |
|---|---|---|
| A0 | `kvasir_a0_full.yaml` | Full RoCU-Net, dense inference |
| A1 | `kvasir_a1_no_constraint.yaml` | `sigmoid(grouped_scores)` replaces the constraint solver in both stages |
| A2 | `kvasir_a2_no_carrier.yaml` | Zero the expanded propagated carrier in both stages; retain the guide and fusion |
| A3 | `kvasir_a3_no_blending.yaml` | Final gate is exactly one in both stages; gate-loss weight is zero |

The carrier, gate and other layers are still instantiated, so initial state keys,
parameter counts and initialization draws match. A2 and A3 measure information
and mechanism removal; they do not demonstrate parameter savings. A1 retains
the parent input, learned gate, uncertainty modulation, boundary head and losses.
A3 removes both learned and uncertainty-modulated blending. All other losses,
including boundary supervision, remain unchanged.

All four initialize the encoder from the same ImageNet MobileNetV3 weights and
initialize the segmentation modules afresh. No Kvasir/ClinicDB/ColonDB trained
segmentation checkpoint is used. The first run may download torchvision weights.
The runner verifies that config differences match only the intended ablations.

## Shared evaluation protocol

- 320 x 320 images, threshold 0.5, identical optimizer/augmentation schedules.
- Seed 42, deterministic mode requested, separate fresh training per variant.
- No TTA and no hard routing, including during validation checkpoint selection.
  A1 has no solver; “dense” means all cells are refined without routing bypass.
- AMP training is allowed, but validation and test use FP32 for all variants.
- Select the best epoch by validation Dice. All training finishes before test.
- One set of manifests under `runs/rocu_kvasir_ablations_seed42/splits/` is copied
  into each run. Exact duplicate image pixels stay in the same split.

The local dataset check found 1,000 distinct decoded images and produced an
800/100/100 split. Pixel grouping does not detect near-duplicates or establish
patient/video independence. It also does not erase prior use of this dataset
for development. State these limits when describing the experiment.

`protocol.json` records the data-content fingerprint, split hashes and complete
configs. Reruns refuse changed data/configs and refuse to overwrite checkpoints.
Saved test metrics are tied to checkpoint and split hashes. A shared seed helps
control variability but does not measure variability over seeds; interrupted
training resumed with the existing trainer need not reproduce augmentation RNG
state exactly.

## Results for the paper

The suite writes separate files for validation and test:

```text
runs/rocu_kvasir_ablations_seed42/
  protocol.json
  splits/
  logs/
  rocu_kvasir_a0_full_seed42/       # best.pt, last.pt, metrics, predictions
  rocu_kvasir_a1_no_constraint_seed42/
  rocu_kvasir_a2_no_carrier_seed42/
  rocu_kvasir_a3_no_blending_seed42/
  ablation_val.csv / .md / .tex
  ablation_test.csv / .md / .tex
```

Columns: Variant, Dice, IoU, Boundary F1 and Occupancy MAE. Scores use an
unweighted mean over images. Per-image standard deviation is not variation over
training seeds. The CSV also includes both stage residuals, best epoch and
checkpoint hashes. Occupancy MAE is computed before thresholding:

```text
E_full = mean(abs(avgpool2(P_full) - P_half))
E_half = mean(abs(avgpool2(P_half) - P_quarter))
Occupancy MAE = (E_full + E_half) / 2
```

Small residuals check the constraint implementation; they alone do not establish
better segmentation. Boundary F1 uses the existing `batch_metrics` implementation
at 320 x 320 with the same boundary extraction and tolerance for every variant.
Specifically, boundaries are the binary mask minus its 3 x 3 erosion; matching
uses a 5 x 5 dilation (Chebyshev tolerance of 2 pixels).

## Prepare, resume, or collect

Prepare/audit the real dataset without starting training:

```bash
python scripts/run_kvasir_ablations.py --prepare-only
```

Resume the suite after interruption (completed runs are checked and skipped):

```bash
python scripts/run_kvasir_ablations.py --device cuda --resume
```

To defer the test phase, use `--train-only`, then `--evaluate-only` when ready.
To regenerate tables without inference, use `--collect-only`. Repeat the same
`--data-root`, `--output-root` and `--epochs` overrides on subsequent commands if
you used them initially. Changing these settings requires a new output root.

An individual config can also be trained after preparing the default manifests:

```bash
python train.py --config configs/ablations/kvasir_a1_no_constraint.yaml --device cuda
```

Individual training exports validation but defers test. To evaluate all completed
individual runs and build the tables, use `--evaluate-only` on the suite runner.

## Hard routing on the same A0 checkpoint

```bash
python scripts/compare_kvasir_routing.py \
  --checkpoint runs/rocu_kvasir_ablations_seed42/rocu_kvasir_a0_full_seed42/best.pt \
  --split val --device cuda
```

This compares dense versus hard inference on identical weights, with no TTA,
FP32 metrics and the routing threshold already stored in the checkpoint. It
reports quality, occupancy MAE, solver-skip fraction over the evaluated images,
and latency/FPS with batch size 1. The output is in the A0 run's `routing_val/`.
Use `--split test` for a fixed-policy final test comparison. There is no threshold
search on test and no retraining. Latency is measured using a synthetic profiling
input; solver-skip fraction in the CSV is measured on the actual dataset split.

Dense inference now correctly reports every cell as processed by the solver.
A1 reports no solver cells. Conv/Linear MACs exclude the bisection operations;
use measured latency and solver-call statistics to assess hard routing.

## Offline software smoke test

```bash
python scripts/run_kvasir_ablations.py --smoke-test --device cpu
```

This creates synthetic data, uses a small offline encoder, and runs one epoch
per variant under `runs/rocu_kvasir_ablations_smoke/`. Exported tables are marked
as software checks, not paper results. The full four-run pipeline and separate
routing comparison were verified locally with this setup. Real Kvasir training
has not been run by this implementation check.
