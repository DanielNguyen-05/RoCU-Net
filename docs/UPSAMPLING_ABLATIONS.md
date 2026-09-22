# Upsampling controls in the shared RoCU scaffold

This suite complements the internal A0–A3 ablations. It compares feature
upsampling choices while holding the backbone, low-resolution decoder, shallow
guide, semantic fusion, boundary/score heads, soft blending and supervision
fixed. The four controls do not use the occupancy-constrained solver.

## Exact comparison

| ID | Carrier upsampling at both 2x stages | Child allocation | Equivalent component variant |
|---|---|---|---|
| U0 — RoCU (dense) | Existing Conv-BN-SiLU + PixelShuffle | Constrained solver | A3 |
| U1 — Bilinear | Bilinear, align_corners=False | Sigmoid on scores | — |
| U2 — PixelShuffle | Existing Conv-BN-SiLU + PixelShuffle | Sigmoid on scores | A0 |
| U3 — CARAFE | Content-aware kernel reassembly | Sigmoid on scores | — |
| U4 — DySample-LP | Grouped learned sampling | Sigmoid on scores | — |

All stages keep the following scaffold, with `U` as the selected feature operator:

```text
upsampled_carrier = U(carrier)
next_carrier = semantic_fusion(upsampled_carrier + guide_projection(guide))
scores, boundary, soft_gate = shared heads(next_carrier, parent occupancy)
refined = constrained_allocation(scores, parent)   # U0
        = sigmoid(scores)                        # U1–U4
next_probability = (1 - soft_gate) * nearest(parent) + soft_gate * refined
```

These rows mean **the named operators inside the same guided/blended scaffold**.
They are not standalone bilinear interpolation of the final binary mask, nor
the full architectures/results from the CARAFE or DySample papers.

U0 versus U2 isolates the occupancy constraint: all feature layers are identical.
U1/U2/U3/U4 compare the feature upsampler without the constraint. Comparing U0
against CARAFE or DySample changes both feature upsampling and the constraint;
use the U0–U2 (A0–A1) comparison to attribute an effect specifically to occupancy.
Do not infer superiority from the implementation before the experiments finish.

The native PixelShuffle control already exists mathematically as A1. The test
suite verifies identical predictions for U2 and A1 under the same initialization.
The new runner is a self-contained five-run suite with its own output directory;
it does **not** automatically import completed A0/A1 checkpoints. Existing A0/A1
results should only be substituted after confirming identical training settings,
split manifests, preprocessing and evaluation. No segmentation checkpoint is
used to initialize the new runs.

## Operator definitions and references

- **PixelShuffle:** keep the original learned 1x1 expansion C→4C, BN and SiLU,
  then apply 2x PixelShuffle. U0 and U2 therefore remain compatible with the
  original A0/A1 architecture and state keys.
- **Bilinear:** interpolate the C-channel carrier by 2 with align_corners=False.
- **CARAFE:** one reassembly group, 5x5 neighborhood, 3x3 kernel encoder,
  compression width 64, spatially normalized child kernels, zero padding. The
  PyTorch implementation contracts coarse patches with child weights before
  shuffling. It follows the [CARAFEPack definition and initialization](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/ops/carafe.html)
  and is checked against an explicit neighborhood reference, including gradients.
- **DySample-LP:** scale 2, four groups, static scope (not DySample+), offset
  multiplier 0.25, normal initialization with std 0.001, pixel-center sampling,
  align_corners=False and border padding. This follows the
  [authors' implementation](https://github.com/tiny-smart/dysample).
  The grid sampler runs in FP32 during AMP and returns the carrier dtype.
  Zero offsets recover bilinear interpolation; tests also check group-specific
  horizontal displacement. Attribution is in `THIRD_PARTY_NOTICES.md`.

No MMCV installation or custom CUDA compilation is required. Consequently,
CARAFE's measured runtime here would describe the **PyTorch reference**, not
MMCV's optimized CUDA kernel. Do not use that runtime to claim the CARAFE method
is inherently slower. The report keeps Conv/Linear GMACs and also exports an
operator-aware estimate that adds four-tap bilinear/grid sampling or the 5x5
CARAFE weighted reassembly. Coordinate generation, softmax, activations and the
occupancy solver remain excluded. The 2x GFLOP value counts a multiply and an
addition as two operations.

Each control installs its operator after initializing the common modules and
preserves the global CPU RNG state. Common parameters and loader RNG are thus
identical under seed 42. Unused PixelShuffle expansion weights are removed from
non-PixelShuffle controls; parameter counts reflect the actual instantiated model.

## Run on Kvasir

```bash
mkdir -p logs
nohup python scripts/run_kvasir_ablations.py --suite upsampling --device cuda \
  > logs/RoCUNet_Kvasir_upsampling.log 2>&1 &
```

The configs are in `configs/upsampling/`. They inherit the A0 optimization,
augmentation and loss settings: 320 × 320, seed 42, shared grouped 800/100/100
split, no hard routing, no TTA, fresh segmentation modules, ImageNet encoder
initialization, AMP training and FP32 validation/test. All variants retain all
multi-scale, boundary and gate losses at exactly the same weights.

The runner trains all five variants first, selecting checkpoints by validation
Dice, then evaluates them on test. This experiment requires training the control
models; switching operators on an already trained RoCU checkpoint is not an
equivalent experiment.

Resume, prepare without training, or collect existing results:

```bash
python scripts/run_kvasir_ablations.py --suite upsampling --device cuda --resume
python scripts/run_kvasir_ablations.py --suite upsampling --prepare-only
python scripts/run_kvasir_ablations.py --suite upsampling --collect-only
```

`--train-only`, `--evaluate-only`, `--epochs`, `--data-root` and `--output-root`
work as in the component suite. Repeat overrides on subsequent commands. The
protocol rejects changed data/configs and stale checkpoint/test metrics.

Output directory: `runs/rocu_kvasir_upsampling_seed42/`.

- `upsampling_val.csv/.md/.tex` and `upsampling_test.csv/.md/.tex` contain Dice,
  IoU, Boundary F1 and occupancy residuals, with separate validation/test tables.
- CSV also contains parameters, Conv/Linear GMACs, upsampling-operator GMACs,
  their operator-aware sum, 2x GFLOPs, best epoch and checkpoint hash.
  `upsampling_{val,test}_paper.tex` is the paste-ready matched table at 320x320.
  Occupancy residual is a constraint diagnostic; Dice, IoU and Boundary F1
  measure segmentation quality.
- `protocol.json` records all configs, data fingerprint and manifest hashes.
- Each run has its own checkpoint, logs, predictions and per-image metrics.

The local check creates the same split manifests as the component suite.
One seed does not establish variation over training seeds, and exact-image
grouping does not establish patient/video independence.

## Offline verification

```bash
python scripts/run_kvasir_ablations.py --suite upsampling --smoke-test --device cpu
```

This runs all five models for one epoch on toy images and exports tables marked
as software checks. It is not a completed Kvasir experiment and must not be used
as paper evidence.
