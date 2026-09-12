# CRS-OCU-Net

Reference PyTorch implementation of **Confidence-Routed Semantic Occupancy
Upsampling for Lightweight Polyp Segmentation**. The repository also preserves
the original OCU-Net implementation for controlled ablation.

The proposed block keeps OCU's defining local occupancy constraint while fixing
its one-channel semantic bottleneck and avoiding iterative refinement in
confident regions during deployment.

## 1. What is new

For a parent probability `p`, CRS-OCU predicts four refined children:

```text
q_ref[j] = sigmoid(score[j] + bias)
mean(q_ref[1:4]) = p
```

It also defines the conservative identity allocation `q_base[j] = p` and blends
both paths with one learned gate per parent cell:

```text
q[j] = (1 - gate) * p + gate * q_ref[j]
```

Both branches have mean `p`; therefore the blended output is conservative too.
The gate is modulated by parent uncertainty `4p(1-p)`. At inference, parent
cells below an uncertainty threshold bypass bisection and use the identity
allocation exactly. A 24-channel semantic carrier and an auxiliary boundary
head preserve information across the two high-resolution transitions.

The default path is:

| Stage | Resolution at 256 input | Main output |
|---|---:|---|
| Shallow guide | 256 x 256 | 16-channel edge/detail feature |
| MobileNetV3 r1 | 128 x 128 | encoder skip |
| MobileNetV3 r3 | 64 x 64 | encoder skip |
| MobileNetV3 r6 | 32 x 32 | encoder skip |
| MobileNetV3 r9 | 16 x 16 | bottleneck |
| Light decoder | 64 x 64 | coarse probability + semantic carrier |
| CRS-OCU-1 | 128 x 128 | half-resolution occupancy |
| CRS-OCU-2 | 256 x 256 | final occupancy + boundary map |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the equations and ablation
definitions.

## 2. Environment

Python 3.10+ and PyTorch 2.2+ are recommended.

```bash
cd OCU-Net
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The paper configuration uses ImageNet-pretrained MobileNetV3-Large and may
download the official torchvision weights on first use. For offline training, set `model.pretrained: false` in a derived config.

## 3. Kvasir-SEG

Copy the dataset without renaming files:

```text
dataset/Kvasir-SEG/
├── images/
│   ├── cju....jpg
│   └── ...
└── masks/
    ├── cju....jpg
    └── ...
```

Images and masks are paired by filename stem, so their extensions may differ.
Check the dataset before training:

```bash
python scripts/check_dataset.py --root dataset/Kvasir-SEG
```

The first run creates deterministic CSV manifests. Training and checkpoint
selection never inspect the held-out test split.

## 4. Train, validate and test

Main CRS-OCU-Net experiment:

```bash
python train.py --config configs/kvasir.yaml --device cuda
```

Useful overrides:

```bash
python train.py --config configs/kvasir.yaml \
  --data-root /absolute/path/to/Kvasir \
  --name crs_ocu_kvasir_seed87 \
  --seed 87 --device cuda
```

Resume an interrupted run:

```bash
python train.py --config configs/kvasir.yaml \
  --resume runs/crs_ocu_kvasir_seed42/last.pt --device cuda
```

Evaluate or re-profile a selected checkpoint:

```bash
python evaluate.py \
  --checkpoint runs/crs_ocu_kvasir_seed42/best.pt \
  --split test --save-predictions --profile --device cuda
```

Predict one image or every image in a directory:

```bash
python predict.py \
  --checkpoint runs/crs_ocu_kvasir_seed42/best.pt \
  --input /path/to/images --output predictions --device cuda
```

This saves probability maps, thresholded masks and red overlays at the original
image resolution.

Architecture-only profiling:

```bash
python profile_model.py --config configs/kvasir.yaml --device cuda \
  --output runs/crs_architecture_profile.json
```

## 5. Paper experiments

Three seeds:

```bash
python scripts/run_multiseed.py \
  --config configs/kvasir.yaml --seeds 13 42 87 --device cuda
```

Ablations at seed 42:

```bash
python scripts/run_ablations.py --device cuda --seed 42
```

The ablation suite evaluates:

- full CRS-OCU-Net;
- no confidence routing (all cells use refined OCU);
- no cross-scale semantic carrier;
- no boundary/routing/structure supervision.

Create a CSV and Markdown comparison from completed runs:

```bash
python make_comparison_table.py \
  OCU=runs/ocu_net_kvasir_seed42 \
  CRS-OCU=runs/crs_ocu_kvasir_seed42 \
  --output runs/ocu_vs_crs
```

Keep the same manifests, image size, seeds and metric code when adding external
baselines such as LV-UNet, DeepNeXt, UNeXt and EGE-UNet.

## 6. Outputs and lightweight metrics

Each run is isolated under `runs/<experiment_name>/`:

```text
best.pt / last.pt
config_resolved.yaml
splits/{train,val,test}.csv
history.csv
training_curves.png
val_metrics.json / test_metrics.json
val_per_image.csv / test_per_image.csv
test_predictions/{probability,binary}/
lightweight_metrics.json
summary.json
train.log
```

Segmentation metrics:

- Dice, IoU, precision, recall/sensitivity;
- specificity, pixel accuracy, F2 and MAE;
- boundary F1;
- local occupancy-conservation error.

Lightweight/deployment metrics:

- total/trainable parameters and model/checkpoint size;
- Conv2d/Linear MACs and `FLOPs = 2 x MACs`;
- mean, median and p95 batch-1 latency plus FPS;
- peak CUDA allocated/reserved memory;
- CRS stage-wise active-cell and estimated solver-skip ratios;
- exact hardware and software environment.

The iterative solver is element-wise and intentionally excluded from MACs, so
latency and routing statistics must be reported beside MACs. Speed claims are
valid only under identical hardware, input size, batch size, warm-up and repeat
settings.

## 7. Smoke test without real data

```bash
python scripts/create_toy_kvasir.py --output dataset/ToyKvasir --count 24 --size 128
# Create a derived config with data.root: dataset/ToyKvasir,
# model.pretrained: false, and training.epochs: 1 before training.
pytest
```

Synthetic results are only for software verification and must never appear in
the paper.

## 8. Original OCU-Net baseline

To run the earlier architecture, set `model.architecture: ocu_net` in a derived config:

```bash
python train.py --config configs/kvasir.yaml --device cuda
```

`model.architecture` selects either `ocu_net` or `crs_ocu_net`, allowing both
methods to use the same data, training, evaluation and profiling pipeline.

## 9. CVC-ClinicDB

The project uses the same CRS-OCU-Net, preprocessing, losses and metrics for
both datasets. `configs/cvc_clinicdb.yaml` inherits the current Kvasir settings
(320 x 320, batch size 8, up to 250 epochs) and uses a separate run directory.

Populate these directories with paired files (the directories in the current
workspace are empty):

```text
dataset/CVC-ClinicDB/
└── PNG/
    ├── Original/       # 1.png, 2.png, ...
    └── Ground Truth/   # 1.png, 2.png, ...
```

Use one representation only. For TIFF data, change `data.image_dir` to
`TIF/Original` and `data.mask_dir` to `TIF/Ground Truth` in the config.
For an extracted dataset with `Original` and `Ground Truth` directly under its
root, set those directory values without the PNG/TIF prefix.

```bash
python scripts/check_dataset.py --config configs/cvc_clinicdb.yaml
python train.py --config configs/cvc_clinicdb.yaml --device cuda
python evaluate.py --checkpoint runs/crs_ocu_cvc_clinicdb_seed42/best.pt --split test --save-predictions --device cuda
python predict.py --checkpoint runs/crs_ocu_cvc_clinicdb_seed42/best.pt --input dataset/CVC-ClinicDB/PNG/Original --output predictions/cvc_clinicdb --device cuda
```

Use `--device auto` on a machine without CUDA. Training automatically evaluates
`best.pt` on the held-out test split and saves results under
`runs/crs_ocu_cvc_clinicdb_seed42/`. The separate evaluate command repeats that
held-out evaluation. `--data-root` relocates the same dataset; it does not turn
a Kvasir checkpoint into a cross-dataset CVC evaluation.

Splits are seeded random image splits, 80%/10%/10%. With 612 pairs this yields
489 train, 61 validation and 62 test images. Existing CSV manifests are reused.
This is a project-defined split, not an official benchmark split or a
sequence/patient-disjoint protocol. Use the same protocol when comparing runs.

Resume with the CVC config:

```bash
python train.py --config configs/cvc_clinicdb.yaml --resume runs/crs_ocu_cvc_clinicdb_seed42/last.pt --device cuda
python scripts/run_multiseed.py --config configs/cvc_clinicdb.yaml --prefix crs_ocu_cvc_clinicdb --seeds 13 42 87 --device cuda
```

Review note: probability BCE and structure BCE previously passed probabilities
into `binary_cross_entropy_with_logits`. They now compute BCE on probabilities
in float32, including under AMP. Existing checkpoints still support prediction,
but new training and reported losses use the corrected objective. For a
controlled Kvasir/CVC comparison, retrain both with this version; avoid resuming
an old run if you need an unchanged training objective. Evaluation metrics are
computed at the configured resized resolution; `predict.py` exports masks at
the original resolution.

## 10. CVC-ColonDB and ETIS

Both new local datasets use `images/` and `masks/` with matching filename stems.
The checked directories contain 380 ColonDB pairs and 196 ETIS pairs.

```bash
python scripts/check_dataset.py --config configs/cvc_colondb.yaml
python scripts/check_dataset.py --config configs/etis.yaml
python train.py --config configs/cvc_colondb.yaml --device cuda
python train.py --config configs/etis.yaml --device cuda
```

The configs inherit the model, image size, augmentation and training settings
from `kvasir.yaml`. They create separate runs:

| Config | Run | Train / validation / test |
|---|---|---|
| `cvc_colondb.yaml` | `runs/rocu_cvc_colondb_seed42` | 304 / 38 / 38 |
| `etis.yaml` | `runs/rocu_etis_seed42` | 156 / 19 / 21 |

These are **within-dataset random image splits**. Training on ColonDB/ETIS and
then testing their held-out subsets answers a different question from testing
a Kvasir-trained model on all ColonDB/ETIS images. Do not describe a model
trained on those datasets as unseen-dataset generalization.

```bash
python evaluate.py --checkpoint runs/rocu_cvc_colondb_seed42/best.pt --save-predictions --device cuda
python evaluate.py --checkpoint runs/rocu_etis_seed42/best.pt --save-predictions --device cuda
```

## 11. Paper figures for RoCU-Net

`visualize.py` exports 300-dpi PNG and PDF figures, English captions, exact
sample IDs, all per-image metrics and checkpoint/protocol metadata. It uses
checkpoint weights without downloading pretrained weights. No training is
performed by this command.

```bash
python visualize.py \
  --checkpoint runs/crs_ocu_kvasir_seed42/best.pt \
  --data-root dataset/Kvasir-SEG \
  --output figures/kvasir --device cuda
  # --selection best # mặc định là phân bố đều (1 mẫu khó nhất 1 mẫu ở khoảng 25%, 1 mẫu trung vị, 1 mẫu ở khoảng 75%, 1 mẫu tốt nhất)
  # - num-samples 5
```

Copy the original `splits/` directory alongside `best.pt` when moving a run.
The visualization command requires those manifests and never regenerates an
in-dataset split. `--data-root` can relocate the data. All images are evaluated
at the checkpoint's configured input resolution with its fixed threshold.

Outputs:

- `qualitative_01.{png,pdf}`: input, ground truth, prediction, contour detail,
  pixel errors and foreground probability. The input rectangle identifies the
  contour crop. TP is green, FP orange, FN purple; contour GT is blue and
  prediction orange. These highlight accurate boundaries and reveal failures.
- `mechanism_01.{png,pdf}`: auxiliary boundary, stage-2 soft routing gate,
  actual solver-active parent cells, and absolute occupancy conservation error.
  These illustrate the architecture's behavior; gate values are not calibrated
  uncertainty and skip fractions do not establish measured speedup.
- `metric_distribution.{png,pdf}`: Dice/IoU/boundary-F1 distributions, Dice versus
  target area and a Dice histogram over **all evaluated images**.
- `per_image_metrics.csv`, `selected_samples.csv`, `figure_metadata.json` and
  `captions.txt`: values and provenance needed to reproduce and describe figures.
- `training_curves.{png,pdf}` when the run contains `history.csv`.

The default six rows cover evenly spaced **Dice ranks from worst to best**.
This avoids silently presenting only successful cases. Use `--selection best`
for an explicitly labeled illustrative success panel, `--selection worst` for
failure analysis, or `--selection random --seed 42`. Use `--sample-ids ID1 ID2`
to lock the same cases across future runs. Keep the distribution plot alongside
selected examples. Requests longer than six rows are paginated.

For a separately trained dataset, substitute its checkpoint. For external
all-image evaluation of a Kvasir checkpoint:

```bash
python visualize.py --checkpoint runs/crs_ocu_kvasir_seed42/best.pt \
  --external-config configs/cvc_colondb.yaml --output figures/kvasir_to_colondb --device cuda
python visualize.py --checkpoint runs/crs_ocu_kvasir_seed42/best.pt \
  --external-config configs/etis.yaml --output figures/kvasir_to_etis --device cuda
```

Here `--external-config` supplies only the target data paths; model settings,
input size and threshold remain those of the source checkpoint. No target
splits are created. Ensure the target images were not used in training before
calling this unseen-dataset evaluation.

Export training curves from available logs without a checkpoint:

```bash
python visualize.py --history logs/RoCUNet_Kvasir.log --output figures/kvasir_history
```

This figure has already been generated locally. Logged values are rounded;
prefer `history.csv` for full precision. Curves are unsmoothed. No actual
checkpoint or saved prediction masks were present when this tool was added,
so qualitative model results still require your trained checkpoint.

For the paper, use qualitative segmentation and the full-split distribution
to show quality; pair the mechanism maps with controlled ablations and measured
latency/FPS to substantiate conservation and efficiency. A single model's maps
cannot establish superiority over baselines.

Metric correction in this update: boundary F1 now returns 0 when both boundary
precision and recall are 0 (disjoint contours). The earlier code returned 1
in that case. Re-evaluate old checkpoints before reporting boundary F1; Dice
and IoU calculations are unchanged.

## 12. ColonDB audit and revised training

See [the evidence-backed ColonDB report](reports/colondb_audit/REPORT.md) for
checkpoint checks, failure cases, the LV-UNet metric/split mismatch, and the
controlled validation-only routing comparison. Original run files are preserved.

The revised augmentation preserves the full rectangular image when rotating.
The candidate ColonDB configs add foreground-retaining random crops and a
moderate Tversky term to reduce missed foreground. They reuse the original
`runs/rocu_cvc_colondb_seed42/splits` files and create separate run directories.

```bash
# Fine-tune your existing ColonDB checkpoint with a fresh optimizer.
python train.py --config configs/cvc_colondb_finetune.yaml --device cuda

# Alternatively, train the revised recipe from ImageNet initialization.
python train.py --config configs/cvc_colondb_v2.yaml --device cuda
```

Both configs disable automatic test evaluation while comparing recipes on
validation. After choosing the recipe:

```bash
python evaluate.py --checkpoint runs/rocu_cvc_colondb_finetune_seed42/best.pt --split test --save-predictions --device cuda
```

`--init-checkpoint` loads weights only into a new run, whereas `--resume` restores
optimizer/scheduler/epoch. Fine-tuning evaluates the initial model first and
keeps those weights as best if no later epoch improves validation Dice. Starting
a new run in a directory already containing checkpoints now raises an error;
use a new `--name` or resume explicitly.

Optional loss settings: `loss.tversky_weight` (default 0, preserving existing
loss behavior) and `loss.tversky_beta` (default 0.7). `training.freeze_encoder_bn`
freezes running statistics only and is enabled in the fine-tuning config.
Rotation corrections affect all future training; prediction of old checkpoints
is unchanged. A short real-data CPU pilot is a validation check, not a complete
GPU training experiment or a new test-set claim.

The downloaded ColonDB data also contains exact duplicate images across saved
splits (346 distinct RGB images among 380 files). For a new comparison that
keeps identical images together, use `configs/cvc_colondb_grouped.yaml` and
train all baselines on the same new manifests. This changes the held-out set:
do not initialize it from the old ColonDB checkpoint. It does not provide
video/patient grouping or resolve differing annotations of the same image.

The initial three-epoch CPU fine-tuning pilot did **not** improve validation
Dice (0.879202 initially versus 0.876031 at epoch 3). These configs remain
experimental; the rotation correction is a verified software fix, while
accuracy benefits from the new recipe still require controlled training.
