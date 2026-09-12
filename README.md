# RoCU-Net

Reference PyTorch implementation of **RoCU-Net** for lightweight polyp
segmentation, using confidence-routed semantic occupancy upsampling.

The proposed block keeps the local occupancy constraint while fixing
its one-channel semantic bottleneck and avoiding iterative refinement in
confident regions during deployment.

## 1. What is new

For a parent probability `p`, RoCU predicts four refined children:

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
| RoCU-1 | 128 x 128 | half-resolution occupancy |
| RoCU-2 | 256 x 256 | final occupancy + boundary map |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the equations.

## 2. Environment

Python 3.10+ and PyTorch 2.2+ are recommended.

```bash
cd RoCU-Net
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Uninstall the current incompatible versions
pip uninstall torch torchvision torchaudio -y
# Install PyTorch compiled for CUDA 12.1 (compatible with your 12.5 driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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

Main RoCU-Net experiment:

```bash
mkdir -p logs
nohup python train.py --config configs/kvasir.yaml --device cuda > logs/RoCUNet_Kvasir.log 2>&1 &
```

Useful overrides:

```bash
python train.py --config configs/kvasir.yaml \
  --data-root /absolute/path/to/Kvasir \
  --name rocu_kvasir_seed87 \
  --seed 87 --device cuda
```

Resume an interrupted run:

```bash
python train.py --config configs/kvasir.yaml \
  --resume runs/rocu_kvasir_seed42/last.pt --device cuda
```

Evaluate or re-profile a selected checkpoint:

```bash
python evaluate.py \
  --checkpoint runs/rocu_kvasir_seed42/best.pt \
  --split test --save-predictions --profile --device cuda
```

Predict one image or every image in a directory:

```bash
python predict.py \
  --checkpoint runs/rocu_kvasir_seed42/best.pt \
  --input /path/to/images --output predictions --device cuda
```

This saves probability maps, thresholded masks and red overlays at the original
image resolution.

Architecture-only profiling:

```bash
python profile_model.py --config configs/kvasir.yaml --device cuda \
  --output runs/rocu_architecture_profile.json
```

## 5. Dataset configurations

The supported configs are `kvasir.yaml`, `cvc_clinicdb.yaml`,
`cvc_colondb.yaml` and `etis.yaml`. Each uses seed 42. ColonDB uses the selected
rotation-fix recipe; experimental configs and multi-seed runners were removed.

Create a CSV and Markdown comparison from completed runs:

```bash
python make_comparison_table.py \
  RoCU-Net-Kvasir=runs/rocu_kvasir_seed42 \
  RoCU-Net-ClinicDB=runs/rocu_cvc_clinicdb_seed42 \
  --output runs/rocu_results
```

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
- RoCU stage-wise active-cell and estimated solver-skip ratios;
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

## 8. Model naming and existing checkpoints

All dataset configs use `model.architecture: rocu_net`. The Python package is
`rocu_net`, the model class is `RoCUNet`, and its two `RoCUBlock` stages are
`rocu1` and `rocu2`. Solver settings are `model.solver_iterations` and
`model.epsilon`. New run directories use the `rocu_` prefix.

Historical checkpoint names and stage keys are translated automatically when
loading for training, evaluation, prediction, visualization and profiling.
Weights, optimizer state and numerical settings are preserved; retraining is
not required. The earlier occupancy-only implementation remains available as
`OccupancyUNet` (`occupancy_unet`) for old checkpoints.

After copying the updated code to the server, migrate existing output names:

```bash
python scripts/migrate_rocu_names.py --apply
```

This renames legacy output folders/files and normalizes saved YAML configs.
It refuses to overwrite an existing destination. Without `--apply`, it prints
the planned changes. Original checkpoint binaries, logs and measured results
retain their recorded content; new outputs use the current names.

## 9. CVC-ClinicDB

The project uses the same RoCU-Net, preprocessing, losses and metrics for
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
python evaluate.py --checkpoint runs/rocu_cvc_clinicdb_seed42/best.pt --split test --save-predictions --device cuda
python predict.py --checkpoint runs/rocu_cvc_clinicdb_seed42/best.pt --input dataset/CVC-ClinicDB/PNG/Original --output predictions/cvc_clinicdb --device cuda
```

Use `--device auto` on a machine without CUDA. Training automatically evaluates
`best.pt` on the held-out test split and saves results under
`runs/rocu_cvc_clinicdb_seed42/`. The separate evaluate command repeats that
held-out evaluation. `--data-root` relocates the same dataset; it does not turn
a Kvasir checkpoint into a cross-dataset CVC evaluation.

Splits are seeded random image splits, 80%/10%/10%. With 612 pairs this yields
489 train, 61 validation and 62 test images. Existing CSV manifests are reused.
This is a project-defined split, not an official benchmark split or a
sequence/patient-disjoint protocol. Use the same protocol when comparing runs.

Resume with the CVC config:

```bash
python train.py --config configs/cvc_clinicdb.yaml --resume runs/rocu_cvc_clinicdb_seed42/last.pt --device cuda
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
| `cvc_colondb.yaml` | `runs/rocu_cvc_colondb_rotation_fix_seed42` | 304 / 38 / 38 |
| `etis.yaml` | `runs/rocu_etis_seed42` | 156 / 19 / 21 |

These are **within-dataset random image splits**. Training on ColonDB/ETIS and
then testing their held-out subsets answers a different question from testing
a Kvasir-trained model on all ColonDB/ETIS images. Do not describe a model
trained on those datasets as unseen-dataset generalization.

```bash
python evaluate.py --checkpoint runs/rocu_cvc_colondb_rotation_fix_seed42/best.pt --save-predictions --device cuda
python evaluate.py --checkpoint runs/rocu_etis_seed42/best.pt --save-predictions --device cuda
```

## 11. Paper figures for RoCU-Net

`visualize.py` exports 300-dpi PNG and PDF figures, English captions, exact
sample IDs, all per-image metrics and checkpoint/protocol metadata. It uses
checkpoint weights without downloading pretrained weights. No training is
performed by this command.

```bash
python visualize.py \
  --checkpoint runs/rocu_kvasir_seed42/best.pt \
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

The default five rows cover evenly spaced **Dice ranks from worst to best**.
This avoids silently presenting only successful cases. Use `--selection best`
for an explicitly labeled illustrative success panel, `--selection worst` for
failure analysis, or `--selection random --seed 42`. Use `--sample-ids ID1 ID2`
to lock the same cases across future runs. Keep the distribution plot alongside
selected examples. Requests longer than six rows are paginated.

For a separately trained dataset, substitute its checkpoint. For external
all-image evaluation of a Kvasir checkpoint:

```bash
python visualize.py --checkpoint runs/rocu_kvasir_seed42/best.pt \
  --external-config configs/cvc_colondb.yaml --output figures/kvasir_to_colondb --device cuda
python visualize.py --checkpoint runs/rocu_kvasir_seed42/best.pt \
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

Use qualitative segmentation and the full-split distribution to show quality,
and report measured latency/FPS alongside the mechanism maps.

Metric correction in this update: boundary F1 now returns 0 when both boundary
precision and recall are 0 (disjoint contours). The earlier code returned 1
in that case. Re-evaluate old checkpoints before reporting boundary F1; Dice
and IoU calculations are unchanged.

## 12. Selected ColonDB result

`configs/cvc_colondb.yaml` now resolves to the exact configuration of
`rocu_cvc_colondb_rotation_fix_seed42`. It uses corrected rotations, the original
loss and augmentation settings, and the original 304/38/38 split. The reported
validation results are Dice **0.8837**, IoU **0.8102**, recall **0.8953**,
precision **0.9001**, and boundary F1 **0.7441**. These are validation scores.

Use the already-trained winning checkpoint to evaluate test and export figures:

```bash
python evaluate.py --checkpoint runs/rocu_cvc_colondb_rotation_fix_seed42/best.pt --split test --save-predictions --device cuda
python visualize.py --checkpoint runs/rocu_cvc_colondb_rotation_fix_seed42/best.pt --output figures/colondb --num-samples 5 --device cuda
```

The winning run is currently on the server; use those commands there, or copy
its folder (including `best.pt` and `splits/`) locally. The original config's
`test_after_training: false` is retained; test is run explicitly as above.

To reproduce training, keep `runs/rocu_cvc_colondb_seed42/splits/` available and
use a fresh output name if the winning run already exists:

```bash
python train.py --config configs/cvc_colondb.yaml --name rocu_cvc_colondb_rotation_fix_repeat --device cuda
```

Earlier audit reports and run outputs are historical records. Their experimental
config commands are superseded by this section.
