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

Main CRS-OCU-Net experiment:

```bash
nohup python train.py --config configs/*.yaml --device cuda > logs/RoCUNet_<dataset>.log 2>&1 &
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
