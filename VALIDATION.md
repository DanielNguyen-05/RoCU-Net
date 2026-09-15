# Validation

The active dataset configs are `configs/kvasir.yaml`,
`configs/cvc_clinicdb.yaml`, `configs/cvc_colondb.yaml`, and `configs/etis.yaml`.
ColonDB uses Kvasir initialization and flip TTA at seed 42.

Check local data and run the test suite:

```bash
python scripts/check_dataset.py --config configs/cvc_colondb.yaml
python -m pytest
```

To evaluate the completed winning run on the server:

```bash
python evaluate.py --checkpoint runs/rocu_cvc_colondb_transfer_seed42/best.pt --split test --save-predictions --device cuda
```

The verified saved test Dice is 0.8969 and IoU is 0.8301 at epoch 91, with
Kvasir pretraining and four-view flip TTA. Synthetic
smoke results are software checks and must not be used as research results.

The RoCU-Net naming migration was checked locally with 29 test functions,
including legacy checkpoint loading, config inheritance and migration collision
handling. All passed via direct invocation (pytest was unavailable locally).
All outputs of the three available trained models were bitwise identical before
and after renaming on a fixed CPU input of shape `[1, 3, 64, 64]`; SHA-256 checks
confirmed all six checkpoint files were unchanged.

A temporary synthetic dataset also passed training, optimizer/epoch resume,
evaluation, prediction, checkpoint-based profiling and visualization with five
examples. New checkpoint keys, resolved configs and summaries used the canonical
RoCU names. This verifies software compatibility, not new dataset scores.

The Kvasir A0–A3 implementation adds checks for direct-sigmoid gradients,
carrier-information removal, final gate=1, omitted gate supervision, identical
initial weights, dense solver accounting, changed-dataset rejection and stale
test-metric rejection. The complete four-variant train/test/table pipeline and
dense/hard routing comparison passed on a synthetic offline dataset. Kvasir was
audited locally: 1,000 unique decoded images, with shared splits of 800/100/100.
These implementation checks do not constitute completed real-data ablations.

Figure 3 routing checks verify that exported masks match the actual sparse-solver
inputs at both stages, nearest-neighbor overlays preserve parent cells, dense
keeps soft blending, zero-threshold routing agrees with dense, confident cells
bypass the solver, and repeated measurements use the same images/metric outputs.
The full export pipeline was also run on the existing Kvasir checkpoint (epoch
197) and all 100 original validation images at 320 × 320 on CPU, with three
warmup forwards and one timed repeat per mode. All five modes gave mean Dice
about 0.88875; this local single-repeat latency preview is not a server/GPU
benchmark. Main and supplementary PNG/PDF/SVG figures, raw timing CSVs and
occupancy arrays were exported, and main-figure rendering was visually checked.

The subsequent Figure 3 paper run used five complete repeats on the same 100
real Kvasir validation images, 30 warmup forwards per mode per repeat, fixed
FP32/batch-1 inference, and two CPU threads. Six routing-figure checks passed,
including balanced order and sample-SD/paired-difference accounting. All 2,500
timings, 500 image/mode metric rows, native grid shapes, balanced orders and saved
artifact hashes were verified. At tau=0.20, CPU latency decreased from 104.434
to 101.439 ms/image, with 98.80% of solver cells skipped and nearly unchanged
Dice. Full results and hardware-qualified paper wording are in
`docs/FIGURE3_PAPER_REPORT.md`. These results are not measurements on the RTX GPU.

The matched upsampling suite passed 50 test functions, including an explicit
CARAFE neighborhood forward/backward reference, DySample's zero-offset bilinear
limit and group-specific offsets, common-weight/RNG equality, and U2/A1 output
equivalence. All five variants completed the toy train/validation/test/table
pipeline and completed-run resume checks. Each full-width model also passed
320 × 320, batch-2 CPU bfloat16-autocast forward/backward checks with finite
gradients. This does not constitute CUDA validation or real Kvasir training.
The prepared real-data manifests exactly match the component suite's
800/100/100 manifests. Details are in `docs/UPSAMPLING_ABLATIONS.md`.
