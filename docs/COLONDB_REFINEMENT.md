# ColonDB refinement

The newly downloaded run `rocu_cvc_colondb_seed42` selects epoch 92. Its saved
server test scores are Dice 0.839675 and IoU 0.759448. This is a different run
from the earlier reported test Dice 0.8470.

On this checkpoint, a CPU FP32 comparison using the original 38-image validation
split selected four-view flip averaging over single-view inference. The input
resolution remains 320 x 320 and the threshold remains 0.5. Test was evaluated
after choosing the policy on validation:

| Split / inference | Dice | IoU | Recall | Precision | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| Validation / single | 0.8866 | 0.8102 | 0.9011 | 0.8945 | 0.7388 |
| Validation / flip | 0.8901 | 0.8140 | 0.9012 | 0.8975 | 0.7500 |
| Test / single | 0.8399 | 0.7598 | 0.8312 | 0.9161 | 0.6791 |
| Test / flip | 0.8516 | 0.7711 | 0.8335 | 0.9266 | 0.7022 |

These CPU results differ slightly from the saved CUDA AMP scores. Comparison
artifacts, per-image metrics and the checkpoint SHA-256 are under
`reports/colondb_refinement/`. Existing run results and weights were preserved.
TTA gives a modest measured improvement; it does not establish a 0.90 test Dice.
Recall remains substantially below precision on test, consistent with missed
foreground. This observation alone does not prove which training change will
improve generalization.

## Use the existing checkpoint

```bash
python evaluate.py --checkpoint runs/rocu_cvc_colondb_seed42/best.pt \
  --split test --tta flip --save-predictions --profile --device cuda
```

The derived results go to `runs/rocu_cvc_colondb_seed42/evaluation_flip/`.
Prediction, profiling and visualization accept the same `--tta flip` option:

```bash
python visualize.py --checkpoint runs/rocu_cvc_colondb_seed42/best.pt \
  --tta flip --num-samples 5 --output figures/colondb_flip --device cuda
```

TTA averages the identity, horizontal, vertical and combined flips after mapping
every output back to the original orientation. Masks and the threshold are not
changed. Report the method as RoCU-Net + flip TTA when using these scores.
Profiling with `--tta flip` measures all four forward passes, including their
MACs and latency. It is more expensive than single-view inference. New ColonDB
configs enable this inference policy; other dataset defaults remain unchanged.

## Fine-tune from this checkpoint

```bash
python scripts/refine_colondb.py \
  --checkpoint runs/rocu_cvc_colondb_seed42/best.pt --device cuda
```

This launches one seed-preserving refinement run, with its full config saved
under `runs/rocu_cvc_colondb_refined_seed42/refinement_config.yaml`:

- Load the architecture, losses, augmentations and weights from the checkpoint.
- Copy all original split manifests; require a separate, empty output directory.
- Use learning rate 5e-5, backbone multiplier 0.1 and frozen encoder BN statistics.
- Vary training sizes by factors 0.9, 1.0, 1.1 and 1.2; keep validation/test at
  the checkpoint resolution. Images and binary masks are resized together.
- Train up to 60 epochs, early stopping patience 15; retain the initialization
  as `best.pt` unless validation Dice improves under the same flip-TTA policy.
- Evaluate test only after checkpoint selection. Save original model state keys
  and the inference policy in the checkpoint, so other tools automatically use it.

This training recipe is a candidate for improving scale robustness and reducing
disruption to learned encoder features. Its test gain has not been measured;
retaining the best validation score does not guarantee a better test score.
This is not an ablation sweep or a multi-seed run. Use `--output` for a different
destination, or the regular `train.py --resume` command with the generated config
to resume an interrupted refinement.

## Software checks

All 32 test functions passed locally, including alignment of all flip views,
occupancy conservation after averaging, unchanged source weights, four-pass MAC
counting and paired image/mask resizing during training. A two-epoch synthetic
refinement also passed initialization fallback, split copying, train/test,
prediction, profiling and visualization with five examples. Synthetic scores
are not research results.
