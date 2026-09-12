# Validation

The active dataset configs are `configs/kvasir.yaml`,
`configs/cvc_clinicdb.yaml`, `configs/cvc_colondb.yaml`, and `configs/etis.yaml`.
ColonDB uses the selected rotation-fix settings at seed 42.

Check local data and run the test suite:

```bash
python scripts/check_dataset.py --config configs/cvc_colondb.yaml
python -m pytest
```

To evaluate the completed winning run on the server:

```bash
python evaluate.py --checkpoint runs/rocu_cvc_colondb_rotation_fix_seed42/best.pt --split test --save-predictions --device cuda
```

The reported selected validation Dice is 0.8837 and IoU is 0.8102. Synthetic
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
