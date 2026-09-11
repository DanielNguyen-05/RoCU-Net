# Validation status

Validated in the build workspace on 2026-09-06:

- every Python source file passes `compileall` and AST parsing;
- all YAML files parse successfully;
- inherited ablation configurations resolve the expected base settings;
- the synthetic Kvasir generator creates paired image/mask files;
- the paper comparison script successfully reads the existing OCU-Net
  `summary.json` format and emits CSV and Markdown tables;
- the original OCU-Net implementation and configuration remain selectable.

Runtime model tests and real Kvasir training could not be executed in the build
workspace because that runtime did not contain PyTorch and
`dataset/Kvasir` contained only its placement README. After installing
`requirements.txt` and copying Kvasir-SEG, run:

```bash
python scripts/create_toy_kvasir.py --output dataset/ToyKvasir --count 24 --size 128
python train.py --config configs/smoke_crs.yaml --device cpu
pytest
python scripts/check_dataset.py --root dataset/Kvasir
python train.py --config configs/kvasir_crs.yaml --device cuda
```

Synthetic smoke metrics must not be used as research results.

