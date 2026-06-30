# AutoDL Environment Setup

This repository currently has two practical environment targets:

- `requirements.txt`
  Current refactored research pipeline.
  Recommended for `run_research_pipeline.py`.
- `requirements-full.txt`
  Full repository environment.
  Use this only if you still need legacy modules such as the old E5 similarity
  calculator.

## Recommended Base Image

Use an AutoDL image that already contains:

- Python `3.10`
- CUDA-compatible PyTorch

This is the safest approach because PyTorch wheels depend on the CUDA runtime of
the selected server image.

## Quick Install

From the project root:

```bash
conda create -n fair_ncl python=3.10 -y
conda activate fair_ncl
```

If your AutoDL image already ships with a usable PyTorch environment, you can
skip manual torch installation and directly install project dependencies:

```bash
pip install -r requirements.txt
```

If you need the full repository environment:

```bash
pip install -r requirements-full.txt
```

## Verify PyTorch First

Before running experiments, verify that PyTorch can see CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
PY
```

If `import torch` fails or `cuda available` is `False`, install a PyTorch build
that matches the CUDA version of your AutoDL image first, then rerun:

```bash
pip install -r requirements.txt
```

## Which File Should You Use

### Use `requirements.txt` when:

- you run the new main entrypoint:
  `python run_research_pipeline.py ...`
- you export ablation/comparison tables
- you export parameter analysis plots
- you do not rely on the old transformer-based similarity calculator

### Use `requirements-full.txt` when:

- you want the whole repository to be importable
- you may still call legacy modules that depend on `transformers`

## Dependency Rationale

The default environment is built from the actual imports in the repository:

- `numpy`, `pandas`
  preprocessing, reporting, metrics, result aggregation
- `tqdm`
  preprocessing and training progress bars
- `scipy`
  statistical utilities in metrics
- `scikit-learn`
  metrics, mutual information, train/test helpers, pairwise similarity
- `matplotlib`
  parameter sensitivity plots
- `openpyxl`
  Excel export for ablation and comparison tables
- `transformers`
  only needed by the optional legacy E5 similarity calculator

For AutoDL images with older preinstalled PyTorch builds, pin
`transformers==4.37.2` to avoid the known
`torch.utils._pytree.register_pytree_node` compatibility error.

## Recommended Runtime Entry

Prefer the refactored pipeline only:

```bash
python run_research_pipeline.py --help
```

Do not mix legacy entrypoints such as `main_ffvae_comparison.py` with the new
pipeline when generating paper tables, otherwise metric definitions can drift.

## Minimal Smoke Test

After installation, run:

```bash
python run_research_pipeline.py --help
```

Then verify preprocessing help:

```bash
python run_research_pipeline.py preprocess --help
```

If both commands work, the environment is usually ready for dataset processing
and experiment execution.
