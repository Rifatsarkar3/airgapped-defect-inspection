# Certifying Edge Defect-Inspection Models for Air-Gapped Deployment — review repository

Code for the manuscript *"Certifying Edge Defect-Inspection Models for Air-Gapped Deployment:
Reproducibility, Graphics-Processor Memory Governance, and Paired Screening of Training-Time
Regularizers"*. It contains the complete executable stack:
the SSF regularizer with its gradient-flow assertion, the paired evaluation and regularizer-screening
protocols, the three-stage memory-governance and backbone-profiling scripts, and all statistical and
figure scripts.

## Script → paper map

| Script | Paper section | Produces |
|---|---|---|
| `exp1_statistical_ablation_v2.py` | 6.1 (Experiment 1) | Table 2, Fig. 1, confirmatory test statistics |
| `legacy/exp2_vram_and_runtime.py` | 6.2 (Experiment 2) | FP32 baseline column of Table 3 |
| `exp6_tensorrt_profiling.py` | 6.2 (Experiment 6) | Table 3 |
| `gen_fig2_deployment_profile.py` | 6.2 | Fig. 2 (reads the Exp 2 and Exp 6 reports) |
| `exp8_stress_test.py` | 6.3 (Experiment 8) | Table 4 |
| `exp9_vram_governance_v2.py` | 6.4 (Experiment 9) | Table 5, Fig. 3 |
| `exp10d_multiseed.py` | 6.5 (Experiment 10) | Table 6 (mean ± SD over 5 seeds) |
| `exp10_cross_domain.py`, `exp10_neudet_6class.py` | 6.5 (seed-42 runs with t-SNE) | Fig. 5 panels |
| `exp12_budget_sensitivity.py` | 6.5 (Experiment 12) | Table 7 |
| `gen_fig8_crossdomain.py`, `gen_fig9_combined.py` | 6.5 | Figs. 4–5 |
| `exp11_paired_validation_v2.py` | 6.6–6.7 (Experiment 5) | Tables 8–9 source data |
| `gen_exp11v2_figures.py` | 6.6–6.7 | Figs. 6–9, paired statistics JSON |
| `exp13_gradient_alignment.py` | 6.7, Eqs. (6)–(7) | gradient alignment A(θ), η |
| `exp15_regularizer_comparison.py` | 6.8 (Experiment 15) | Table 10 |
| `exp16_edge_backbone_profiling.py` | 6.9 (Experiment 16) | Table 11 |
| `protocol_validation.py` | 4.1 (gradient-flow gate) | accepts the corrected SSF term, rejects the legacy one |
| `exp14_broken_mystery.py`, `exp14b_selection.py` | 7 (decomposition) | inert term × initialization; validation- vs test-epoch selection |
| `data_prep/` | 3 | benchmark folder layouts |
| `legacy/` | provenance only | see `legacy/README.md` |

## Checkpoints (`Models/`)

- `Gate1_Champion_EfficientNetB0.pth` — the casting-pretrained backbone of Experiments 10, 12, 13 and
  16: the Experiment 1 EfficientNet-B0 checkpoint with the highest **validation** F1 (ties broken by
  validation recall, validation accuracy and cycle index). Its test result played no part in the choice.
- `PGML_Champion_EfficientNetB0.pth` — the EfficientNet-B0 weight file profiled in Experiments 6, 8 and
  9. It comes from the superseded run described in `legacy/README.md` (inert physics term, test-set
  epoch selection) and is used only as an instance of the architecture: `exp16_edge_backbone_profiling.py`
  shows identical allocator peaks and throughput within 0.3% for the Gate-1 champion.
- `exp1_statistical_ablation_v2.py` and `exp11_paired_validation_v2.py` also write convenience copies
  (`Champion_v2_*.pth`, `Exp11v2_PGML_Champion.pth`) chosen by best *test* F1. No reported result uses them.

## Corrected protocol versions (`_v2` scripts)

`exp1_statistical_ablation_v2.py`, `exp9_vram_governance_v2.py` and `exp11_paired_validation_v2.py`
supersede earlier internal versions with methodological bugs found in our own pre-submission audit:

- **Test-set leakage in architecture selection (Experiment 1).** The original script had no validation
  split. `_v2` carves a stratified 15% validation split from the training partition, selects on
  validation, and only afterwards evaluates each of the selected architecture's 30 checkpoints once on
  the test set.
- **Recall/F1 positive-class inversion (Experiments 1 and 5).** `ImageFolder` sorts `def_front` before
  `ok_front`, and the original `evaluate()` treated the OK class as positive. `_v2` uses the defective
  class. With recall corrected, SSF shows no significant effect on any metric.
- **Confounded memory-governance ablation (Experiment 9).** `_v2` changes exactly one factor per step.

All three are resume-safe; `exp15_regularizer_comparison.py` is resume-safe per (cycle, arm).

## Environment

Python 3.14.3, PyTorch 2.12.0.dev+cu128, CUDA 12.8, cuDNN 9.20, single NVIDIA RTX 5070 (12 GB),
Windows 11 Pro, default (unlocked) clocks. Install with `pip install -r requirements.txt` (PyTorch from
the CUDA 12.8 wheel index). `exp1v2`/`exp11v2`/`exp15` download ImageNet weights from torchvision on
first use; in an air-gapped deployment these are pre-cached. `exp16` needs `timm` but no downloads.

## Datasets (not redistributed)

- **Casting (primary)** — "casting product image data for quality inspection" (Pilot Technocast, Rajkot,
  India), Kaggle, CC BY-NC-ND 4.0:
  https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
  Place as `casting_data/train/{def_front,ok_front}` and `casting_data/test/{def_front,ok_front}`.
- **NEU-DET** — download to `NEU-DET/`, then `python data_prep/prepare_neudet.py` (builds the 6-class
  `NEU-DET_ByClass/` layout; see the script docstring for why this step is load-bearing).
- **DAGM 2007** — download to `DAGM 2007-Industrial Defect Detection Dataset/`, then
  `python data_prep/prepare_dagm.py`.
- **KolektorSDD** — download, then `python data_prep/sort_kolektor.py` (builds `Kolektor/{Defect,OK}`).

## Reproduction order

```
python exp1_statistical_ablation_v2.py   # ~12 h: 30 cycles x 4 architectures
python exp11_paired_validation_v2.py     # ~8 h: 30 paired cycles
python gen_exp11v2_figures.py
python exp15_regularizer_comparison.py   # ~11 h: 3 regularizer arms x 30 cycles + 5-seed CE re-run
python exp6_tensorrt_profiling.py
python gen_fig2_deployment_profile.py
python exp9_vram_governance_v2.py        # ~4 h: 4 conditions
python exp8_stress_test.py               # 24 h continuous
python exp16_edge_backbone_profiling.py  # ~10 min, idle GPU
python exp10d_multiseed.py               # Table 6 (5 seeds)
python exp12_budget_sensitivity.py
python exp10_cross_domain.py             # t-SNE/silhouette panels (seed 42)
python exp10_neudet_6class.py
python exp13_gradient_alignment.py
python gen_fig8_crossdomain.py
python gen_fig9_combined.py
```

All scripts seed their RNGs explicitly and run from the repository root.
