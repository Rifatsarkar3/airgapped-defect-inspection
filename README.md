# Beyond the Cloud — Review Repository

Code accompanying the EAAI submission *"Beyond the Cloud: An Air-Gapped
Defect-Inspection Framework with Deterministic VRAM Governance and a
Controlled Evaluation of Physics-Guided Regularization"* (double-anonymized
review). This repository contains the complete executable software stack:
the SSF regularizer with its gradient-flow assertion, the paired evaluation
protocol, the three-stage VRAM governance loops, and all statistical and
figure scripts.

## Script → paper map

| Script | Paper section | Produces |
|---|---|---|
| `exp1_statistical_ablation.py` | VI.A (Experiment 1) | Table 2, Fig. 1 |
| `exp11_paired_validation.py` | VI.B–C (Experiment 5) | Tables 3–4 source data |
| `gen_exp11_figures.py` | VI.B–C | Figs. 2–5, paired statistics JSON |
| `exp6_tensorrt_profiling.py` | VI.D (Experiment 6) | Table 5 |
| `exp8_stress_test.py` | VI.E (Experiment 8) | Table 6 |
| `exp9_vram_governance.py` | VI.F (Experiment 9) | Table 7 |
| `gen_fig7_governance.py` | VI.F | Fig. 7 |
| `exp10d_multiseed.py` | VI.G (Experiment 10) | Table 8 (mean ± SD over 5 seeds) |
| `exp10_cross_domain.py` | VI.G (single-seed run with t-SNE: DAGM, Kolektor) | Fig. 9 panels |
| `exp10_neudet_6class.py` | VI.G (single-seed run with t-SNE: NEU-DET, 6-class) | Fig. 9 panels |
| `exp12_budget_sensitivity.py` | VI.G | Table 9 |
| `gen_fig8_crossdomain.py`, `gen_fig9_combined.py` | VI.G | Figs. 8–9 |
| `data_prep/` | III | benchmark folder layouts |

## The SSF regularizer and its gradient-flow assertion

`exp11_paired_validation.py` is the authoritative SSF implementation
(Sec. IV.A): a Sobel gradient-energy hinge on standardized stage-2 feature
maps, evaluated under paired arms with shared initialization, shared data
order, and validation-based model selection. `assert_physics_grad_flow()`
aborts the run if the physics term contributes zero parameter gradient —
the failure mode of the input-space variant discussed in Sec. IV.A, which is
mathematically incapable of training the network and is excluded from all
results.

## Environment

Measured execution environment: Python 3.14.3, PyTorch 2.12.0.dev+cu128,
CUDA 12.8, single NVIDIA RTX 5070 (12 GB), Windows 11. Install via
`pip install -r requirements.txt` (PyTorch from the CUDA 12.8 wheel index).
`exp1`/`exp11` download ImageNet weights from torchvision on first use; in
an air-gapped deployment these are pre-cached locally.

## Datasets (not redistributed)

- **Casting (primary)** — "casting product image data for quality
  inspection" (Pilot Technocast, Rajkot, India), Kaggle, CC BY-NC-ND 4.0:
  https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
  Place as `casting_data/train/{def_front,ok_front}` and
  `casting_data/test/{def_front,ok_front}`. Images are not redistributed
  here, in keeping with the license.
- **NEU-DET** — download the official distribution to `NEU-DET/`, then run
  `python data_prep/prepare_neudet.py` (builds the 6-class
  `NEU-DET_ByClass/` layout; see the script docstring for why this step is
  load-bearing).
- **DAGM 2007** — download to `DAGM 2007-Industrial Defect Detection
  Dataset/`, then `python data_prep/prepare_dagm.py`.
- **KolektorSDD** — download, then `python data_prep/sort_kolektor.py`
  (builds `Kolektor/{Defect,OK}`).

## Reproduction order

```
python exp1_statistical_ablation.py      # ~hours: 30 cycles x 4 architectures
python exp11_paired_validation.py        # ~7 h: 30 paired cycles (resume-safe)
python gen_exp11_figures.py
python exp6_tensorrt_profiling.py
python exp9_vram_governance.py
python exp8_stress_test.py               # 24 h continuous
python exp10d_multiseed.py               # Table 8 statistics (5 seeds)
python exp10_cross_domain.py             # t-SNE/silhouette figures (seed 42)
python exp10_neudet_6class.py
python exp12_budget_sensitivity.py
python gen_fig7_governance.py
python gen_fig8_crossdomain.py
python gen_fig9_combined.py
```

All experiment scripts seed their RNGs explicitly; `exp11` is resume-safe
(rerunning skips completed cycle/arm pairs).

## Notes

- `Models/PGML_Champion_EfficientNetB0.pth` is the deployed champion
  checkpoint profiled in Experiments 6/8/9 and fine-tuned in Experiment 10.
  The filename's "PGML" tag is historical project naming; as established in
  the paper (Sec. VI.B), the checkpoint is cross-entropy-trained.
- Deployment experiments (6/8/9) measure the architecture and inference
  pipeline; their results are independent of the training-loss comparison.
