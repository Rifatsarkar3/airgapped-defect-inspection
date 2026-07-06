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
| `exp1_statistical_ablation_v2.py` | VI.A (Experiment 1) | Table 2, Fig. 1 |
| `exp6_tensorrt_profiling.py` | VI.B (Experiment 6) | Table 3 |
| `exp8_stress_test.py` | VI.C (Experiment 8) | Table 4 |
| `exp9_vram_governance_v2.py` | VI.D (Experiment 9) | Table 5, Fig. 3 |
| `exp10d_multiseed.py` | VI.E (Experiment 10) | Table 6 (mean ± SD over 5 seeds) |
| `exp10_cross_domain.py` | VI.E (single-seed run with t-SNE: DAGM, Kolektor) | Fig. 5 panels |
| `exp10_neudet_6class.py` | VI.E (single-seed run with t-SNE: NEU-DET, 6-class) | Fig. 5 panels |
| `exp12_budget_sensitivity.py` | VI.E | Table 7 |
| `gen_fig8_crossdomain.py`, `gen_fig9_combined.py` | VI.E | Figs. 4–5 |
| `exp11_paired_validation_v2.py` | VI.F–G (Experiment 5) | Tables 8–9 source data |
| `gen_exp11v2_figures.py` | VI.F–G | Figs. 6–9, paired statistics JSON |
| `exp13_gradient_alignment.py` | VI.G / Eqs. (3)–(4) | gradient alignment A(θ), η |
| `protocol_validation.py` | IV (gradient-flow gate) | accepts corrected SSF, rejects legacy term |
| `exp14_broken_mystery.py` | VII (decomposition) | inert-term × initialization (2×2) |
| `exp14b_selection.py` | VII (decomposition) | validation- vs test-epoch selection |
| `data_prep/` | III | benchmark folder layouts |

## Corrected protocol versions (`_v2` scripts)

`exp1_statistical_ablation_v2.py`, `exp9_vram_governance_v2.py`, and
`exp11_paired_validation_v2.py` supersede three earlier internal versions
that had methodological bugs caught during our own pre-submission audit and
fixed before this repository was finalized. Documented here for
transparency, since the corrections changed one reported finding:

- **Test-set leakage in architecture selection (Experiment 1).** The
  original script had no validation split — all 120 training runs (30
  cycles × 4 architectures) were evaluated directly against the held-out
  test partition, and those same numbers both selected the champion
  architecture and were reported as its deployed accuracy. `_v2` carves a
  stratified 15% validation split from the training partition, selects on
  validation for all cycles, and touches the test set exactly once at the
  end for a confirmatory pass on the selected architecture only.
- **Recall/F1 positive-class inversion (Experiments 1 and 5/11).** Because
  the dataset's class folders sort `def_front` (defective) before
  `ok_front` alphabetically, the original `evaluate()` hardcoded the
  nominal/OK class as positive, so every "recall" was the OK-class's
  recall, not the defective-class catch rate the paper's safety framing
  assumes. `_v2` fixes the positive class to defective across both scripts.
  This was the more consequential of the two fixes: it does not just change
  the numbers in Table 2, it reverses the paper's only originally
  statistically significant finding in the paired SSF evaluation (Sec.
  VI.F–G) — with recall corrected, SSF shows no significant effect on any
  metric, mean or variance, consistent with the gradient-alignment
  diagnostic (η ≈ 0.98) already in the paper.
- **Confounded VRAM governance ablation (Experiment 9).** The original
  3-condition ablation changed precision, batch size, CUDA Graph, and
  SCADA-contention handling simultaneously between conditions, so throughput
  and VRAM deltas could not be attributed to a single factor. `_v2` adds a
  4th condition and fixes batch size at 32 throughout, changing exactly one
  factor per step.

All three `_v2` scripts are resume-safe at a fine grain (per cycle/arm for
exp1v2 and exp11v2, per condition for exp9v2 — condition-level because peak
VRAM is an in-process CUDA counter that resets per condition and can't be
reconstructed from partial samples): an interrupted run can simply be
restarted with the same command.

## The SSF regularizer and its gradient-flow assertion

`exp11_paired_validation_v2.py` is the authoritative SSF implementation
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
`exp1v2`/`exp11v2` download ImageNet weights from torchvision on first use;
in an air-gapped deployment these are pre-cached locally.

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
python exp1_statistical_ablation_v2.py   # ~12h: 30 cycles x 4 architectures (resume-safe)
python exp11_paired_validation_v2.py     # ~8h: 30 paired cycles (resume-safe)
python gen_exp11v2_figures.py
python exp6_tensorrt_profiling.py
python exp9_vram_governance_v2.py        # ~4h: 4 conditions (resume-safe)
python exp8_stress_test.py               # 24 h continuous
python exp10d_multiseed.py               # Table 6 statistics (5 seeds)
python exp10_cross_domain.py             # t-SNE/silhouette figures (seed 42)
python exp10_neudet_6class.py
python exp12_budget_sensitivity.py
python gen_fig8_crossdomain.py
python gen_fig9_combined.py
```

All experiment scripts seed their RNGs explicitly; `exp1v2`, `exp9v2`, and
`exp11v2` are resume-safe (rerunning skips completed work rather than
starting over).

## Notes

- `Models/PGML_Champion_EfficientNetB0.pth` is the deployed champion
  checkpoint profiled in Experiments 6/8/9 and fine-tuned in Experiment 10.
  The filename's "PGML" tag is historical project naming; as established in
  the paper (Sec. VI.F), the checkpoint is cross-entropy-trained.
- Deployment experiments (6/8/9) measure the architecture and inference
  pipeline; their results are independent of the training-loss comparison.
- `exp14_broken_mystery.py` loads `Models/Best_EfficientNetB0_Exp1.pth`, the
  champion checkpoint from the pre-correction Experiment 1 run, used only to
  decompose the superseded formulation's inflated numbers (Sec. VII) — not
  otherwise part of the reported results.
