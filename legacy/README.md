# legacy/ — superseded scripts kept for provenance only

These scripts are **not** part of the corrected protocol and their accuracy numbers are
**not** evidence for any claim in the paper. They are kept because files they produced are
still referenced:

| Script | Why it is kept |
|---|---|
| `exp2_vram_and_runtime.py` | 24-h FP32 baseline run (1,000-frame ring buffer). Its throughput (3,029.6 FPS) and allocator peak (4,386.4 MB) are the baseline column of Table 3 and Fig. 2. It loaded a pre-correction Exp 1 weight file that is no longer distributed; any EfficientNet-B0 weights reproduce its measurements, which depend on architecture and precision only (Experiment 16). |
| `exp5_pgml_training.py` | Produced `Models/PGML_Champion_EfficientNetB0.pth`, the EfficientNet-B0 weight file profiled in Experiments 6, 8 and 9. Its physics term was inert (computed under `no_grad` on input pixels) and its epoch and seed were selected on the test set, so it is used **only** as an EfficientNet-B0 weight file for architecture-level profiling; `exp16_edge_backbone_profiling.py` shows throughput and memory are identical for the validation-selected Gate-1 champion. |
| `exp5b_pgml_n30.py` | The superseded n = 30 run whose 99.865% ± 0.067% figure is decomposed in Section 7 (`exp14_broken_mystery.py`, `exp14b_selection.py`). |

Raw outputs of these runs are in `Results/legacy_provenance/` of the project workspace.
