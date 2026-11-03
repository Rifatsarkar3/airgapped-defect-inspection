# -*- coding: utf-8 -*-
"""Combined Fig 9: 2x3 t-SNE grid (top: zero-shot, bottom: after 15% FT)
from exp10b npz dumps. Honest labels, no legacy tags."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

PANELS = [("(a) NEU-DET", "NEUDET"), ("(b) DAGM", "DAGM"), ("(c) Kolektor", "Kolektor")]
fig, axes = plt.subplots(2, 3, figsize=(16, 7))
fig.suptitle("Champion backbone t-SNE projections — zero-shot (top) vs after "
             "15% CE fine-tune (bottom)", fontsize=13, fontweight="bold")
cmap = plt.get_cmap("tab10")

for col, (label, safe) in enumerate(PANELS):
    z = np.load(f"Results/exp10R_tsne_{safe}.npz", allow_pickle=True)
    labels, names = z["labels"], list(z["class_names"])
    for row, (proj, sil, tag) in enumerate([
            (z["tsne_b"], float(z["sil_b"]), "zero-shot"),
            (z["tsne_a"], float(z["sil_a"]), "after 15% FT")]):
        ax = axes[row][col]
        for ci, cn in enumerate(names):
            m = labels == ci
            if m.any():
                ax.scatter(proj[m, 0], proj[m, 1], c=[cmap(ci % 10)],
                           label=str(cn), s=12, alpha=0.75, linewidths=0)
        ax.set_title(f"{label} — {tag}", fontsize=10)
        ax.text(0.02, 0.98, f"Silhouette = {sil:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.legend(fontsize=7, framealpha=0.7, markerscale=1.3, loc="lower right")
plt.tight_layout()
plt.savefig("Figures/Figure_Exp10R_Combined_tSNE.png", dpi=350, bbox_inches="tight")
plt.close()
print("written Figures/Figure_Exp10R_Combined_tSNE.png")
