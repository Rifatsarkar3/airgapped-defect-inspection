# -*- coding: utf-8 -*-
"""Fig 8: cross-domain fine-tune accuracy and silhouette by initialization,
mean ± SD over the 5-seed protocol (reads Results/exp10D_multiseed.csv).
Zero-shot bars are omitted: random-head zero-shot is head-init noise."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

rows = list(csv.DictReader(open("Results/exp10D_multiseed.csv", encoding="utf-8")))
DS = ["NEU-DET", "DAGM", "Kolektor"]
LBL = ["NEU-DET\n(6-class)", "DAGM\n(binary)", "Kolektor\n(binary)"]


def stat(ds, init, budget, key):
    v = np.array([float(r[key]) for r in rows
                  if r["dataset"] == ds and r["init"] == init
                  and abs(float(r["budget"]) - budget) < 1e-6])
    return v.mean(), v.std(ddof=1)


x = np.arange(3); w = 0.19
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Cross-Domain Transfer by Initialization — CE Fine-Tune, 5 Epochs, "
             "mean ± SD over 5 seeds", fontsize=13, fontweight="bold")

ax = axes[0]
series = [("ImageNet-init, 15% budget", "imagenet", 0.15, "#888888", ""),
          ("Champion-init, 15% budget", "champion", 0.15, "#2E5C8A", ""),
          ("ImageNet-init, 100% data", "imagenet", 1.00, "#BBBBBB", "//"),
          ("Champion-init, 100% data", "champion", 1.00, "#5B9BD5", "//")]
for i, (lbl, init, budget, c, h) in enumerate(series):
    mus, sds = zip(*[stat(ds, init, budget, "acc_ft") for ds in DS])
    off = (i - 1.5) * w
    bars = ax.bar(x + off, mus, w, yerr=sds, capsize=4, label=lbl, color=c,
                  edgecolor="black", lw=0.7, hatch=h, alpha=0.85)
    for b, m, s in zip(bars, mus, sds):
        ax.text(b.get_x() + b.get_width()/2, m + s + 1.2, f"{m:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=c, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(LBL, fontsize=11, fontweight="bold")
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 118)
ax.set_title("Fine-Tune Accuracy by Initialization and Budget")
ax.legend(fontsize=9, loc="lower right", ncol=2)
ax.grid(axis="y", ls="--", alpha=0.4); ax.set_axisbelow(True)

ax = axes[1]
for i, (lbl, init, c) in enumerate([("ImageNet-init", "imagenet", "#888888"),
                                    ("Champion-init", "champion", "#2E5C8A")]):
    mus, sds = zip(*[stat(ds, init, 0.15, "sil_after") for ds in DS])
    off = (i - 0.5) * 0.3
    bars = ax.bar(x + off, mus, 0.3, yerr=sds, capsize=4, label=lbl, color=c,
                  edgecolor="black", lw=0.7, alpha=0.85)
    for b, m in zip(bars, mus):
        ax.text(b.get_x() + b.get_width()/2, m + 0.012, f"{m:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=c)
ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(LBL, fontsize=11, fontweight="bold")
ax.set_ylabel("Silhouette score (after 15% FT)")
ax.set_title("Latent-Space Separability (15% budget)")
ax.legend(fontsize=9); ax.grid(axis="y", ls="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("Figures/Figure_Exp10R_Champion_vs_Baseline.png", dpi=150,
            bbox_inches="tight")
plt.close()
print("written Figures/Figure_Exp10R_Champion_vs_Baseline.png")
