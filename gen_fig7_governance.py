# -*- coding: utf-8 -*-
"""Fig 7 replacement: VRAM governance ablation from MEASURED exp9 data
(Results/exp9_pgml_governance_report.txt). Replaces the legacy '(Projected)'
mock-up whose numbers contradicted Table 7."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
conds   = ["A — Default\n(float32)", "B — Partial\n(FP16 only)", "C — Full\nGovernance"]
colors  = ["#C0392B", "#CA8A04", "#1E8449"]
oom     = [0, 0, 0]
fps     = [2237.0, 4435.2, 5221.5]
fps_sd  = [528.0, 1101.4, 1284.2]
cv      = [23.61, 24.83, 24.59]
vram    = [1629, 610, 3644]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("VRAM Governance Ablation — Champion EfficientNet-B0 (measured, 60-min windows)",
             fontsize=13, fontweight="bold")

ax = axes[0][0]
ax.bar(conds, oom, color=colors, alpha=0.85, edgecolor="black", lw=0.7)
for i, v in enumerate(oom):
    ax.text(i, v + 0.02, str(v), ha="center", fontweight="bold")
ax.set_ylabel("OOM count"); ax.set_ylim(0, 1)
ax.set_title("OOM Events (60-min window)")
ax.grid(axis="y", ls="--", alpha=0.3); ax.set_axisbelow(True)

ax = axes[0][1]
ax.bar(conds, fps, yerr=fps_sd, capsize=5, color=colors, alpha=0.85,
       edgecolor="black", lw=0.7)
ax.axhline(3029.6, color="darkred", ls="--", lw=1.4, label="Target = 3,029.6 FPS")
for i, v in enumerate(fps):
    ax.text(i, v + fps_sd[i] + 120, f"{v:,.0f}", ha="center", fontweight="bold",
            fontsize=9)
ax.set_ylabel("Frames per second"); ax.set_ylim(0, 7200)
ax.set_title("Mean FPS ± SD")
ax.legend(fontsize=9); ax.grid(axis="y", ls="--", alpha=0.3); ax.set_axisbelow(True)

ax = axes[1][0]
ax.bar(conds, cv, color=colors, alpha=0.85, edgecolor="black", lw=0.7)
for i, v in enumerate(cv):
    ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)
ax.set_ylabel("FPS CV (%)"); ax.set_ylim(0, 30)
ax.set_title("FPS Coefficient of Variation (60-min window)")
ax.text(0.5, 0.06,
        "Cold-start dominated; steady-state CV = 2.90% over 24 h (Section VI.E)",
        transform=ax.transAxes, ha="center", fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8DC", ec="gray", alpha=0.9))
ax.grid(axis="y", ls="--", alpha=0.3); ax.set_axisbelow(True)

ax = axes[1][1]
ax.bar(conds, vram, color=colors, alpha=0.85, edgecolor="black", lw=0.7)
ax.axhline(4066, color="darkred", ls="--", lw=1.4, label="SCADA limit = 4,066 MB")
for i, v in enumerate(vram):
    ax.text(i, v + 90, f"{v:,} MB", ha="center", fontweight="bold", fontsize=9)
ax.set_ylabel("Peak VRAM (MB)"); ax.set_ylim(0, 5000)
ax.set_title("Peak VRAM (Condition C includes 900-frame ring pre-allocation)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", ls="--", alpha=0.3); ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("Figures/Figure_Exp9_Governance_Measured.png", dpi=160,
            bbox_inches="tight")
plt.close()
print("written Figures/Figure_Exp9_Governance_Measured.png")
