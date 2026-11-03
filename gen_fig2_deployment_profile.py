"""
gen_fig2_deployment_profile.py — Fig. 2: FP32 baseline (Exp 2, 1,000-frame proxy) versus
FP16 + CUDA Graph (Exp 6, 900-frame proxy): throughput and PyTorch allocator peak.

Reads the numbers from the two experiment reports instead of hard-coding them:
    Results/exp2_deployment_summary.txt   (or Results/legacy_provenance/, see legacy/README.md)
    Results/exp6_profiling_report.txt
Output: Figures/Figure_Exp6_DeploymentProfile_Honest.png
"""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGET_MB = 4066


def grab(path, pattern):
    m = re.search(pattern, open(path, encoding="utf-8", errors="replace").read())
    if not m:
        raise ValueError(f"{pattern!r} not found in {path}")
    return float(m.group(1))


exp2 = next(p for p in ("Results/exp2_deployment_summary.txt",
                        "Results/legacy_provenance/exp2_deployment_summary.txt") if os.path.exists(p))
exp6 = "Results/exp6_profiling_report.txt"
fps = [grab(exp2, r"Global mean FPS\s*:\s*([\d.]+)"), grab(exp6, r"FPS\s*:\s*([\d.]+)")]
mem = [grab(exp2, r"Peak VRAM used\s*:\s*([\d.]+)"), grab(exp6, r"Peak VRAM\s*:\s*([\d.]+)")]

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "axes.labelweight": "bold",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
labels = ["FP32 baseline (Exp 2)\n1,000-frame proxy", "FP16 + CUDA Graph (Exp 6)\n900-frame proxy"]
colors = ["#C0392B", "#1E8449"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
bars = ax.bar(labels, fps, width=0.5, color=colors, alpha=0.85)
bars[1].set_hatch("//")
ax.axhline(fps[0], color="red", ls="--", lw=1.5, label=f"Baseline target: {fps[0]:,.1f} FPS")
for i, v in enumerate(fps):
    ax.text(i, v + 120, f"{v:,.1f} FPS\n({v / fps[0]:.2f}x)", ha="center", fontweight="bold", fontsize=9)
ax.set_ylabel("Throughput (images/s)")
ax.set_ylim(0, 8000)
ax.set_title("Inference throughput, RTX 5070, batch 32")
ax.legend(fontsize=9)
ax.grid(ls="--", alpha=0.3, axis="y")
ax.set_axisbelow(True)

ax = axes[1]
bars = ax.bar(labels, mem, width=0.5, color=colors, alpha=0.85)
bars[1].set_hatch("//")
ax.axhline(BUDGET_MB, color="red", ls="--", lw=1.5, label=f"Study budget: {BUDGET_MB:,} MB")
ax.axhspan(BUDGET_MB, 5400, color="red", alpha=0.08, label="Budget exceeded")
for i, v in enumerate(mem):
    ok = v < BUDGET_MB
    ax.text(i, v + 60, f"{v:,.1f} MB [{'PASS' if ok else 'FAIL'}]", ha="center",
            color=colors[i], fontweight="bold", fontsize=9)
ax.set_ylabel("PyTorch allocator peak (MB)")
ax.set_ylim(0, 5400)
ax.set_title("Allocator peak with synthetic ring-buffer proxy")
ax.legend(fontsize=9, loc="lower right")
ax.grid(ls="--", alpha=0.3, axis="y")
ax.set_axisbelow(True)

plt.tight_layout()
os.makedirs("Figures", exist_ok=True)
plt.savefig("Figures/Figure_Exp6_DeploymentProfile_Honest.png", dpi=350, bbox_inches="tight")
print("written Figures/Figure_Exp6_DeploymentProfile_Honest.png", fps, mem)
