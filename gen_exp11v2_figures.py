# -*- coding: utf-8 -*-
"""
gen_exp11v2_figures.py — Figures 6-9 regenerated from the recall-fixed exp11v2 data.

Same logic as anon_repo/gen_exp11_figures.py (including the Fig-8 mean+SD upper-clip
fix), retargeted at the exp11v2 outputs instead of the superseded exp11 v1 files, and
writing to a v2-suffixed stats file so the v1 exp11_stats.json is preserved for audit.

Reads:  Results/exp11v2_paired_results.csv, Results/exp11v2_epoch_log.csv
Writes: Figures/Figure_Exp11_PairedCycles.png      (Fig 6 slot, ~1780x740)
        Figures/Figure_Exp11_ArmComparison.png     (Fig 7 slot, ~1780x740)
        Figures/Figure_Exp11_TrainingDynamics.png  (Fig 8 slot, ~1780x740)
        Figures/Figure_Exp11_Distributions.png     (Fig 9 slot, ~4770x1779)
        Results/exp11v2_stats.json                 (numbers for manuscript)
"""
import csv, json, sys
import numpy as np
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
C_BASE, C_SSF = "#C0392B", "#2E5C8A"

rows = list(csv.DictReader(open("Results/exp11v2_paired_results.csv", encoding="utf-8")))
arms = {"baseline": {}, "pgml": {}}
for r in rows:
    arms[r["arm"]][int(r["cycle"])] = r
common = sorted(set(arms["baseline"]) & set(arms["pgml"]))
N = len(common)
print(f"pairs: {N}")

def vec(arm, key):
    return np.array([float(arms[arm][c][key]) for c in common])

stats_out = {"n_pairs": N}
for key, label in [("acc", "accuracy"), ("recall", "recall"), ("f1", "f1")]:
    b, p = vec("baseline", key), vec("pgml", key)
    d = p - b
    t, pv = st.ttest_rel(p, b)
    s_, d_ = p + b, p - b
    pm_r, pm_p = st.pearsonr(s_, d_)        # Pitman-Morgan paired variance test
    stats_out[label] = {
        "base_mu": round(b.mean(), 4), "base_sd": round(b.std(ddof=1), 4),
        "ssf_mu": round(p.mean(), 4), "ssf_sd": round(p.std(ddof=1), 4),
        "delta_mu": round(d.mean(), 4), "delta_sd": round(d.std(ddof=1), 4),
        "t": round(t, 3), "p": round(pv, 4),
        "sd_ratio": round(b.std(ddof=1) / p.std(ddof=1), 2) if p.std() > 0 else None,
        "pm_p": round(pm_p, 4),
    }
json.dump(stats_out, open("Results/exp11v2_stats.json", "w"), indent=2)
print(json.dumps(stats_out, indent=2))

acc_b, acc_p = vec("baseline", "acc"), vec("pgml", "acc")
rc_b,  rc_p  = vec("baseline", "recall"), vec("pgml", "recall")
f1_b,  f1_p  = vec("baseline", "f1"), vec("pgml", "f1")

# ---- Fig 6 slot: per-cycle paired accuracy ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.plot(common, acc_b, "o-", color=C_BASE, lw=1.2, ms=4, label="Baseline (CE)")
ax.plot(common, acc_p, "s-", color=C_SSF, lw=1.2, ms=4, label="SSF (CE + physics)")
ax.set_xlabel("Paired cycle (shared seed)"); ax.set_ylabel("Test accuracy (%)")
ax.set_title("Per-cycle test accuracy, identical initialization")
ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.3)
ax = axes[1]
d = acc_p - acc_b
ax.bar(common, d, color=[C_SSF if x >= 0 else C_BASE for x in d], alpha=0.8)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(d.mean(), color="navy", ls="--", lw=1.2,
           label=f"Mean paired delta = {d.mean():+.3f} pp")
ax.set_xlabel("Paired cycle"); ax.set_ylabel("Accuracy delta (SSF - Baseline, pp)")
ax.set_title("Within-pair accuracy difference")
ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("Figures/Figure_Exp11_PairedCycles.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Fig 7 slot: arm comparison bars + deltas ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = ["Accuracy", "Recall", "F1-Score"]
bmu = [acc_b.mean(), rc_b.mean(), f1_b.mean()]
bsd = [acc_b.std(ddof=1), rc_b.std(ddof=1), f1_b.std(ddof=1)]
pmu = [acc_p.mean(), rc_p.mean(), f1_p.mean()]
psd = [acc_p.std(ddof=1), rc_p.std(ddof=1), f1_p.std(ddof=1)]
x = np.arange(3); w = 0.36
ax = axes[0]
ax.bar(x - w/2, bmu, w, yerr=bsd, capsize=4, color=C_BASE, alpha=0.85, label="Baseline (CE)")
ax.bar(x + w/2, pmu, w, yerr=psd, capsize=4, color=C_SSF, alpha=0.85, label="SSF (CE + physics)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Test metric (%)"); ax.set_ylim(98.6, 100.4)
ax.set_title(f"Paired arms, n={N} shared-seed cycles")
ax.legend(fontsize=9, loc="lower right"); ax.grid(ls="--", alpha=0.3, axis="y")
ax = axes[1]
deltas = [acc_p - acc_b, rc_p - rc_b, f1_p - f1_b]
mus = [v.mean() for v in deltas]
cis = [1.96 * v.std(ddof=1) / np.sqrt(N) for v in deltas]
ax.bar(x, mus, 0.5, yerr=cis, capsize=5,
       color=[C_SSF if m >= 0 else C_BASE for m in mus], alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Paired delta (SSF - Baseline, pp)")
ax.set_title("Within-pair differences with 95% CI")
ax.grid(ls="--", alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("Figures/Figure_Exp11_ArmComparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Fig 8 slot: training dynamics from epoch log --------------------------
ep_rows = list(csv.DictReader(open("Results/exp11v2_epoch_log.csv", encoding="utf-8")))
def epoch_series(arm, key):
    by_ep = {}
    for r in ep_rows:
        if r["arm"] == arm and int(r["cycle"]) in common:
            by_ep.setdefault(int(r["epoch"]), []).append(float(r[key]))
    eps = sorted(by_ep)
    return eps, np.array([np.mean(by_ep[e]) for e in eps]), \
           np.array([np.std(by_ep[e]) for e in eps])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
for arm, c, lbl in [("baseline", C_BASE, "Baseline (CE)"), ("pgml", C_SSF, "SSF (CE + physics)")]:
    eps, mu, sd = epoch_series(arm, "val_f1")
    ax.plot(eps, mu, "-o", color=c, ms=3, lw=1.4, label=lbl)
    ax.fill_between(eps, np.maximum(mu - sd, 0), np.minimum(mu + sd, 100), color=c, alpha=0.15)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation F1 (%)")
ax.set_title(f"Validation F1 convergence (mean +/- SD, n={N} cycles)")
ax.legend(fontsize=9, loc="lower right"); ax.grid(ls="--", alpha=0.3)
ax = axes[1]
eps, mu, sd = epoch_series("pgml", "phys_avg")
ax.plot(eps, mu, "-s", color=C_SSF, ms=3, lw=1.4, label="Mean physics hinge value")
ax.fill_between(eps, np.maximum(mu - sd, 0), mu + sd, color=C_SSF, alpha=0.15)
ax.set_xlabel("Epoch"); ax.set_ylabel(r"$L_{SSF}$ (hinge value)")
ax.set_title("Physics term during training (SSF arm)")
ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig("Figures/Figure_Exp11_TrainingDynamics.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Fig 9 slot: distributions ---------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, (lbl, b, p) in zip(axes, [("Accuracy", acc_b, acc_p),
                                  ("Recall", rc_b, rc_p),
                                  ("F1-Score", f1_b, f1_p)]):
    vp = ax.violinplot([b, p], positions=[1, 2], showmedians=True)
    for body, c in zip(vp["bodies"], [C_BASE, C_SSF]):
        body.set_facecolor(c); body.set_alpha(0.45)
    vp["cmedians"].set_color("black")
    ax.scatter([1] * N, b, color=C_BASE, s=14, alpha=0.6, zorder=3)
    ax.scatter([2] * N, p, color=C_SSF, s=14, alpha=0.6, zorder=3)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Baseline (CE)", "SSF"])
    ax.set_ylabel(f"{lbl} (%)"); ax.set_title(lbl)
    ax.grid(ls="--", alpha=0.3); ax.set_axisbelow(True)
    ax.text(0.97, 0.03,
            f"Baseline: {b.mean():.3f}+/-{b.std(ddof=1):.3f}\n"
            f"SSF: {p.mean():.3f}+/-{p.std(ddof=1):.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
fig.suptitle(f"Paired evaluation distributions across n={N} shared-seed cycles",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("Figures/Figure_Exp11_Distributions.png", dpi=300, bbox_inches="tight")
plt.close()
print("figures written")
