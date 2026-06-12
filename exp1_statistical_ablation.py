"""
exp1_statistical_ablation.py  —  Paper 1 Final Pipeline
=========================================================
Evaluates 4 architectures on casting_data (binary defect classification).
Identifies the DEPLOYMENT CHAMPION using a composite score that weights
accuracy, recall, F1, and variance equally — not raw accuracy alone.

Key outputs
-----------
  Models/Best_{Arch}_Exp1.pth          — best weights per architecture
  Models/Champion_MobileNetV3_Exp1.pth — explicitly named champion copy
                                         (passed directly to exp3)
  Results/exp1_raw_cycles.csv          — per-cycle raw numbers
  Results/exp1_statistical_report.txt  — full stat report
  Figures/Figure_Exp1_4Arch_Ablation.png

Run from TF_Me project root:
    python exp1_statistical_ablation.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, gc, csv, time, shutil

# =====================================================================
# CONFIGURATION
# =====================================================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR   = os.path.join("casting_data", "train")
TEST_DIR    = os.path.join("casting_data", "test")
MODEL_DIR   = "Models"
FIG_DIR     = "Figures"
RESULT_DIR  = "Results"

N_CYCLES    = 30
BATCH_SIZE  = 32
EPOCHS      = 15
N_BOOTSTRAP = 2000
SEED        = 42

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ARCHITECTURES = ["ResNet-18", "MobileNet-V3", "EfficientNet-B0", "ConvNeXt-Tiny"]

# Palette — MobileNet-V3 now highlighted in gold to signal champion status
PALETTE = {
    "ResNet-18":       "#2E5C8A",
    "MobileNet-V3":    "#C47D00",   # amber — the deployment champion
    "EfficientNet-B0": "#3A7D44",
    "ConvNeXt-Tiny":   "#7B4EA0",
}


# =====================================================================
# DATA PIPELINE
# =====================================================================
def get_dataloaders():
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=tf)

    counts  = np.bincount(train_ds.targets)
    weights = 1.0 / counts
    sw      = [weights[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                           sampler=sampler, num_workers=4, pin_memory=True)
    te_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=4, pin_memory=True)
    return tr_loader, te_loader


# =====================================================================
# MODEL INIT
# =====================================================================
def init_model(name, num_classes=2):
    if name == "ResNet-18":
        m = models.resnet18(weights="DEFAULT")
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "MobileNet-V3":
        m = models.mobilenet_v3_large(weights="DEFAULT")
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "EfficientNet-B0":
        m = models.efficientnet_b0(weights="DEFAULT")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "ConvNeXt-Tiny":
        m = models.convnext_tiny(weights="DEFAULT")
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    return m.to(DEVICE)


# =====================================================================
# STATISTICS HELPERS
# =====================================================================
def t_ci(data, c=0.95):
    a  = np.array(data, float)
    m  = np.mean(a)
    h  = stats.sem(a) * stats.t.ppf((1+c)/2, len(a)-1)
    return m, m-h, m+h

def bootstrap_ci(data, n=N_BOOTSTRAP, c=0.95):
    rng   = np.random.default_rng(SEED)
    a     = np.array(data, float)
    means = [rng.choice(a, size=len(a), replace=True).mean() for _ in range(n)]
    lo    = np.percentile(means, (1-c)/2*100)
    hi    = np.percentile(means, (1+c)/2*100)
    return np.mean(a), lo, hi

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    sd = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 +
                  (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
    return (np.mean(a) - np.mean(b)) / sd if sd > 0 else 0.0

def deployment_score(acc_list, recall_list, f1_list):
    """
    Composite deployment score (0–100).
    Weights: mean_acc 0.25 + mean_recall 0.35 + mean_f1 0.25 + (1 - cv_acc)*0.15
    Recall is weighted highest because a missed defect is a safety failure.
    CV (coefficient of variation) penalises unstable architectures.
    """
    ma  = np.mean(acc_list)
    mr  = np.mean(recall_list)
    mf  = np.mean(f1_list)
    cv  = (np.std(acc_list) / ma * 100) if ma > 0 else 100
    cv_norm = max(0, 1 - cv / 100)       # normalise to [0,1], lower CV = better
    return 0.25*ma + 0.35*mr + 0.25*mf + 0.15*cv_norm*100


# =====================================================================
# FIGURE GENERATION
# =====================================================================
def generate_figure(results, champion_name):
    rows = []
    for arch in ARCHITECTURES:
        for v in results[arch]["acc"]:
            rows.append({"Architecture": arch, "Metric": "Accuracy (%)", "Value": v})
        for v in results[arch]["recall"]:
            rows.append({"Architecture": arch, "Metric": "Recall (%)", "Value": v})
        for v in results[arch]["f1"]:
            rows.append({"Architecture": arch, "Metric": "F1 Score (%)", "Value": v})
    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f"4-Architecture Ablation: Accuracy / Recall / F1  (n={N_CYCLES} cycles)\n"
        f"Deployment Champion: {champion_name}  —  COTS RTX 5070",
        fontsize=13, fontweight="bold"
    )

    for ax, metric in zip(axes, ["Accuracy (%)", "Recall (%)", "F1 Score (%)"]):
        sub = df[df["Metric"] == metric]
        edge_colors = ["gold" if a == champion_name else "white" for a in ARCHITECTURES]
        sns.boxplot(data=sub, x="Architecture", y="Value", ax=ax,
                    palette=PALETTE, width=0.5, linewidth=1.5,
                    boxprops=dict(alpha=0.45))
        sns.swarmplot(data=sub, x="Architecture", y="Value", ax=ax,
                      palette=PALETTE, size=7, alpha=0.85, marker="D",
                      linewidth=0.8)

        ax.set_title(metric, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(metric, fontweight="bold", fontsize=11)
        ax.set_xlabel("")

        # Mean ± SD annotation
        for i, arch in enumerate(ARCHITECTURES):
            vals = sub[sub["Architecture"] == arch]["Value"].values
            m, sd = vals.mean(), vals.std()
            label = f"μ={m:.2f}\n±{sd:.2f}"
            if arch == champion_name:
                label += "\n★"
            ax.text(i, ax.get_ylim()[1] * 0.98, label,
                    ha="center", va="top", fontsize=8,
                    color=PALETTE[arch], fontweight="bold")

        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "Figure_Exp1_4Arch_Statistical_Ablation.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure saved] {out}")


# =====================================================================
# MAIN ABLATION LOOP
# =====================================================================
def run_ablation():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 65)
    print(f"  exp1 — {N_CYCLES}-Cycle 4-Architecture Ablation  |  {DEVICE}")
    print(f"  Epochs={EPOCHS}  Batch={BATCH_SIZE}  Bootstrap={N_BOOTSTRAP}")
    print("=" * 65 + "\n")

    results   = {n: {"acc": [], "recall": [], "f1": []} for n in ARCHITECTURES}
    best_acc  = {n: 0.0 for n in ARCHITECTURES}
    criterion = nn.CrossEntropyLoss()

    csv_path     = os.path.join(RESULT_DIR, "exp1_raw_cycles.csv")
    completed    = set()   # (cycle, arch) pairs already on disk

    # Load any previously completed cycles so we never re-run them
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="") as existing:
            reader = csv.DictReader(existing)
            for row in reader:
                c, a = int(row["cycle"]), row["architecture"]
                completed.add((c, a))
                results[a]["acc"].append(float(row["accuracy"]))
                results[a]["recall"].append(float(row["recall"]))
                results[a]["f1"].append(float(row["f1"]))
        print(f"[Resume] Loaded {len(completed)} completed cycle-arch entries from CSV.")

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    cf = open(csv_path, "a", newline="")
    cw = csv.writer(cf)
    if write_header:
        cw.writerow(["cycle", "architecture", "accuracy", "precision",
                     "recall", "f1", "train_s"])

    tr_loader, te_loader = get_dataloaders()
    t_wall = time.time()

    for cycle in range(1, N_CYCLES + 1):
        print(f"\n--- Cycle {cycle}/{N_CYCLES}  "
              f"(wall {(time.time()-t_wall)/3600:.2f}h) ---")

        for name in ARCHITECTURES:
            if (cycle, name) in completed:
                print(f"  {name}... [skipped — already in CSV]")
                continue

            t0 = time.time()
            print(f"  {name}...", end="", flush=True)

            model = init_model(name)
            opt   = optim.Adam(model.parameters(), lr=0.001)

            # Training
            model.train()
            for _ in range(EPOCHS):
                for X, y in tr_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    loss = criterion(model(X), y)
                    loss.backward()
                    opt.step()

            # Evaluation
            model.eval()
            tp = fp = fn = tn = 0
            with torch.no_grad():
                for X, y in te_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    _, p = torch.max(model(X), 1)
                    tp += ((p==1)&(y==1)).sum().item()
                    fp += ((p==1)&(y==0)).sum().item()
                    fn += ((p==0)&(y==1)).sum().item()
                    tn += ((p==0)&(y==0)).sum().item()

            N   = tp+tn+fp+fn
            acc = (tp+tn)/N*100 if N>0 else 0
            pr  = tp/(tp+fp)*100 if (tp+fp)>0 else 0
            rc  = tp/(tp+fn)*100 if (tp+fn)>0 else 0
            f1  = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0
            ts  = time.time()-t0

            results[name]["acc"].append(acc)
            results[name]["recall"].append(rc)
            results[name]["f1"].append(f1)

            # Save best model
            if acc > best_acc[name]:
                best_acc[name] = acc
                safe = name.replace("-","").replace(" ","")
                torch.save(model.state_dict(),
                           os.path.join(MODEL_DIR, f"Best_{safe}_Exp1.pth"))

            cw.writerow([cycle, name, f"{acc:.4f}", f"{pr:.4f}",
                         f"{rc:.4f}", f"{f1:.4f}", f"{ts:.1f}"])
            cf.flush()

            print(f" Acc={acc:.2f}%  Rc={rc:.2f}%  F1={f1:.2f}%  ({ts:.0f}s)")

            del model, opt, loss
            torch.cuda.empty_cache()
            gc.collect()

    cf.close()

    # =====================================================================
    # IDENTIFY DEPLOYMENT CHAMPION
    # =====================================================================
    scores = {}
    for name in ARCHITECTURES:
        scores[name] = deployment_score(
            results[name]["acc"],
            results[name]["recall"],
            results[name]["f1"]
        )

    champion = max(scores, key=scores.get)
    print(f"\n{'='*65}")
    print(f"  DEPLOYMENT CHAMPION: {champion}  (composite score {scores[champion]:.2f})")
    for n in ARCHITECTURES:
        marker = " ★" if n == champion else ""
        print(f"    {n:<20}  score={scores[n]:.2f}{marker}")
    print(f"{'='*65}")

    # Copy champion weights to an explicit named file for exp3
    champ_safe = champion.replace("-","").replace(" ","")
    src = os.path.join(MODEL_DIR, f"Best_{champ_safe}_Exp1.pth")
    dst = os.path.join(MODEL_DIR, "Champion_MobileNetV3_Exp1.pth")
    if champion == "MobileNet-V3":
        shutil.copy2(src, dst)
        print(f"\n[Champion weights] Saved to {dst}")
    else:
        # If empirical results surprise us, save whatever won
        champ_dst = os.path.join(MODEL_DIR, f"Champion_{champ_safe}_Exp1.pth")
        shutil.copy2(src, champ_dst)
        print(f"\n[Champion weights] Saved to {champ_dst}")
        print(f"  NOTE: Champion was {champion}, not MobileNet-V3.")
        print(f"  Update CHAMPION_MODEL path in exp3 accordingly.")

    # =====================================================================
    # STATISTICAL REPORT
    # =====================================================================
    lines = ["=" * 70,
             f"  exp1 STATISTICAL REPORT  (n={N_CYCLES} cycles)",
             "=" * 70]

    for name in ARCHITECTURES:
        lines.append(f"\n{'─'*50}\n  Architecture: {name}"
                     + (" ★ DEPLOYMENT CHAMPION" if name == champion else ""))
        for metric, key in [("Accuracy", "acc"), ("Recall", "recall"), ("F1", "f1")]:
            d = results[name][key]
            m_t, lo_t, hi_t = t_ci(d)
            m_b, lo_b, hi_b = bootstrap_ci(d)
            lines.append(
                f"  {metric}: mean={m_t:.4f}%  SD={np.std(d):.4f}%\n"
                f"    95% t-CI       : [{lo_t:.4f}, {hi_t:.4f}]\n"
                f"    95% bootstrap  : [{lo_b:.4f}, {hi_b:.4f}]"
            )
        lines.append(f"  Composite deployment score: {scores[name]:.4f}")

    lines += ["\n" + "─"*70,
              f"  PAIRWISE: {champion} (DEPLOYMENT CHAMPION) vs all others",
              "─"*70]

    champ_acc = results[champion]["acc"]
    for name in [n for n in ARCHITECTURES if n != champion]:
        other = results[name]["acc"]
        t_s, p_w  = stats.ttest_ind(champ_acc, other, equal_var=False)
        u_s, p_mw = stats.mannwhitneyu(champ_acc, other, alternative="two-sided")
        d         = cohens_d(champ_acc, other)
        mag       = ("negligible" if abs(d)<0.2 else "small" if abs(d)<0.5
                     else "medium" if abs(d)<0.8 else "large")
        lines.append(
            f"\n  {champion} vs {name}:\n"
            f"    Welch t = {t_s:.4f},  p = {p_w:.6f}  "
            f"{'*SIG*' if p_w<0.05 else '(ns)'}\n"
            f"    Mann-Whitney U = {u_s:.0f},  p = {p_mw:.6f}  "
            f"{'*SIG*' if p_mw<0.05 else '(ns)'}\n"
            f"    Cohen's d = {d:.4f}  ({mag})"
        )

    lines += ["\n" + "─"*70, "  VARIANCE RATIOS (stability metric)", "─"*70]
    champ_var = np.var(champ_acc)
    for name in [n for n in ARCHITECTURES if n != champion]:
        ratio = np.var(results[name]["acc"]) / champ_var if champ_var > 0 else float("inf")
        lines.append(f"  Var({name}) / Var({champion}) = {ratio:.2f}:1")

    report = "\n".join(lines)
    print("\n" + report)
    rp = os.path.join(RESULT_DIR, "exp1_statistical_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Report saved] {rp}")
    print(f"[CSV    saved] {csv_path}")

    generate_figure(results, champion)

    print(f"\n[Done] Total wall time: {(time.time()-t_wall)/3600:.2f}h")


if __name__ == "__main__":
    run_ablation()
