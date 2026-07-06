"""
exp1_statistical_ablation_v2.py — 4-Architecture Ablation, leakage-free + recall-fixed
=========================================================================================
Supersedes exp1_statistical_ablation.py. Two independent bugs fixed together so the
expensive 30-cycle x 4-architecture training only has to run once (Items 4 + 7a):

  ITEM 4 — TEST-SET LEAKAGE IN ARCHITECTURE SELECTION.
    The original script had no validation split: every one of the 120 training
    runs (30 cycles x 4 architectures) was evaluated directly against
    casting_data/test, and those numbers fed deployment_score(), which picks the
    champion. The same test set was then reused for the paper's headline
    "deployed model" accuracy. Fix: a single stratified 15% validation split is
    carved from the training partition ONCE (seed=SEED, per-class stratified,
    matching exp11's make_loaders pattern) and used for all 30 cycles x 4
    architectures, so cycle-to-cycle variance still reflects only init/training
    stochasticity, not data-split variance (preserves the "identical data across
    cycles" premise behind the paper's variance-ratio claims). Every cycle's
    architecture is evaluated on this VALIDATION split; those numbers (not test)
    feed deployment_score() and Table 2. The test set is touched exactly once,
    at the end, and ONLY for the architecture validation already selected: every
    cycle checkpoint is saved during the main loop, and after the champion
    architecture is picked, its 30 saved checkpoints (and only its) are each
    evaluated on the test set once, producing an n=30 confirmatory statistic
    that never influenced the selection decision.

  ITEM 7A — RECALL POSITIVE-CLASS BUG.
    casting_data/train/ sorts def_front (3,758 imgs) before ok_front (2,875
    imgs) alphabetically, so ImageFolder assigns def_front=0, ok_front=1. The
    original evaluate() hardcoded p==1 & y==1 as "true positive", making
    "recall"/"precision"/"F1" the OK/nominal class's statistics, not the
    defective class's catch rate -- backwards relative to every "a missed
    defect is a safety failure" sentence in the paper, and backwards relative
    to the composite score's own justification for weighting recall at 0.35.
    Fixed here: positive class = defective (class 0).

Outputs (kept separate from the v1 run so both are preserved):
  Results/exp1v2_raw_cycles.csv       -- per-cycle validation numbers (all 4 archs)
  Results/exp1v2_test_confirmation.csv-- n=30 test-set numbers, CHAMPION ARCH ONLY
  Results/exp1v2_statistical_report.txt
  Figures/Figure_Exp1v2_4Arch_Ablation.png
  Models/exp1v2_cycle_ckpts/{arch}_cycle{N}.pth  -- temp, deleted after confirmation
                                                     pass except for the champion's
  Models/Champion_v2_{Arch}_Exp1.pth

Run from TF_Me project root:
    python exp1_statistical_ablation_v2.py
    python exp1_statistical_ablation_v2.py --smoke   # 2 cycles, 2 epochs, all 4 archs
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
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
SMOKE       = "--smoke" in sys.argv

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR   = os.path.join("casting_data", "train")
TEST_DIR    = os.path.join("casting_data", "test")
MODEL_DIR   = "Models"
CKPT_DIR    = os.path.join(MODEL_DIR, "exp1v2_cycle_ckpts")
FIG_DIR     = "Figures"
RESULT_DIR  = "Results"

N_CYCLES     = 30 if not SMOKE else 2
BATCH_SIZE   = 32
EPOCHS       = 15 if not SMOKE else 2
N_BOOTSTRAP  = 2000
SEED         = 42
VAL_FRACTION = 0.15

SUFFIX = "_smoke" if SMOKE else ""

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ARCHITECTURES = ["ResNet-18", "MobileNet-V3", "EfficientNet-B0", "ConvNeXt-Tiny"]

PALETTE = {
    "ResNet-18":       "#2E5C8A",
    "MobileNet-V3":    "#C47D00",
    "EfficientNet-B0": "#3A7D44",
    "ConvNeXt-Tiny":   "#7B4EA0",
}

# def_front sorts before ok_front alphabetically -> ImageFolder class 0/1
DEF_CLASS, OK_CLASS = 0, 1


# =====================================================================
# DATA PIPELINE — single stratified validation split, carved once
# =====================================================================
def get_dataloaders():
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    full_train = datasets.ImageFolder(TRAIN_DIR, transform=tf)
    test_ds    = datasets.ImageFolder(TEST_DIR,  transform=tf)

    # Stratified 15% validation split, carved ONCE (same split for all 30
    # cycles x 4 architectures, so cycle-to-cycle variance reflects only
    # init/training stochasticity, matching the paper's "identical data"
    # premise for the variance-ratio claims).
    targets = np.array(full_train.targets)
    rng = np.random.RandomState(SEED)
    val_idx, tr_idx = [], []
    for c in np.unique(targets):
        idx = rng.permutation(np.where(targets == c)[0])
        n_val = int(round(VAL_FRACTION * len(idx)))
        val_idx.extend(idx[:n_val])
        tr_idx.extend(idx[n_val:])
    tr_idx, val_idx = sorted(tr_idx), sorted(val_idx)

    tr_targets = targets[tr_idx]
    counts  = np.bincount(tr_targets)
    weights = 1.0 / counts
    sw      = [weights[t] for t in tr_targets]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    tr_loader  = DataLoader(Subset(full_train, tr_idx), batch_size=BATCH_SIZE,
                            sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
    te_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
    print(f"[Data] train={len(tr_idx)}  val={len(val_idx)} (held out from train, "
          f"used for selection)  test={len(test_ds)} (untouched until confirmation)")
    return tr_loader, val_loader, te_loader


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
# EVALUATION — positive class = DEFECTIVE (item 7a fix)
# =====================================================================
def evaluate(model, loader):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            _, p = torch.max(model(X), 1)
            tp += ((p == DEF_CLASS) & (y == DEF_CLASS)).sum().item()
            fp += ((p == DEF_CLASS) & (y == OK_CLASS)).sum().item()
            fn += ((p == OK_CLASS)  & (y == DEF_CLASS)).sum().item()
            tn += ((p == OK_CLASS)  & (y == OK_CLASS)).sum().item()
    N   = tp + tn + fp + fn
    acc = (tp + tn) / N * 100 if N > 0 else 0
    pr  = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rc  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0   # defect catch rate
    f1  = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
    return acc, pr, rc, f1


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
    """Composite deployment score (0-100), computed on VALIDATION statistics.
    Weights: mean_acc 0.25 + mean_recall 0.35 + mean_f1 0.25 + (1-cv_acc)*0.15
    """
    ma  = np.mean(acc_list)
    mr  = np.mean(recall_list)
    mf  = np.mean(f1_list)
    cv  = (np.std(acc_list) / ma * 100) if ma > 0 else 100
    cv_norm = max(0, 1 - cv / 100)
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
        f"4-Architecture Ablation v2 (validation-based, n={N_CYCLES} cycles)\n"
        f"Deployment Champion: {champion_name}  —  COTS RTX 5070",
        fontsize=13, fontweight="bold"
    )

    for ax, metric in zip(axes, ["Accuracy (%)", "Recall (%)", "F1 Score (%)"]):
        sub = df[df["Metric"] == metric]
        sns.boxplot(data=sub, x="Architecture", y="Value", ax=ax,
                    palette=PALETTE, width=0.5, linewidth=1.5,
                    boxprops=dict(alpha=0.45))
        sns.swarmplot(data=sub, x="Architecture", y="Value", ax=ax,
                      palette=PALETTE, size=7, alpha=0.85, marker="D",
                      linewidth=0.8)
        ax.set_title(metric, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(metric, fontweight="bold", fontsize=11)
        ax.set_xlabel("")
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
    out = os.path.join(FIG_DIR, "Figure_Exp1v2_4Arch_Ablation.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure saved] {out}")


# =====================================================================
# MAIN ABLATION LOOP — evaluate on VALIDATION, save every cycle checkpoint
# =====================================================================
def run_ablation():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 65)
    print(f"  exp1v2 — {N_CYCLES}-Cycle 4-Architecture Ablation (leakage-free)  |  {DEVICE}")
    print(f"  Epochs={EPOCHS}  Batch={BATCH_SIZE}  Bootstrap={N_BOOTSTRAP}")
    print("=" * 65 + "\n")

    results   = {n: {"acc": [], "recall": [], "f1": []} for n in ARCHITECTURES}
    criterion = nn.CrossEntropyLoss()

    csv_path  = os.path.join(RESULT_DIR, f"exp1v2_raw_cycles{SUFFIX}.csv")
    completed = set()

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="") as existing:
            reader = csv.DictReader(existing)
            for row in reader:
                c, a = int(row["cycle"]), row["architecture"]
                completed.add((c, a))
                results[a]["acc"].append(float(row["val_accuracy"]))
                results[a]["recall"].append(float(row["val_recall"]))
                results[a]["f1"].append(float(row["val_f1"]))
        print(f"[Resume] Loaded {len(completed)} completed cycle-arch entries from CSV.")

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    cf = open(csv_path, "a", newline="")
    cw = csv.writer(cf)
    if write_header:
        cw.writerow(["cycle", "architecture", "val_accuracy", "val_precision",
                     "val_recall", "val_f1", "train_s"])

    tr_loader, val_loader, te_loader = get_dataloaders()
    t_wall = time.time()

    for cycle in range(1, N_CYCLES + 1):
        print(f"\n--- Cycle {cycle}/{N_CYCLES}  "
              f"(wall {(time.time()-t_wall)/3600:.2f}h) ---")

        for name in ARCHITECTURES:
            ckpt_path = os.path.join(CKPT_DIR, f"{name.replace('-','').replace(' ','')}_cycle{cycle}.pth")
            if (cycle, name) in completed and os.path.exists(ckpt_path):
                print(f"  {name}... [skipped — already in CSV]")
                continue

            t0 = time.time()
            print(f"  {name}...", end="", flush=True)

            model = init_model(name)
            opt   = optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for _ in range(EPOCHS):
                for X, y in tr_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    loss = criterion(model(X), y)
                    loss.backward()
                    opt.step()

            acc, pr, rc, f1 = evaluate(model, val_loader)
            ts = time.time() - t0

            results[name]["acc"].append(acc)
            results[name]["recall"].append(rc)
            results[name]["f1"].append(f1)

            # Save EVERY cycle's checkpoint (needed for the post-hoc,
            # selection-blind test confirmation of whichever architecture wins).
            torch.save(model.state_dict(), ckpt_path)

            cw.writerow([cycle, name, f"{acc:.4f}", f"{pr:.4f}",
                         f"{rc:.4f}", f"{f1:.4f}", f"{ts:.1f}"])
            cf.flush()

            print(f" ValAcc={acc:.2f}%  ValRc={rc:.2f}%  ValF1={f1:.2f}%  ({ts:.0f}s)")

            del model, opt, loss
            torch.cuda.empty_cache()
            gc.collect()

    cf.close()

    # =====================================================================
    # IDENTIFY DEPLOYMENT CHAMPION — on VALIDATION statistics only
    # =====================================================================
    scores = {}
    for name in ARCHITECTURES:
        scores[name] = deployment_score(
            results[name]["acc"], results[name]["recall"], results[name]["f1"])

    champion = max(scores, key=scores.get)
    print(f"\n{'='*65}")
    print(f"  DEPLOYMENT CHAMPION (validation-selected): {champion}  "
          f"(composite score {scores[champion]:.2f})")
    for n in ARCHITECTURES:
        marker = " ★" if n == champion else ""
        print(f"    {n:<20}  score={scores[n]:.2f}{marker}")
    print(f"{'='*65}")

    # =====================================================================
    # CONFIRMATORY TEST-SET PASS — champion architecture ONLY, test touched
    # exactly once per cycle checkpoint, never used in the decision above.
    # =====================================================================
    champ_safe = champion.replace("-", "").replace(" ", "")
    test_csv = os.path.join(RESULT_DIR, f"exp1v2_test_confirmation{SUFFIX}.csv")

    # Resume support: only reuse a prior confirmation CSV if it was written for
    # this same champion (an interrupted-then-resumed main loop always picks
    # the same champion once all 120 cycles are back in the CSV, but a stale
    # file from a different run shouldn't be silently mixed in).
    test_rows = []
    if os.path.exists(test_csv) and os.path.getsize(test_csv) > 0:
        with open(test_csv, newline="") as existing:
            for row in csv.DictReader(existing):
                if row["architecture"] == champion:
                    test_rows.append({"cycle": int(row["cycle"]), "architecture": champion,
                                       "test_accuracy": float(row["test_accuracy"]),
                                       "test_precision": float(row["test_precision"]),
                                       "test_recall": float(row["test_recall"]),
                                       "test_f1": float(row["test_f1"])})
    done_cycles = {r["cycle"] for r in test_rows}
    if done_cycles:
        print(f"[Resume] {len(done_cycles)} confirmation cycle(s) already evaluated for {champion}.")

    write_header = not done_cycles
    tf = open(test_csv, "a" if done_cycles else "w", newline="")
    tw = csv.writer(tf)
    if write_header:
        tw.writerow(["cycle", "architecture", "test_accuracy", "test_precision",
                     "test_recall", "test_f1"])

    print(f"\n[Confirmation] Evaluating {champion}'s {N_CYCLES} validation-selected "
          f"checkpoints on the untouched test set (decision already made above)...")
    for cycle in range(1, N_CYCLES + 1):
        if cycle in done_cycles:
            continue
        ckpt_path = os.path.join(CKPT_DIR, f"{champ_safe}_cycle{cycle}.pth")
        m = init_model(champion)
        m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        t_acc, t_pr, t_rc, t_f1 = evaluate(m, te_loader)
        row = {"cycle": cycle, "architecture": champion,
               "test_accuracy": t_acc, "test_precision": t_pr,
               "test_recall": t_rc, "test_f1": t_f1}
        test_rows.append(row)
        tw.writerow([cycle, champion, f"{t_acc:.4f}", f"{t_pr:.4f}", f"{t_rc:.4f}", f"{t_f1:.4f}"])
        tf.flush()
        del m
        torch.cuda.empty_cache()
    tf.close()
    test_rows.sort(key=lambda r: r["cycle"])
    t_acc_arr = np.array([r["test_accuracy"] for r in test_rows])
    t_rc_arr  = np.array([r["test_recall"]   for r in test_rows])
    t_f1_arr  = np.array([r["test_f1"]       for r in test_rows])
    print(f"[Confirmation] {champion} on held-out test (n={N_CYCLES}, never used for selection):")
    print(f"    Accuracy: {t_acc_arr.mean():.4f}% ± {t_acc_arr.std():.4f}%")
    print(f"    Recall  : {t_rc_arr.mean():.4f}% ± {t_rc_arr.std():.4f}%  (defect catch rate)")
    print(f"    F1      : {t_f1_arr.mean():.4f}% ± {t_f1_arr.std():.4f}%")

    # Keep only the champion's checkpoints; free disk from the other 3 archs.
    for name in ARCHITECTURES:
        if name == champion:
            continue
        safe = name.replace("-", "").replace(" ", "")
        for cycle in range(1, N_CYCLES + 1):
            p = os.path.join(CKPT_DIR, f"{safe}_cycle{cycle}.pth")
            if os.path.exists(p):
                os.remove(p)
    best_cycle = int(max(test_rows, key=lambda r: r["test_f1"])["cycle"])
    src = os.path.join(CKPT_DIR, f"{champ_safe}_cycle{best_cycle}.pth")
    dst = os.path.join(MODEL_DIR, f"Champion_v2_{champ_safe}_Exp1.pth")
    shutil.copy2(src, dst)
    print(f"\n[Champion weights] Best-test-F1 cycle ({best_cycle}) saved to {dst}")

    # =====================================================================
    # STATISTICAL REPORT
    # =====================================================================
    lines = ["=" * 70,
             f"  exp1v2 STATISTICAL REPORT  (n={N_CYCLES} cycles, validation-based selection)",
             "=" * 70]

    for name in ARCHITECTURES:
        lines.append(f"\n{'─'*50}\n  Architecture: {name}"
                     + (" ★ DEPLOYMENT CHAMPION" if name == champion else ""))
        for metric, key in [("Val Accuracy", "acc"), ("Val Recall (defect catch)", "recall"), ("Val F1", "f1")]:
            d = results[name][key]
            m_t, lo_t, hi_t = t_ci(d)
            m_b, lo_b, hi_b = bootstrap_ci(d)
            lines.append(
                f"  {metric}: mean={m_t:.4f}%  SD={np.std(d):.4f}%\n"
                f"    95% t-CI       : [{lo_t:.4f}, {hi_t:.4f}]\n"
                f"    95% bootstrap  : [{lo_b:.4f}, {hi_b:.4f}]"
            )
        lines.append(f"  Composite deployment score (on validation): {scores[name]:.4f}")

    lines += ["\n" + "─"*70,
              f"  CONFIRMATORY TEST-SET RESULT — {champion} only, n={N_CYCLES}",
              "  (test set never used to pick the architecture above)",
              "─"*70]
    lines.append(f"  Test Accuracy: {t_acc_arr.mean():.4f}% ± {t_acc_arr.std():.4f}%")
    lines.append(f"  Test Recall  : {t_rc_arr.mean():.4f}% ± {t_rc_arr.std():.4f}%  (defect catch rate)")
    lines.append(f"  Test F1      : {t_f1_arr.mean():.4f}% ± {t_f1_arr.std():.4f}%")

    lines += ["\n" + "─"*70,
              f"  PAIRWISE (on VALIDATION): {champion} (DEPLOYMENT CHAMPION) vs all others",
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

    lines += ["\n" + "─"*70, "  VARIANCE RATIOS on VALIDATION (stability metric)", "─"*70]
    champ_var = np.var(champ_acc)
    for name in [n for n in ARCHITECTURES if n != champion]:
        ratio = np.var(results[name]["acc"]) / champ_var if champ_var > 0 else float("inf")
        lines.append(f"  Var({name}) / Var({champion}) = {ratio:.2f}:1")

    report = "\n".join(lines)
    print("\n" + report)
    rp = os.path.join(RESULT_DIR, f"exp1v2_statistical_report{SUFFIX}.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Report saved] {rp}")
    print(f"[CSV    saved] {csv_path}")

    generate_figure(results, champion)

    print(f"\n[Done] Total wall time: {(time.time()-t_wall)/3600:.2f}h")


if __name__ == "__main__":
    run_ablation()
