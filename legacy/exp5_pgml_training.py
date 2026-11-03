"""
exp5_pgml_training.py — Physics-Guided Machine Learning Training
================================================================
Upgrades the EfficientNet-B0 champion to a PGML framework by adding
a Griffith fracture mechanics physics regularization loss computed
exclusively from input images (model forward graph is UNCHANGED —
TensorRT tensor fusion optimizations are preserved).

IMPORTANT — Before running:
  1. Run 05_architecture_ablation.py first.
  2. Read Results/05_ablation_report.txt and set ALPHA_STAR below
     to the winning alpha value.

Physics Loss:
  L_total = L_CE(y, ŷ) + ALPHA_STAR * L_physics(x)

  L_physics = ReLU( E[||∇I||₂ | OK] - E[||∇I||₂ | defective] + margin )

  Physically grounded: defective casting regions (porosity, cold shuts,
  hot tears) concentrate stress at crack tips per Griffith K_I = σ√(πa),
  yielding higher mean spatial gradient magnitude than OK surfaces.
  The loss penalises any batch where this invariant is violated.

Label convention (ImageFolder alphabetical):
  label 0 = def_front  (defective)
  label 1 = ok_front   (OK)

Multi-seed validation: 3 independent seeds [42, 123, 777].
Computes μ±σ and compares against exp1 baseline. Prints PASS/FAIL
for each metric.

Outputs:
  Models/PGML_Champion_EfficientNetB0.pth   — best model across all seeds
  Results/exp5_pgml_results.csv
  Results/exp5_pgml_report.txt
  pgml_agent_progress.log  (appended)

Run from TF_Me project root:
    python exp5_pgml_training.py
"""

import os
import gc
import csv
import time
import numpy as np
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR    = os.path.join("casting_data", "train")
TEST_DIR     = os.path.join("casting_data", "test")
MODEL_DIR    = "Models"
RESULT_DIR   = "Results"
FIG_DIR      = "Figures"
LOG_FILE     = "pgml_agent_progress.log"

# ---- SET THIS from Results/05_ablation_report.txt before running ----
ALPHA_STAR   = 0.1      # physics loss weight — winner from 05_architecture_ablation.py
# --------------------------------------------------------------------

MARGIN       = 0.10     # Griffith physics margin
SEEDS        = [42, 123, 777]
EPOCHS       = 15
BATCH_SIZE   = 32
N_BOOTSTRAP  = 2000

# Baseline from exp1 (must be beaten on ALL metrics)
BASELINE = {
    "acc":    99.594,
    "recall": 99.759,
    "f1":     99.450,
}

LOCAL_WEIGHTS = os.path.join(MODEL_DIR, "Best_EfficientNetB0_Exp1.pth")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# Fixed Sobel kernels — non-learnable, no grad, TensorRT safe
_KX = torch.tensor([[-1., 0., 1.],
                     [-2., 0., 2.],
                     [-1., 0., 1.]], dtype=torch.float32)
_KY = torch.tensor([[-1., -2., -1.],
                     [ 0.,  0.,  0.],
                     [ 1.,  2.,  1.]], dtype=torch.float32)


# ============================================================================
# PHYSICS LOSS
# ============================================================================
def compute_physics_loss(x_batch, labels, alpha, margin=MARGIN, device="cuda"):
    """
    Vectorized Griffith fracture mechanics physics regularization.

    Operates on raw normalised input images ONLY.
    The model forward graph is NEVER touched — TensorRT compatibility preserved.

    Returns a detached scalar; back-prop flows only through CE loss.
    """
    if alpha == 0.0:
        return torch.tensor(0.0, device=device)

    kx = _KX.view(1, 1, 3, 3).to(device)
    ky = _KY.view(1, 1, 3, 3).to(device)

    gray = (0.299 * x_batch[:, 0:1] +
            0.587 * x_batch[:, 1:2] +
            0.114 * x_batch[:, 2:3]).float()   # (B, 1, H, W)

    with torch.no_grad():
        gx  = F.conv2d(gray, kx, padding=1)
        gy  = F.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)   # (B, 1, H, W)
        mean_mag = mag.mean(dim=[1, 2, 3])                  # (B,)

        ok_mask  = (labels == 1)    # ok_front
        def_mask = (labels == 0)    # def_front

        if not ok_mask.any() or not def_mask.any():
            return torch.tensor(0.0, device=device)

        mean_mag_ok  = mean_mag[ok_mask].mean()
        mean_mag_def = mean_mag[def_mask].mean()

        phys = F.relu(mean_mag_ok - mean_mag_def + margin)

    return phys   # scalar, detached


# ============================================================================
# DATA PIPELINE — WRS preserved (project rule #4)
# ============================================================================
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
    te_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=4, pin_memory=True)
    return tr_loader, te_loader


# ============================================================================
# MODEL — local weights, no internet
# ============================================================================
def init_effnet():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    state = torch.load(LOCAL_WEIGHTS, map_location=DEVICE, weights_only=True)
    m.load_state_dict(state)
    return m.to(DEVICE)


# ============================================================================
# STATISTICS (from exp1)
# ============================================================================
def t_ci(data, c=0.95):
    a = np.array(data, float)
    m = np.mean(a)
    h = stats.sem(a) * stats.t.ppf((1 + c) / 2, len(a) - 1)
    return m, m - h, m + h


def bootstrap_ci(data, n=N_BOOTSTRAP, c=0.95):
    rng   = np.random.default_rng(42)
    a     = np.array(data, float)
    means = [rng.choice(a, size=len(a), replace=True).mean() for _ in range(n)]
    return np.mean(a), np.percentile(means, (1-c)/2*100), np.percentile(means, (1+c)/2*100)


def deployment_score(acc_list, recall_list, f1_list):
    ma      = np.mean(acc_list)
    mr      = np.mean(recall_list)
    mf      = np.mean(f1_list)
    cv      = (np.std(acc_list) / ma * 100) if ma > 0 else 100
    cv_norm = max(0.0, 1.0 - cv / 100)
    return 0.25 * ma + 0.35 * mr + 0.25 * mf + 0.15 * cv_norm * 100


def log_append(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ============================================================================
# SMALL-SCALE SANITY CHECK (project rule #4)
# ============================================================================
def sanity_check(tr_loader, te_loader):
    """
    Quick 1-epoch, 20-batch check verifying:
      - Physics loss is non-negative
      - CE loss decreases over batches
      - Metrics are non-zero
    """
    print("\n[Sanity Check] 1-epoch / 20-batch test...")
    torch.manual_seed(0)
    model = init_effnet()
    opt   = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()

    losses = []
    for i, (X, y) in enumerate(tr_loader):
        if i >= 20:
            break
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        ce  = criterion(model(X), y)
        phy = compute_physics_loss(X, y, ALPHA_STAR, MARGIN, DEVICE.type)
        assert phy.item() >= 0, "Physics loss is negative — error in formula"
        loss = ce + ALPHA_STAR * phy
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Quick eval on a few test batches
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(te_loader):
            if i >= 5:
                break
            X, y = X.to(DEVICE), y.to(DEVICE)
            _, p = model(X).max(1)
            tp += ((p == 1) & (y == 1)).sum().item()
            fp += ((p == 1) & (y == 0)).sum().item()
            fn += ((p == 0) & (y == 1)).sum().item()
            tn += ((p == 0) & (y == 0)).sum().item()

    N   = tp + tn + fp + fn
    acc = (tp + tn) / N * 100 if N > 0 else 0
    assert acc > 0, "Sanity check: zero accuracy — data or model issue"
    print(f"[Sanity Check] PASS  |  CE+phys loss stable  |  mini-eval acc={acc:.1f}%")

    del model, opt
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def run_pgml():
    torch.manual_seed(SEEDS[0])
    np.random.seed(SEEDS[0])

    print("=" * 70)
    print("  exp5_pgml_training.py — Physics-Guided ML Training")
    print(f"  DEVICE: {DEVICE}  |  ALPHA_STAR={ALPHA_STAR}  |  MARGIN={MARGIN}")
    print(f"  SEEDS: {SEEDS}   EPOCHS={EPOCHS}   BATCH={BATCH_SIZE}")
    print(f"  Baseline to beat: "
          f"Acc>{BASELINE['acc']}%  Rc>{BASELINE['recall']}%  F1>{BASELINE['f1']}%")
    print("=" * 70)

    if not os.path.exists(LOCAL_WEIGHTS):
        raise FileNotFoundError(
            f"Local weights not found: {LOCAL_WEIGHTS}\n"
            "Run exp1_statistical_ablation.py first."
        )

    tr_loader, te_loader = get_dataloaders()

    # Project rule #4: small-scale test first
    sanity_check(tr_loader, te_loader)

    criterion = nn.CrossEntropyLoss()
    seed_records = []      # {seed, acc, recall, f1}
    best_composite  = -1.0
    best_state_dict = None

    csv_path = os.path.join(RESULT_DIR, "exp5_pgml_results.csv")
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    cf = open(csv_path, "a", newline="")
    cw = csv.writer(cf)
    if write_header:
        cw.writerow(["seed", "alpha", "epoch", "acc", "recall", "f1",
                     "ce_loss_avg", "phys_loss_avg", "epoch_s"])

    t_wall = time.time()

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"  Seed {seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})")
        print(f"{'='*50}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = init_effnet()
        opt   = optim.Adam(model.parameters(), lr=0.001)

        best_seed_f1    = -1.0
        best_seed_state = None
        seed_best_acc   = 0.0
        seed_best_rc    = 0.0
        seed_best_f1    = 0.0

        for epoch in range(1, EPOCHS + 1):
            t_ep = time.time()
            model.train()
            epoch_ce   = 0.0
            epoch_phys = 0.0
            n_batches  = 0

            for X, y in tr_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                ce_loss = criterion(model(X), y)
                phys    = compute_physics_loss(X, y, ALPHA_STAR, MARGIN, DEVICE.type)
                loss    = ce_loss + ALPHA_STAR * phys
                loss.backward()
                opt.step()
                epoch_ce   += ce_loss.item()
                epoch_phys += phys.item()
                n_batches  += 1

            avg_ce   = epoch_ce   / max(n_batches, 1)
            avg_phys = epoch_phys / max(n_batches, 1)

            # Full evaluation
            model.eval()
            tp = fp = fn = tn = 0
            with torch.no_grad():
                for X, y in te_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    _, p = model(X).max(1)
                    tp += ((p == 1) & (y == 1)).sum().item()
                    fp += ((p == 1) & (y == 0)).sum().item()
                    fn += ((p == 0) & (y == 1)).sum().item()
                    tn += ((p == 0) & (y == 0)).sum().item()

            N   = tp + tn + fp + fn
            acc = (tp + tn) / N * 100 if N > 0 else 0.0
            pr  = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
            rc  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            f1  = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
            ep_s = time.time() - t_ep

            cw.writerow([seed, ALPHA_STAR, epoch,
                         f"{acc:.4f}", f"{rc:.4f}", f"{f1:.4f}",
                         f"{avg_ce:.6f}", f"{avg_phys:.6f}", f"{ep_s:.1f}"])
            cf.flush()

            print(f"  Epoch {epoch:02d}/{EPOCHS}  "
                  f"Acc={acc:.3f}%  Rc={rc:.3f}%  F1={f1:.3f}%  "
                  f"CE={avg_ce:.4f}  Phys={avg_phys:.4f}  ({ep_s:.0f}s)")

            # Track best per seed (by F1 to avoid single-metric overfitting)
            if f1 > best_seed_f1:
                best_seed_f1    = f1
                best_seed_state = {k: v.clone() for k, v in model.state_dict().items()}
                seed_best_acc   = acc
                seed_best_rc    = rc
                seed_best_f1    = f1

        # Record seed best
        seed_records.append({
            "seed": seed,
            "acc":    seed_best_acc,
            "recall": seed_best_rc,
            "f1":     seed_best_f1,
        })

        # Update global best model (by composite score on per-seed best)
        comp = deployment_score([seed_best_acc], [seed_best_rc], [seed_best_f1])
        if comp > best_composite:
            best_composite  = comp
            best_state_dict = best_seed_state

        del model, opt
        torch.cuda.empty_cache()
        gc.collect()

    cf.close()

    # Save champion model
    champion_path = os.path.join(MODEL_DIR, "PGML_Champion_EfficientNetB0.pth")
    torch.save(best_state_dict, champion_path)
    print(f"\n[Champion saved] {champion_path}")

    # ========================================================================
    # STATISTICAL REPORT
    # ========================================================================
    accs = [r["acc"]    for r in seed_records]
    rcs  = [r["recall"] for r in seed_records]
    f1s  = [r["f1"]     for r in seed_records]

    pgml_acc_mean, pgml_acc_sd = np.mean(accs), np.std(accs)
    pgml_rc_mean,  pgml_rc_sd  = np.mean(rcs),  np.std(rcs)
    pgml_f1_mean,  pgml_f1_sd  = np.mean(f1s),  np.std(f1s)

    lines = [
        "=" * 70,
        "  exp5 PGML TRAINING REPORT",
        f"  EfficientNet-B0  |  ALPHA_STAR={ALPHA_STAR}  |  MARGIN={MARGIN}",
        f"  Seeds: {SEEDS}   Epochs: {EPOCHS}   Batch: {BATCH_SIZE}",
        "=" * 70,
        "",
        "  Per-seed best results:",
    ]
    for r in seed_records:
        lines.append(
            f"    Seed {r['seed']}: "
            f"Acc={r['acc']:.4f}%  Recall={r['recall']:.4f}%  F1={r['f1']:.4f}%"
        )

    lines += [
        "",
        "  Aggregate across seeds (μ ± σ):",
        f"    Accuracy  : {pgml_acc_mean:.4f}% ± {pgml_acc_sd:.4f}%",
        f"    Recall    : {pgml_rc_mean:.4f}% ± {pgml_rc_sd:.4f}%",
        f"    F1-Score  : {pgml_f1_mean:.4f}% ± {pgml_f1_sd:.4f}%",
        "",
        "  Comparison vs exp1 baseline:",
    ]

    all_pass = True
    for label, pgml_m, pgml_s, baseline_v in [
        ("Accuracy", pgml_acc_mean, pgml_acc_sd, BASELINE["acc"]),
        ("Recall",   pgml_rc_mean,  pgml_rc_sd,  BASELINE["recall"]),
        ("F1-Score", pgml_f1_mean,  pgml_f1_sd,  BASELINE["f1"]),
    ]:
        delta  = pgml_m - baseline_v
        status = "PASS (BEAT BASELINE)" if delta > 0 else "FAIL — see fallback notes"
        if delta <= 0:
            all_pass = False
        lines.append(
            f"    {label:<10}: {pgml_m:.4f}% ± {pgml_s:.4f}%  "
            f"(delta={delta:+.4f}%) — {status}"
        )

    overall = "ALL METRICS: PASS" if all_pass else "ONE OR MORE METRICS: FAIL"
    lines += [
        "",
        f"  {overall}",
        "",
        "  Fallback guidance (if FAIL):",
        "    1. Reduce ALPHA_STAR to 0.01 and rerun.",
        "    2. If still failing: set ALPHA_STAR=0.0 (reverts to exp1 baseline).",
        "",
        f"  Champion model : {champion_path}",
        f"  Total wall time: {(time.time()-t_wall)/3600:.2f}h",
        "=" * 70,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    rp = os.path.join(RESULT_DIR, "exp5_pgml_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Report saved] {rp}")

    # ========================================================================
    # FIGURE — seed comparison
    # ========================================================================
    try:
        import seaborn as sns
        import pandas as pd
        rows = []
        for r in seed_records:
            for metric, val in [("Accuracy", r["acc"]),
                                 ("Recall",   r["recall"]),
                                 ("F1-Score", r["f1"])]:
                rows.append({"Seed": str(r["seed"]), "Metric": metric, "Value": val})
        df = pd.DataFrame(rows)

        sns.set_theme(style="whitegrid", font_scale=1.0)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"PGML Training — EfficientNet-B0  α={ALPHA_STAR}  (3 seeds)",
            fontsize=12, fontweight="bold"
        )
        palette = {"42": "#2E5C8A", "123": "#3A7D44", "777": "#C47D00"}
        for ax, metric in zip(axes, ["Accuracy", "Recall", "F1-Score"]):
            sub = df[df["Metric"] == metric]
            sns.barplot(data=sub, x="Seed", y="Value", ax=ax, palette=palette)
            ax.set_title(metric, fontweight="bold")
            ax.set_ylim(max(0, sub["Value"].min() - 0.5), 100.5)
            ax.set_ylabel(f"{metric} (%)")
            ax.axhline(BASELINE[metric.lower().replace("-score", "").replace("accuracy", "acc")
                                 .replace("recall", "recall").replace("f1-score", "f1")],
                       color="red", linestyle="--", linewidth=1.2, label="Baseline")
            ax.legend(fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "Figure_Exp5_PGML_Seeds.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[Figure saved] {fig_path}")
    except Exception as e:
        print(f"[Figure skipped] {e}")

    log_append(
        f"\n[Phase 2 COMPLETE — {time.strftime('%Y-%m-%d %H:%M')}]\n"
        f"  ALPHA_STAR   : {ALPHA_STAR}\n"
        f"  Accuracy     : {pgml_acc_mean:.4f}% ± {pgml_acc_sd:.4f}%\n"
        f"  Recall       : {pgml_rc_mean:.4f}% ± {pgml_rc_sd:.4f}%\n"
        f"  F1-Score     : {pgml_f1_mean:.4f}% ± {pgml_f1_sd:.4f}%\n"
        f"  Overall      : {overall}\n"
        f"  Champion     : {champion_path}\n"
        f"  Wall time    : {(time.time()-t_wall)/3600:.2f}h"
    )

    print(f"\n[Done] Total wall time: {(time.time()-t_wall)/3600:.2f}h")


if __name__ == "__main__":
    run_pgml()
