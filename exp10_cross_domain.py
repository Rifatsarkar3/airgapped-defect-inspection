"""
exp10_cross_domain.py — Cross-Domain Fine-Tuning under a 15% Data Budget
=========================================================================
Fine-tunes the casting-pretrained champion EfficientNet-B0 on DAGM and
Kolektor (binary surface-defect benchmarks) with a frozen backbone and a
head-only cross-entropy objective, at 15% and 100% data budgets, with t-SNE
and Silhouette latent-space analysis. NEU-DET (native 6-class) is handled by
exp10_neudet_6class.py.

Each dataset is seeded independently (SEED=42), so results do not depend on
registry order. The casting-pretrained backbone is the Gate-1 champion
(Models/Gate1_Champion_EfficientNetB0.pth, selected on validation only); see README.

Outputs:
  Figures/Figure_Exp10R_{Dataset}_tSNE.png
  Results/exp10R_crossdomain.csv / _report.txt

Run from repository root:  python exp10_cross_domain.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import gc
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from exp10d_multiseed import balanced_sampler
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn not found — t-SNE and Silhouette will be skipped.")

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR       = "Models"
RESULT_DIR      = "Results"
FIG_DIR         = "Figures"
LOG_FILE        = "pgml_agent_progress.log"

CHAMPION_WEIGHTS    = os.path.join(MODEL_DIR, "Gate1_Champion_EfficientNetB0.pth")

FINETUNE_EPOCHS = 5
TRAIN_FRACTION  = 0.15
FULL_FRACTION   = 1.00      # 100% data baseline for Directive 3
BATCH_SIZE      = 32
FEAT_LIMIT      = 700
TSNE_PERPLEXITY = 30
SEED            = 42

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# Datasets — same roots as exp3
DATASET_REGISTRY = [
    {"name": "DAGM",     "root": "DAGM_Cleaned", "binary": True, "color": "#EA0CD0"},
    {"name": "Kolektor", "root": "Kolektor",     "binary": True, "color": "#D41212"},
]

# ============================================================================
# HELPERS
# ============================================================================
def log_append(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_loaders(root, fraction=TRAIN_FRACTION):
    ds    = datasets.ImageFolder(root, transform=get_transform())
    n_cls = len(ds.classes)
    idx   = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED)).tolist()
    pin   = (DEVICE.type == "cuda")
    if fraction >= 1.0:
        # 100% data: use all samples for fine-tuning; fixed held-out test set (last 15%)
        n_te  = max(1, int(0.15 * len(ds)))
        tr_loader = DataLoader(Subset(ds, idx[:-n_te]), batch_size=BATCH_SIZE,
                               sampler=balanced_sampler(ds, idx[:-n_te], SEED),
                               num_workers=4, pin_memory=pin)
        te_loader = DataLoader(Subset(ds, idx[-n_te:]),  batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=4, pin_memory=pin)
    else:
        n_tr  = int(fraction * len(ds))
        tr_loader = DataLoader(Subset(ds, idx[:n_tr]),  batch_size=BATCH_SIZE,
                               sampler=balanced_sampler(ds, idx[:n_tr], SEED),
                               num_workers=4, pin_memory=pin)
        te_loader = DataLoader(Subset(ds, idx[n_tr:]),  batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=4, pin_memory=pin)
    return tr_loader, te_loader, n_cls, ds.classes


def load_champion_backbone(n_cls, device):
    """Load the casting champion checkpoint, freeze backbone, replace head."""
    if not os.path.exists(CHAMPION_WEIGHTS):
        raise FileNotFoundError(f"PGML weights not found: {CHAMPION_WEIGHTS}")
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    m.load_state_dict(torch.load(CHAMPION_WEIGHTS, map_location=device, weights_only=True))
    for p in m.parameters():
        p.requires_grad_(False)
    # Replace head for target n_cls
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, n_cls)
    m.classifier[1].requires_grad_(True)
    return m.to(device)


def evaluate(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            _, p = torch.max(model(X), 1)
            correct += (p == y).sum().item(); total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def extract_features(model, loader, max_s=FEAT_LIMIT):
    act = {}
    hook = model.avgpool.register_forward_hook(lambda m, i, o: act.update({"f": o.detach().cpu()}))
    feats, labs, seen = [], [], 0
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            if seen >= max_s:
                break
            model(X.to(DEVICE))
            f   = act["f"].squeeze(-1).squeeze(-1)
            rem = max_s - seen
            feats.append(f[:rem].numpy()); labs.append(y[:rem].numpy()); seen += f.shape[0]
    hook.remove()
    return np.vstack(feats), np.hstack(labs)


def cluster_metrics(feats, labels):
    if not SKLEARN_OK:
        return float("nan"), float("nan")
    n = len(np.unique(labels))
    if n < 2 or feats.shape[0] < n + 1:
        return float("nan"), float("nan")
    sil = silhouette_score(feats, labels, sample_size=min(3000, len(feats)), random_state=SEED)
    dbi = davies_bouldin_score(feats, labels)
    return sil, dbi


def compute_tsne(feats):
    if not SKLEARN_OK:
        return np.zeros((len(feats), 2))
    return TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=SEED,
                max_iter=500, init="pca").fit_transform(feats)

# ============================================================================
# PHYSICS LOSS (binary only — same formulation as exp5)
# ============================================================================

# ============================================================================
# FIGURES
# ============================================================================
def fig_tsne(ds_name, class_names, labels, tsne_b, tsne_a,
             acc_b, acc_a, sil_b, sil_a):
    cmap = plt.cm.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Cross-Domain t-SNE — {ds_name}\n"
                 f"Casting-Pretrained Champion Backbone  |  CE Fine-Tune (15%, 5 epochs)",
                 fontsize=12, fontweight="bold")
    for ax, proj, title, sub in [
        (axes[0], tsne_b, f"{ds_name} — Zero-Shot",
         f"Acc={acc_b:.1f}%  Sil={sil_b:.3f}"),
        (axes[1], tsne_a, f"{ds_name} — After 15% Fine-Tune",
         f"Acc={acc_a:.1f}%  Sil={sil_a:.3f}"),
    ]:
        for ci, cn in enumerate(class_names):
            mask = labels == ci
            if not mask.any():
                continue
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=[cmap(ci / max(len(class_names)-1, 1))],
                       label=cn, s=16, alpha=0.75, linewidths=0)
        ax.set_title(title, fontweight="bold", fontsize=10, pad=6)
        ax.set_xlabel("Dim 1", fontsize=9); ax.set_ylabel("Dim 2", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.7, markerscale=1.3,
                  ncol=max(1, len(class_names)//4))
        ax.text(0.02, 0.98, sub, transform=ax.transAxes, fontsize=8,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.grid(ls="--", alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    safe = ds_name.replace("-", "").replace(" ", "_")
    out  = os.path.join(FIG_DIR, f"Figure_Exp10R_{safe}_tSNE.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    np.savez(os.path.join("Results", f"exp10R_tsne_{safe}.npz"),
             labels=labels, tsne_b=tsne_b, tsne_a=tsne_a,
             acc_b=acc_b, acc_a=acc_a, sil_b=sil_b, sil_a=sil_a,
             class_names=np.array(class_names, dtype=object))
    print(f"  [Fig] {out}")


# ============================================================================
# PROCESS ONE DATASET
# ============================================================================
def process_dataset(ds_cfg):
    dn   = ds_cfg["name"]
    root = ds_cfg["root"]
    is_binary = ds_cfg["binary"]

    print(f"\n  {'─'*52}\n  Fine-tune: {dn}\n  {'─'*52}")
    if not os.path.isdir(root):
        print(f"  [SKIP] {root} not found.")
        return None

    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr, te, n_cls, cls_names = get_loaders(root)
    print(f"  Classes ({n_cls}): {cls_names}")

    model = load_champion_backbone(n_cls, DEVICE)
    print("  Loss: cross-entropy (head-only fine-tune)")

    acc_b  = evaluate(model, te)
    if SKLEARN_OK:
        feats_b, lbls = extract_features(model, te)
        sil_b, dbi_b  = cluster_metrics(feats_b, lbls)
        tsne_b        = compute_tsne(feats_b)
    else:
        feats_b = lbls = None; sil_b = dbi_b = float("nan"); tsne_b = None
    print(f"  Zero-shot: Acc={acc_b:.2f}%  Sil={sil_b:.4f}")

    # Fine-tune ----------------------------------------------------------------
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.classifier[1].parameters(), lr=1e-3)
    model.train()
    for ep in range(FINETUNE_EPOCHS):
        loss_sum = n_batches = 0
        for X, y in tr:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward(); opt.step()
            loss_sum += loss.item(); n_batches += 1
        print(f"  EP{ep+1} CE={loss_sum/n_batches:.4f}", flush=True)

    acc_a  = evaluate(model, te)
    if SKLEARN_OK:
        feats_a, _ = extract_features(model, te)
        sil_a, dbi_a = cluster_metrics(feats_a, lbls)
        tsne_a = compute_tsne(feats_a)
    else:
        feats_a = None; sil_a = dbi_a = float("nan"); tsne_a = None
    print(f"  Post-FT (15%): Acc={acc_a:.2f}%  Sil={sil_a:.4f}")

    # t-SNE figure
    if SKLEARN_OK and tsne_b is not None:
        fig_tsne(dn, cls_names, lbls, tsne_b, tsne_a,
                 acc_b, acc_a, sil_b, sil_a)

    # 100% Data Baseline pass (Directive 3 — agility vs peak-accuracy comparison)
    print(f"\n  --- 100% Data Fine-Tune Baseline ({dn}) ---")
    tr_full, te_full, _, _ = get_loaders(root, fraction=FULL_FRACTION)
    model_full = load_champion_backbone(n_cls, DEVICE)
    opt_full   = optim.Adam(model_full.classifier[1].parameters(), lr=1e-3)
    model_full.train()
    for ep in range(FINETUNE_EPOCHS):
        loss_sum = n_batches = 0
        for X, y in tr_full:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt_full.zero_grad()
            loss = crit(model_full(X), y)
            loss.backward(); opt_full.step()
            loss_sum += loss.item(); n_batches += 1
        print(f"  Full EP{ep+1} CE={loss_sum/n_batches:.4f}", flush=True)
    acc_full = evaluate(model_full, te_full)
    print(f"  Post-FT (100%): Acc={acc_full:.2f}%")
    del model_full, opt_full
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - t0
    result  = {
        "ds_name":     dn, "n_classes": n_cls,
        "acc_before":  acc_b, "acc_after": acc_a, "acc_delta": acc_a - acc_b,
        "acc_full_ft": acc_full,
        "sil_before":  sil_b, "sil_after": sil_a,
        "sil_delta":   (sil_a - sil_b) if not (np.isnan(sil_b) or np.isnan(sil_a)) else float("nan"),
        "dbi_before":  dbi_b, "dbi_after": dbi_a,
        "elapsed_s":   elapsed,
    }

    for obj in [model, opt, crit]:
        del obj
    if feats_b is not None: del feats_b
    if feats_a is not None: del feats_a
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    gc.collect()
    return result

# ============================================================================
# MAIN
# ============================================================================
def run():
    torch.manual_seed(SEED); np.random.seed(SEED)

    print("=" * 65)
    print(f"  exp10 — Cross-Domain Fine-Tuning  |  {DEVICE}")
    print(f"  Datasets: {[d['name'] for d in DATASET_REGISTRY]}")
    print(f"  EPOCHS={FINETUNE_EPOCHS}  TRAIN_FRAC={TRAIN_FRACTION}")
    print("=" * 65)

    all_results = []
    for ds_cfg in DATASET_REGISTRY:
        r = process_dataset(ds_cfg)
        if r is not None:
            all_results.append(r)

    if not all_results:
        print("\n[WARN] No results — check dataset paths.")
        return

    # CSV
    csv_path = os.path.join(RESULT_DIR, "exp10R_crossdomain.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["dataset", "n_classes",
                    "acc_zero_shot", "acc_ft_15pct", "acc_delta",
                    "acc_full_ft_100pct",
                    "sil_before", "sil_after", "sil_delta", "elapsed_s"])
        for r in all_results:
            w.writerow([r["ds_name"], r["n_classes"],
                        f"{r['acc_before']:.4f}", f"{r['acc_after']:.4f}",
                        f"{r['acc_delta']:+.4f}",
                        f"{r.get('acc_full_ft', float('nan')):.4f}",
                        f"{r['sil_before']:.4f}", f"{r['sil_after']:.4f}",
                        f"{r['sil_delta']:+.4f}", f"{r['elapsed_s']:.1f}"])
    print(f"\n[CSV] {csv_path}")

    # Report
    lines = [
        "=" * 70,
        "  exp10 — CROSS-DOMAIN FINE-TUNING REPORT",
        f"  Model: casting-pretrained champion EfficientNet-B0 (CE fine-tune)",
        f"  15% Budget: Rapid in-shift line-change agility protocol",
        f"  100% Budget: Peak-accuracy upper-bound baseline",
        "=" * 70,
    ]
    for r in all_results:
        full_acc  = r.get("acc_full_ft", float("nan"))
        agility_gap = full_acc - r["acc_after"] if not np.isnan(full_acc) else float("nan")
        lines.append(
            f"\n  {r['ds_name']} ({r['n_classes']} classes)  |  Loss: CE\n"
            f"  {'─'*50}\n"
            f"  Zero-Shot        : Acc={r['acc_before']:.2f}%  Sil={r['sil_before']:.4f}\n"
            f"  15% Budget FT    : Acc={r['acc_after']:.2f}%  Sil={r['sil_after']:.4f}\n"
            f"  100% Budget FT   : Acc={full_acc:.2f}%  (peak, not agility target)\n"
            f"  Agility gap      : {agility_gap:+.2f}pp  (price of 85% data reduction)\n"
            f"  Delta vs ZS (15%): Acc={r['acc_delta']:+.2f}pp  Sil={r['sil_delta']:+.4f}"
        )

    lines += ["", "=" * 70]
    report = "\n".join(lines)
    print("\n" + report)

    rp = os.path.join(RESULT_DIR, "exp10R_crossdomain_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Report] {rp}")

    log_append(
        f"\n[exp10 cross-domain PGML — {time.strftime('%Y-%m-%d %H:%M')}]\n" +
        "\n".join(
            f"  {r['ds_name']}: ZS={r['acc_before']:.2f}% "
            f"FT={r['acc_after']:.2f}% (delta={r['acc_delta']:+.2f}%)"
            for r in all_results
        )
    )
    print("\n[Done]")


if __name__ == "__main__":
    run()
