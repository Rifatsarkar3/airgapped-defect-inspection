"""
exp11_paired_validation.py — Corrected PGML vs Baseline Paired Validation
==========================================================================
Supersedes exp5/exp5b. Fixes three protocol defects found in the 2026-06-11
audit (see archive/README.md):

  1. DETACHED PHYSICS LOSS — exp5/exp5b computed the SSF term under
     torch.no_grad() on raw input pixels, so it had no gradient path to the
     model. Here the Griffith gradient-energy hinge is applied to an
     intermediate FEATURE MAP, making it differentiable w.r.t. the weights:

         L_phys = ReLU( E[||grad F(x)||_2 | OK] - E[||grad F(x)||_2 | def] + MARGIN )

     where F(x) is the per-image standardized channel-mean of the
     EfficientNet-B0 stage-FEATURE_STAGE feature map and the gradient is a
     fixed 3x3 Sobel operator.

  2. CONFOUNDED INITIALIZATION — exp5b warm-started from the exp1 champion
     checkpoint while the baseline trained from ImageNet weights. Here both
     arms of every cycle start from IDENTICAL ImageNet-pretrained weights,
     identical classifier init (same seed), and identical data order. The
     only difference between arms is the physics term.

  3. TEST-SET SELECTION — exp5b picked the best epoch by test F1. Here a
     stratified 15% validation split is carved from train; the best epoch is
     selected on validation F1 and the test set is evaluated exactly once
     per arm, on that state.

A gradient-flow assertion runs before any training: if the physics term
contributes zero gradient to the model parameters, the script aborts.

Protocol: N_CYCLES=30 paired cycles, seeds 200..229 (disjoint from exp1/exp5
seed pools). Statistics: paired t-test across seeds + variance comparison.

Outputs:
    Results/exp11_paired_results.csv     -- one row per (cycle, arm)
    Results/exp11_epoch_log.csv          -- per-epoch validation metrics
    Results/exp11_paired_report.txt      -- statistical report
    Models/Exp11_PGML_Champion.pth       -- best pgml arm state (by test F1)
    pgml_agent_progress.log (appended)

Run from TF_Me project root:
    python exp11_paired_validation.py            # full n=30 (resume-safe)
    python exp11_paired_validation.py --smoke    # 1 cycle, 2 epochs, 25 batches
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import csv
import gc
import time
import numpy as np
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# ============================================================================
# CONFIGURATION
# ============================================================================
SMOKE        = "--smoke" in sys.argv

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR    = os.path.join("casting_data", "train")
TEST_DIR     = os.path.join("casting_data", "test")
MODEL_DIR    = "Models"
RESULT_DIR   = "Results"
LOG_FILE     = "pgml_agent_progress.log"

ALPHA_STAR    = 0.1     # physics loss weight (a-priori; prior ablation void)
MARGIN        = 0.10    # Griffith hinge margin in standardized feature units
FEATURE_STAGE = 2       # EfficientNet-B0 features[2] output: 24ch, 56x56
EPOCHS        = 15 if not SMOKE else 2
BATCH_SIZE    = 32
VAL_FRACTION  = 0.15
N_CYCLES      = 30 if not SMOKE else 1
MAX_BATCHES   = None if not SMOKE else 25
LR            = 1e-3

# Seeds disjoint from exp1 (n=30 pool), exp5 {42,123,777}, exp5b 100..129
CYCLE_SEEDS   = list(range(200, 200 + N_CYCLES))

SUFFIX       = "_smoke" if SMOKE else ""
SUMMARY_CSV  = os.path.join(RESULT_DIR, f"exp11_paired_results{SUFFIX}.csv")
EPOCH_CSV    = os.path.join(RESULT_DIR, f"exp11_epoch_log{SUFFIX}.csv")
REPORT_TXT   = os.path.join(RESULT_DIR, f"exp11_paired_report{SUFFIX}.txt")
CHAMPION_PTH = os.path.join(MODEL_DIR, "Exp11_PGML_Champion.pth")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Fixed Sobel kernels — non-learnable
_KX = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_KY = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

# ImageFolder class order is alphabetical: def_front=0, ok_front=1
DEF_CLASS, OK_CLASS = 0, 1


# ============================================================================
# MODEL — forward that also exposes the tapped feature map
# ============================================================================
def build_model():
    m = models.efficientnet_b0(weights="DEFAULT")
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def forward_with_feature(model, x):
    """Standard EfficientNet forward, additionally returning the
    features[FEATURE_STAGE] activation for the physics term."""
    feat = None
    h = x
    for i, block in enumerate(model.features):
        h = block(h)
        if i == FEATURE_STAGE:
            feat = h
    h = model.avgpool(h).flatten(1)
    return model.classifier(h), feat


# ============================================================================
# SOBEL SURFACE FINGERPRINTING — COUPLED PHYSICS LOSS
# ============================================================================
def ssf_coupled_loss(feat, labels, margin=MARGIN):
    """Griffith gradient-energy hinge on the model's feature map.

    Differentiable w.r.t. model parameters: pushes the network toward
    representations in which defective surfaces carry higher mean spatial
    gradient energy than nominal ones, by at least `margin`.
    """
    fm = feat.mean(dim=1, keepdim=True)                       # channel-mean map
    mu = fm.mean(dim=(2, 3), keepdim=True)
    sd = fm.std(dim=(2, 3), keepdim=True)
    fm = (fm - mu) / (sd + 1e-5)                              # per-image standardize
    kx = _KX.to(fm.device, fm.dtype)
    ky = _KY.to(fm.device, fm.dtype)
    gx = F.conv2d(fm, kx, padding=1)
    gy = F.conv2d(fm, ky, padding=1)
    mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8).mean(dim=(1, 2, 3))
    ok_mask  = labels == OK_CLASS
    def_mask = labels == DEF_CLASS
    if not ok_mask.any() or not def_mask.any():
        return mag.sum() * 0.0                                # keeps graph, zero value
    return F.relu(mag[ok_mask].mean() - mag[def_mask].mean() + margin)


def assert_physics_grad_flow():
    """Abort unless the physics term alone produces nonzero parameter
    gradients — the exact silent failure that invalidated exp5/exp5b."""
    torch.manual_seed(0)
    model = build_model().to(DEVICE)
    X = torch.randn(8, 3, 224, 224, device=DEVICE)
    y = torch.tensor([DEF_CLASS, OK_CLASS] * 4, device=DEVICE)
    _, feat = forward_with_feature(model, X)
    phys = ssf_coupled_loss(feat, y, margin=1.0)              # force hinge active
    phys.backward()
    gnorm = sum(p.grad.abs().sum().item()
                for p in model.parameters() if p.grad is not None)
    del model
    torch.cuda.empty_cache()
    if gnorm <= 0:
        raise RuntimeError(
            "Physics loss contributes ZERO gradient to model parameters. "
            "Refusing to run a void experiment.")
    print(f"[Grad check] PASS — physics-term parameter grad L1 norm = {gnorm:.4e}")


# ============================================================================
# DATA — stratified train/val split, identical order across arms
# ============================================================================
def make_loaders(seed):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_train = datasets.ImageFolder(TRAIN_DIR, transform=tf)
    test_ds    = datasets.ImageFolder(TEST_DIR,  transform=tf)

    targets = np.array(full_train.targets)
    rng = np.random.RandomState(seed)
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
    g = torch.Generator(); g.manual_seed(seed)
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True, generator=g)

    tr_loader  = DataLoader(Subset(full_train, tr_idx), batch_size=BATCH_SIZE,
                            sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
    te_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
    return tr_loader, val_loader, te_loader


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate(model, loader):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            _, p = model(X).max(1)
            tp += ((p == 1) & (y == 1)).sum().item()
            fp += ((p == 1) & (y == 0)).sum().item()
            fn += ((p == 0) & (y == 1)).sum().item()
            tn += ((p == 0) & (y == 0)).sum().item()
    N   = tp + tn + fp + fn
    acc = (tp + tn) / N * 100 if N else 0.0
    pr  = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    rc  = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    f1  = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
    return acc, rc, f1


def log_append(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ============================================================================
# ONE ARM — one full training run, val-based selection, single test eval
# ============================================================================
def run_arm(arm, cycle_idx, seed, epoch_writer, epoch_file):
    use_phys = (arm == "pgml")

    # Identical init and data order across arms: same seed everywhere.
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, val, te = make_loaders(seed)
    torch.manual_seed(seed)               # classifier head init parity
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)

    best_val_f1, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        phys_sum, n_batches = 0.0, 0
        for X, y in tr:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if use_phys:
                logits, feat = forward_with_feature(model, X)
                phys = ssf_coupled_loss(feat, y)
                loss = criterion(logits, y) + ALPHA_STAR * phys
                phys_sum += phys.item()
            else:
                loss = criterion(model(X), y)
            loss.backward()
            opt.step()
            n_batches += 1
            if MAX_BATCHES and n_batches >= MAX_BATCHES:
                break

        v_acc, v_rc, v_f1 = evaluate(model, val)
        if v_f1 > best_val_f1:
            best_val_f1, best_epoch = v_f1, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        phys_avg = phys_sum / max(n_batches, 1)
        epoch_writer.writerow([cycle_idx, seed, arm, epoch,
                               f"{v_acc:.4f}", f"{v_rc:.4f}", f"{v_f1:.4f}",
                               f"{phys_avg:.6f}"])
        epoch_file.flush()
        print(f"    [{arm:8s}] ep{epoch:02d}  val F1={v_f1:.4f}%"
              + (f"  phys={phys_avg:.4f}" if use_phys else ""), flush=True)

    model.load_state_dict(best_state)
    t_acc, t_rc, t_f1 = evaluate(model, te)

    state = best_state
    del model, opt
    torch.cuda.empty_cache()
    gc.collect()
    return {"acc": t_acc, "recall": t_rc, "f1": t_f1,
            "best_epoch": best_epoch, "val_f1": best_val_f1, "state": state}


# ============================================================================
# STATISTICS + REPORT (computed from the summary CSV — resume-friendly)
# ============================================================================
def write_report():
    rows = list(csv.DictReader(open(SUMMARY_CSV, encoding="utf-8")))
    by_arm = {"baseline": {}, "pgml": {}}
    for r in rows:
        by_arm[r["arm"]][int(r["cycle"])] = r
    common = sorted(set(by_arm["baseline"]) & set(by_arm["pgml"]))
    if not common:
        print("[Report] no completed pairs yet"); return

    def vec(arm, key):
        return np.array([float(by_arm[arm][c][key]) for c in common])

    lines = ["=" * 70,
             "  exp11 PAIRED VALIDATION REPORT — corrected protocol",
             f"  alpha={ALPHA_STAR}  margin={MARGIN}  feature_stage={FEATURE_STAGE}",
             f"  epochs={EPOCHS}  batch={BATCH_SIZE}  val_fraction={VAL_FRACTION}",
             f"  completed pairs: n={len(common)}",
             "=" * 70, ""]
    for key, label in [("acc", "Accuracy"), ("recall", "Recall"), ("f1", "F1-Score")]:
        b, p = vec("baseline", key), vec("pgml", key)
        d = p - b
        if len(common) >= 2:
            t, pv = stats.ttest_rel(p, b)
            sig = "p < 0.001" if pv < 0.001 else f"p = {pv:.4f}"
        else:
            t, sig = float("nan"), "n/a"
        lines += [f"  {label}:",
                  f"    Baseline : {b.mean():.4f}% +/- {b.std():.4f}%",
                  f"    PGML     : {p.mean():.4f}% +/- {p.std():.4f}%",
                  f"    Paired delta: {d.mean():+.4f}% (paired t={t:.3f}, {sig})",
                  f"    SD ratio (baseline/PGML): "
                  f"{(b.std()/p.std() if p.std() > 0 else float('inf')):.2f}x", ""]
    report = "\n".join(lines + ["=" * 70])
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\n[Report saved] {REPORT_TXT}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("  exp11_paired_validation.py — corrected PGML paired protocol"
          + ("  [SMOKE]" if SMOKE else ""))
    print(f"  DEVICE={DEVICE}  alpha={ALPHA_STAR}  margin={MARGIN}  "
          f"stage={FEATURE_STAGE}  cycles={N_CYCLES}  epochs={EPOCHS}")
    print("=" * 70, flush=True)

    assert_physics_grad_flow()

    done = set()
    if os.path.exists(SUMMARY_CSV):
        for r in csv.DictReader(open(SUMMARY_CSV, encoding="utf-8")):
            done.add((int(r["cycle"]), r["arm"]))
        print(f"[Resume] {len(done)} completed (cycle, arm) entries found")
        summary_file = open(SUMMARY_CSV, "a", newline="", encoding="utf-8")
        summary_writer = csv.writer(summary_file)
    else:
        summary_file = open(SUMMARY_CSV, "w", newline="", encoding="utf-8")
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(["cycle", "seed", "arm", "acc", "recall", "f1",
                                 "best_epoch", "best_val_f1"])

    new_epoch_log = not os.path.exists(EPOCH_CSV)
    epoch_file = open(EPOCH_CSV, "a", newline="", encoding="utf-8")
    epoch_writer = csv.writer(epoch_file)
    if new_epoch_log:
        epoch_writer.writerow(["cycle", "seed", "arm", "epoch",
                               "val_acc", "val_recall", "val_f1", "phys_avg"])

    champ_f1 = -1.0
    t_wall = time.time()

    for i, seed in enumerate(CYCLE_SEEDS):
        cycle = i + 1
        print(f"\n--- Cycle {cycle:02d}/{N_CYCLES}  seed={seed} ---", flush=True)
        for arm in ("baseline", "pgml"):
            if (cycle, arm) in done:
                print(f"    [{arm:8s}] already complete — skipped")
                continue
            t0 = time.time()
            res = run_arm(arm, cycle, seed, epoch_writer, epoch_file)
            summary_writer.writerow([cycle, seed, arm,
                                     f"{res['acc']:.4f}", f"{res['recall']:.4f}",
                                     f"{res['f1']:.4f}", res["best_epoch"],
                                     f"{res['val_f1']:.4f}"])
            summary_file.flush()
            print(f"    [{arm:8s}] test Acc={res['acc']:.4f}%  "
                  f"Recall={res['recall']:.4f}%  F1={res['f1']:.4f}%  "
                  f"(best ep{res['best_epoch']}, {(time.time()-t0)/60:.1f} min)",
                  flush=True)
            if arm == "pgml" and res["f1"] > champ_f1 and not SMOKE:
                champ_f1 = res["f1"]
                torch.save(res["state"], CHAMPION_PTH)

    summary_file.close()
    epoch_file.close()

    write_report()

    total_h = (time.time() - t_wall) / 3600
    if not SMOKE:
        log_append(f"\n[exp11 COMPLETE — {time.strftime('%Y-%m-%d %H:%M')}]\n"
                   f"  Corrected paired protocol, n={N_CYCLES}. "
                   f"Report: {REPORT_TXT}. Wall time {total_h:.2f}h")
    print(f"\n[Done] Wall time: {total_h:.2f}h")


if __name__ == "__main__":
    main()
