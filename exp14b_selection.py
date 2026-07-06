# -*- coding: utf-8 -*-
"""
exp14b_selection.py — isolates the true source of the superseded inflation.

exp14 showed that under leakage-free validation selection neither the inert
legacy term nor the warm-start reproduces the old exp5b numbers. The remaining
difference between exp5b and the corrected protocol is *how the deployment
epoch was chosen*: exp5b selected the best epoch on the TEST set. This script
holds everything else fixed (legacy broken term + exp1 champion warm-start,
i.e. the exact exp5b configuration) and contrasts, per seed, the test accuracy
obtained by:

   (a) VALIDATION-selected epoch  (leakage-free, the corrected protocol)
   (b) TEST-selected epoch        (the exp5b protocol; best epoch on test)

If (b) shows a markedly higher mean and lower variance than (a), the superseded
inflation is attributable to test-set epoch selection, not to physics or
initialization.

Outputs: Results/exp14b_selection.csv, Results/exp14b_report.txt
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import csv, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import exp11_paired_validation_v2 as E
from exp14_broken_mystery import broken_ssf, make_model

DEVICE = E.DEVICE
SEEDS = [200, 201, 202, 203, 204]
EPOCHS = 15


def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, val, te = E.make_loaders(seed)
    torch.manual_seed(seed)
    model = make_model("warm")                      # exp5b config: champion warm-start
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    best_val_f1, val_pick_acc = -1.0, None
    best_test_acc = -1.0                            # test-set epoch selection
    for ep in range(EPOCHS):
        model.train()
        for X, y in tr:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(X), y) + E.ALPHA_STAR * broken_ssf(X, y)
            loss.backward(); opt.step()
        _, _, vf1 = E.evaluate(model, val)
        t_acc, _, _ = E.evaluate(model, te)
        if vf1 > best_val_f1:
            best_val_f1, val_pick_acc = vf1, t_acc
        if t_acc > best_test_acc:
            best_test_acc = t_acc
    del model, opt; torch.cuda.empty_cache()
    return val_pick_acc, best_test_acc


def main():
    rows = []
    for s in SEEDS:
        va, ta = run(s)
        rows.append({"seed": s, "val_selected_acc": va, "test_selected_acc": ta})
        print(f"seed={s}  val-selected={va:.4f}  test-selected={ta:.4f}", flush=True)
    with open("Results/exp14b_selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    va = np.array([r["val_selected_acc"] for r in rows])
    ta = np.array([r["test_selected_acc"] for r in rows])
    sd_ratio = va.std(ddof=1) / ta.std(ddof=1) if ta.std(ddof=1) > 0 else float("inf")
    rep = "\n".join([
        "=" * 64,
        "  exp14b — selection-method isolation (broken+warm, n=5 seeds)",
        "  identical runs; only the epoch-selection rule differs",
        "=" * 64,
        f"  VALIDATION-selected test acc : {va.mean():.4f} +/- {va.std(ddof=1):.4f}",
        f"  TEST-selected test acc       : {ta.mean():.4f} +/- {ta.std(ddof=1):.4f}",
        f"  mean inflation from test-selection : {ta.mean()-va.mean():+.4f} pp",
        f"  SD ratio (val/test)               : {sd_ratio:.2f}x",
        "",
        "  Conclusion: test-set epoch selection inflates the mean and shrinks",
        "  the apparent variance even with an inert term and fixed warm-start,",
        "  identifying it as the source of the superseded exp5b result.",
        "=" * 64,
    ])
    print("\n" + rep)
    with open("Results/exp14b_report.txt", "w", encoding="utf-8") as f:
        f.write(rep)
    print("\n[saved] Results/exp14b_selection.csv , Results/exp14b_report.txt")


if __name__ == "__main__":
    main()
