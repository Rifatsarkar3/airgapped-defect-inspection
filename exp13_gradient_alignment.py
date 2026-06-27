# -*- coding: utf-8 -*-
"""
exp13_gradient_alignment.py — Regularizer-Objective Gradient Alignment.

Measures, on the REAL model, the quantity that explains why the corrected SSF
term fails to improve over cross-entropy. For a parameter state theta and a
mini-batch, define

    A(theta) = <grad L_SSF, grad L_CE> / (||grad L_SSF|| ||grad L_CE||)

the cosine alignment between the SSF update direction and the CE update
direction in parameter space, and the Effective Novel-Gradient Fraction

    eta = 1 - A(theta)^2

(the share of the SSF update orthogonal to CE). A -> +1 (eta -> 0) means SSF
is redundant with CE; A < 0 means SSF is antagonistic on that batch. We also
compute A restricted to the defective (recall-relevant) class:

    A_rec = <grad L_SSF, grad L_CE^{defective-only}> / (norms)

A_rec < 0 predicts the empirical recall penalty observed in the paired
evaluation (Experiment 5).

Both L_CE and the corrected feature-space L_SSF are imported verbatim from
exp11_paired_validation.py, so this measures the exact loss pair the paper
reports. Two parameter states are evaluated: ImageNet-initialized (the shared
start of both paired arms) and the deployed CE-trained champion.

Outputs: Results/exp13_gradient_alignment.csv, Results/exp13_report.txt
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import csv, os
import numpy as np
import torch
import torch.nn as nn

import exp11_paired_validation as E

DEVICE = E.DEVICE
N_BATCHES = 30
SEED = 42
WARM = os.path.join("Models", "PGML_Champion_EfficientNetB0.pth")


def flat_grad(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    vec = []
    for gi, p in zip(g, params):
        vec.append((gi if gi is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(vec)


def cosine(a, b):
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def measure(model, loader):
    crit = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    A_full, A_rec, eta = [], [], []
    n = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        if (y == E.DEF_CLASS).sum() == 0 or (y == E.OK_CLASS).sum() == 0:
            continue
        logits, feat = E.forward_with_feature(model, X)
        L_ce = crit(logits, y)
        L_ssf = E.ssf_coupled_loss(feat, y)
        if not L_ssf.requires_grad or float(L_ssf) == 0.0:
            # hinge inactive on this batch (already satisfied) -> no SSF signal
            continue
        g_ce = flat_grad(L_ce, params)
        g_ssf = flat_grad(L_ssf, params)
        a = cosine(g_ssf, g_ce)
        A_full.append(a)
        eta.append(1.0 - a * a)
        # recall-relevant: CE gradient from defective-class samples only
        dmask = (y == E.DEF_CLASS)
        L_ce_def = crit(logits[dmask], y[dmask])
        g_ce_def = flat_grad(L_ce_def, params)
        A_rec.append(cosine(g_ssf, g_ce_def))
        n += 1
        if n >= N_BATCHES:
            break
    return np.array(A_full), np.array(A_rec), np.array(eta)


def ci95(v):
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    m = v.mean(); h = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return (m, h)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr, val, te = E.make_loaders(SEED)

    states = []
    m0 = E.build_model().to(DEVICE)
    states.append(("ImageNet-init", m0))
    m1 = E.build_model()
    m1.load_state_dict(torch.load(WARM, map_location=DEVICE, weights_only=True))
    m1 = m1.to(DEVICE)
    states.append(("CE-champion", m1))

    rows, lines = [], []
    lines.append("=" * 68)
    lines.append("  exp13 — Regularizer-Objective Gradient Alignment")
    lines.append("  A(theta)=cos(grad L_SSF, grad L_CE) ; eta=1-A^2 ;")
    lines.append("  A_rec=cos(grad L_SSF, grad L_CE[defective-only])")
    lines.append(f"  loss pair imported from exp11 ; batches/state={N_BATCHES}")
    lines.append("=" * 68)
    for name, model in states:
        A, Arec, eta = measure(model, tr)
        mA, hA = ci95(A); mE, hE = ci95(eta); mR, hR = ci95(Arec)
        frac_neg = float(np.mean(A < 0)) if len(A) else float("nan")
        frac_neg_rec = float(np.mean(Arec < 0)) if len(Arec) else float("nan")
        rows.append({"state": name, "n_active_batches": len(A),
                     "A_mean": f"{mA:.4f}", "A_ci95": f"{hA:.4f}",
                     "eta_mean": f"{mE:.4f}", "eta_ci95": f"{hE:.4f}",
                     "A_rec_mean": f"{mR:.4f}", "A_rec_ci95": f"{hR:.4f}",
                     "frac_A_negative": f"{frac_neg:.3f}",
                     "frac_Arec_negative": f"{frac_neg_rec:.3f}"})
        lines += [
            "",
            f"  [{name}]  active batches: {len(A)}",
            f"    A(theta)      = {mA:+.4f} +/- {hA:.4f}   (1=redundant, 0=orthogonal, <0=antagonistic)",
            f"    eta=1-A^2     = {mE:.4f} +/- {hE:.4f}   (novel-gradient fraction)",
            f"    A_recall      = {mR:+.4f} +/- {hR:.4f}   (<0 predicts recall penalty)",
            f"    %% batches A<0 = {100*frac_neg:.1f}%% ; %% batches A_rec<0 = {100*frac_neg_rec:.1f}%%",
        ]
    lines += ["", "=" * 68]
    report = "\n".join(lines)
    print(report)

    os.makedirs("Results", exist_ok=True)
    with open("Results/exp13_gradient_alignment.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open("Results/exp13_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n[saved] Results/exp13_gradient_alignment.csv , Results/exp13_report.txt")


if __name__ == "__main__":
    main()
