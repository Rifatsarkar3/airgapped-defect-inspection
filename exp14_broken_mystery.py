# -*- coding: utf-8 -*-
"""
exp14_broken_mystery.py — Why the broken formulation looked positive.

The superseded n=30 "validation" (exp5b) reported a large variance reduction
and high accuracy for an SSF term that was, in fact, inert: it computed Sobel
gradient energy on raw input pixels inside torch.no_grad(), contributing zero
parameter gradient. This experiment isolates the cause by crossing the inert
term against the only other thing exp5b changed — the weight initialization.

2x2 design, 5 seeds per cell, validation-based epoch selection (so test-set
leakage is removed and initialization is the only free variable):

    {BROKEN-SSF (inert) , CE-only}  x  {WARM-start (exp1 champion) , ImageNet}

If the apparent exp5b gain was warm-start confounding, then (i) BROKEN-SSF and
CE-only are statistically identical within each init (the inert term does
nothing), and (ii) WARM-start shows higher mean accuracy and lower
cycle-to-cycle variance than ImageNet init, regardless of the loss.

Outputs: Results/exp14_broken_mystery.csv, Results/exp14_report.txt
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import csv, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import exp11_paired_validation as E

DEVICE = E.DEVICE
SEEDS = [200, 201, 202, 203, 204]
EPOCHS = 15
WARM = os.path.join("Models", "Best_EfficientNetB0_Exp1.pth")

# Raw-pixel Sobel kernels (the BROKEN variant operated on the input image)
_KX = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_KY = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def broken_ssf(x, labels, margin=E.MARGIN):
    """Faithful reproduction of the inert legacy term: raw-pixel gradient
    energy computed entirely under no_grad, returned detached. Adds a constant
    to the loss with zero gradient path to parameters."""
    with torch.no_grad():
        kx, ky = _KX.to(x.device), _KY.to(x.device)
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8).mean(dim=(1, 2, 3))
        ok = labels == E.OK_CLASS
        df = labels == E.DEF_CLASS
        if ok.sum() == 0 or df.sum() == 0:
            return torch.zeros((), device=x.device)
        return F.relu(mag[ok].mean() - mag[df].mean() + margin)


def make_model(init):
    m = E.build_model()
    if init == "warm":
        m.load_state_dict(torch.load(WARM, map_location="cpu", weights_only=True))
    return m.to(DEVICE)


def recall(model, loader):
    model.eval(); tp = fn = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            p = model(X).argmax(1)
            tp += ((p == E.DEF_CLASS) & (y == E.DEF_CLASS)).sum().item()
            fn += ((p != E.DEF_CLASS) & (y == E.DEF_CLASS)).sum().item()
    return tp / (tp + fn) * 100 if (tp + fn) else 0.0


def run(loss_kind, init, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, val, te = E.make_loaders(seed)
    torch.manual_seed(seed)
    model = make_model(init)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    best_f1, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        for X, y in tr:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(X), y)
            if loss_kind == "broken":
                loss = loss + E.ALPHA_STAR * broken_ssf(X, y)
            loss.backward(); opt.step()
        _, _, f1 = E.evaluate(model, val)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    acc, rec, f1 = E.evaluate(model, te)
    del model, opt; torch.cuda.empty_cache()
    return acc, rec, f1


def main():
    cells = [("broken", "warm"), ("CE", "warm"),
             ("broken", "imagenet"), ("CE", "imagenet")]
    rows = []
    for loss_kind, init in cells:
        for seed in SEEDS:
            acc, rec, f1 = run(loss_kind, init, seed)
            rows.append({"loss": loss_kind, "init": init, "seed": seed,
                         "acc": acc, "recall": rec, "f1": f1})
            print(f"{loss_kind:6s} {init:8s} seed={seed}  acc={acc:.4f}  "
                  f"rec={rec:.4f}  f1={f1:.4f}", flush=True)

    with open("Results/exp14_broken_mystery.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def cell(loss_kind, init, key):
        v = np.array([r[key] for r in rows if r["loss"] == loss_kind and r["init"] == init])
        return v.mean(), v.std(ddof=1)

    L = ["=" * 70,
         "  exp14 — Broken-Formulation Mystery (2x2, n=5 seeds/cell)",
         "  validation-based selection; BROKEN-SSF is inert (no_grad, raw pixel)",
         "=" * 70, ""]
    for init in ("warm", "imagenet"):
        L.append(f"  init = {init}")
        for loss_kind in ("broken", "CE"):
            ma, sa = cell(loss_kind, init, "acc")
            mr, sr = cell(loss_kind, init, "recall")
            L.append(f"    {loss_kind:6s}: acc {ma:.4f} +/- {sa:.4f}   recall {mr:.4f} +/- {sr:.4f}")
        L.append("")
    # key contrasts
    bw_m, bw_s = cell("broken", "warm", "acc")
    bi_m, bi_s = cell("broken", "imagenet", "acc")
    cw_m, cw_s = cell("CE", "warm", "acc")
    ci_m, ci_s = cell("CE", "imagenet", "acc")
    L += [
        "  Key contrasts (accuracy):",
        f"    warm-start advantage (broken):  {bw_m - bi_m:+.4f} pp  (SD {bw_s:.4f} vs {bi_s:.4f})",
        f"    warm-start advantage (CE):      {cw_m - ci_m:+.4f} pp  (SD {cw_s:.4f} vs {ci_s:.4f})",
        f"    broken-vs-CE within warm:       {bw_m - cw_m:+.4f} pp  (inert term => ~0 expected)",
        f"    broken-vs-CE within imagenet:   {bi_m - ci_m:+.4f} pp  (inert term => ~0 expected)",
        "",
        "  Reading: if warm-start advantage >> broken-vs-CE gap, the apparent",
        "  exp5b 'physics' gain was initialization, not the (inert) SSF term.",
        "=" * 70,
    ]
    rep = "\n".join(L)
    print("\n" + rep)
    with open("Results/exp14_report.txt", "w", encoding="utf-8") as f:
        f.write(rep)
    print("\n[saved] Results/exp14_broken_mystery.csv , Results/exp14_report.txt")


if __name__ == "__main__":
    main()
