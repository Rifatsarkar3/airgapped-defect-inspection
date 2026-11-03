# -*- coding: utf-8 -*-
"""
exp10d_multiseed.py — multi-seed cross-domain statistics (Table 6 source).

Single-seed cross-domain runs proved seed-sensitive (~1-3 pp swings on
Kolektor), so Table 8 reports mean ± SD over N_SEEDS independent seeds per
(dataset, initialization, budget) cell. Each seed controls both the data
split permutation and all training RNG. Protocol otherwise mirrors exp10:
frozen backbone, head-only Adam(1e-3) CE fine-tune, 5 epochs. As in the casting
experiments, the head is trained with a class-balanced sampler (WRS, weights
1/class_count on the training subset), and balanced accuracy and the held-out
majority-class rate are recorded: DAGM and Kolektor are ~87-89% "OK", so plain
accuracy alone cannot tell a defect detector from a constant predictor.

Outputs: Results/exp10D_multiseed.csv, Results/exp10D_report.txt
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import csv, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import silhouette_score

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = os.path.join("Models", "Gate1_Champion_EfficientNetB0.pth")
SEEDS   = [42, 43, 44, 45, 46]
EPOCHS  = 5
BATCH   = 32
FEAT_LIMIT = 700
DATASETS = [("NEU-DET", "NEU-DET_ByClass"), ("DAGM", "DAGM_Cleaned"),
            ("Kolektor", "Kolektor")]
BUDGETS  = [0.15, 1.00]

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def loaders(root, fraction, seed):
    ds  = datasets.ImageFolder(root, transform=TF)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed)).tolist()
    if fraction >= 1.0:
        n_te = max(1, int(0.15 * len(ds)))
        tr_idx, te_idx = idx[:-n_te], idx[-n_te:]
    else:
        n_tr = int(fraction * len(ds))
        tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
    tr = DataLoader(Subset(ds, tr_idx), batch_size=BATCH,
                    sampler=balanced_sampler(ds, tr_idx, seed),
                    num_workers=4, pin_memory=True)
    te = DataLoader(Subset(ds, te_idx), batch_size=BATCH, shuffle=False,
                    num_workers=4, pin_memory=True)
    return tr, te, len(ds.classes)


def balanced_sampler(ds, idx, seed):
    """WRS with per-class weights 1/class_count on the training subset."""
    t = np.array(ds.targets)[idx]
    w = 1.0 / np.bincount(t)[t]
    return WeightedRandomSampler(w.tolist(), len(w), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def build(init, n_cls):
    if init == "champion":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        m.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE, weights_only=True))
    else:
        m = models.efficientnet_b0(weights="DEFAULT")
    for p in m.parameters():
        p.requires_grad_(False)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    return m.to(DEVICE)


def evaluate(model, loader):
    """Accuracy, balanced accuracy (mean per-class recall) and majority-class rate, %."""
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for X, y in loader:
            ps.append(model(X.to(DEVICE)).argmax(1).cpu()); ys.append(y)
    y, p = torch.cat(ys), torch.cat(ps)
    cls = y.unique()
    bal = torch.stack([(p[y == c] == c).float().mean() for c in cls]).mean().item()
    maj = max((y == c).float().mean().item() for c in cls)
    return 100 * (p == y).float().mean().item(), 100 * bal, 100 * maj


def sil_after(model, loader, seed):
    model.eval(); feats, lbls = [], []
    with torch.no_grad():
        for X, y in loader:
            f = model.avgpool(model.features(X.to(DEVICE))).flatten(1)
            feats.append(f.cpu().numpy()); lbls.append(y.numpy())
            if sum(map(len, lbls)) >= FEAT_LIMIT:
                break
    f = np.concatenate(feats)[:FEAT_LIMIT]
    l = np.concatenate(lbls)[:FEAT_LIMIT]
    if len(set(l.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(f, l, sample_size=min(3000, len(f)),
                                  random_state=seed))


def main():
    rows = []
    t0 = time.time()
    for name, root in DATASETS:
        for init in ("champion", "imagenet"):
            for budget in BUDGETS:
                for seed in SEEDS:
                    torch.manual_seed(seed); np.random.seed(seed)
                    tr, te, n_cls = loaders(root, budget, seed)
                    model = build(init, n_cls)
                    zs = evaluate(model, te)[0] if budget < 1.0 else float("nan")
                    crit = nn.CrossEntropyLoss()
                    opt  = optim.Adam(model.classifier[1].parameters(), lr=1e-3)
                    model.train()
                    for _ in range(EPOCHS):
                        for X, y in tr:
                            X, y = X.to(DEVICE), y.to(DEVICE)
                            opt.zero_grad()
                            crit(model(X), y).backward()
                            opt.step()
                    acc, bal, maj = evaluate(model, te)
                    s = sil_after(model, te, seed) if budget < 1.0 else float("nan")
                    rows.append({"dataset": name, "init": init,
                                 "budget": budget, "seed": seed,
                                 "acc_zs": f"{zs:.4f}", "acc_ft": f"{acc:.4f}",
                                 "bal_ft": f"{bal:.4f}", "maj": f"{maj:.4f}",
                                 "sil_after": f"{s:.4f}"})
                    print(f"{name:9s} {init:9s} b={int(budget*100):3d}% "
                          f"seed={seed}  zs={zs:6.2f}  ft={acc:6.2f}  bal={bal:6.2f}  "
                          f"maj={maj:6.2f}  "
                          f"sil={s:.4f}", flush=True)
                    del model, opt
                    torch.cuda.empty_cache()

    with open("Results/exp10D_multiseed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    lines = [f"exp10D — multi-seed cross-domain (n={len(SEEDS)} seeds: {SEEDS})",
             "frozen backbone, head-only CE FT with class-balanced sampler, 5 epochs", ""]
    for name, _ in DATASETS:
        for init in ("champion", "imagenet"):
            for budget in BUDGETS:
                sel = [r for r in rows if r["dataset"] == name
                       and r["init"] == init and r["budget"] == budget]
                a = np.array([float(r["acc_ft"]) for r in sel])
                b = np.array([float(r["bal_ft"]) for r in sel])
                m = np.array([float(r["maj"]) for r in sel])
                line = (f"  {name:9s} {init:9s} {int(budget*100):3d}%: "
                        f"acc {a.mean():6.2f} ± {a.std(ddof=1):.2f}   "
                        f"bal {b.mean():6.2f} ± {b.std(ddof=1):.2f}   maj {m.mean():6.2f}")
                if budget < 1.0:
                    z = np.array([float(r["acc_zs"]) for r in sel])
                    s = np.array([float(r["sil_after"]) for r in sel])
                    line += (f"   zs {z.mean():6.2f} ± {z.std(ddof=1):.2f}"
                             f"   sil {s.mean():+.4f} ± {s.std(ddof=1):.4f}")
                lines.append(line)
        lines.append("")
    rep = "\n".join(lines)
    open("Results/exp10D_report.txt", "w", encoding="utf-8").write(rep)
    print("\n" + rep)
    print(f"wall {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
