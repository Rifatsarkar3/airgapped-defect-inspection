# -*- coding: utf-8 -*-
"""
exp10c_neudet_fix.py — corrected NEU-DET cross-domain arm.

The legacy exp3/exp10 NEU-DET runs pointed ImageFolder at the dataset's
train/validation split folders, so the learned task was split-membership
(majority class = 80%), not defect classification. This script reruns
NEU-DET as its native 6-class task from NEU-DET_ByClass/ for BOTH
initializations (casting-pretrained champion, ImageNet), mirroring exp10's
protocol (seed 42 permutation, frozen backbone, head-only Adam 1e-3,
5 epochs, 15% and 100% budgets, silhouette + t-SNE on champion arm).

Outputs: Results/exp10C_neudet.csv, Results/exp10R_tsne_NEUDET.npz (replaces
the void dump so the Fig 9 builder runs unchanged).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import csv, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from exp10d_multiseed import balanced_sampler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT    = "NEU-DET_ByClass"
WEIGHTS = os.path.join("Models", "Gate1_Champion_EfficientNetB0.pth")
SEED    = 42
EPOCHS  = 5
BATCH   = 32
FEAT_LIMIT = 700

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def loaders(fraction):
    ds  = datasets.ImageFolder(ROOT, transform=TF)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED)).tolist()
    if fraction >= 1.0:
        n_te = max(1, int(0.15 * len(ds)))
        tr_idx, te_idx = idx[:-n_te], idx[-n_te:]
    else:
        n_tr = int(fraction * len(ds))
        tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
    tr = DataLoader(Subset(ds, tr_idx), batch_size=BATCH,
                    sampler=balanced_sampler(ds, tr_idx, SEED),
                    num_workers=4, pin_memory=True)
    te = DataLoader(Subset(ds, te_idx), batch_size=BATCH, shuffle=False,
                    num_workers=4, pin_memory=True)
    return tr, te, ds.classes


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
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            ok += (model(X).argmax(1) == y).sum().item(); tot += y.numel()
    return ok / tot * 100


def features(model, loader):
    model.eval(); feats, lbls = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            f = model.avgpool(model.features(X)).flatten(1)
            feats.append(f.cpu().numpy()); lbls.append(y.numpy())
            if sum(len(a) for a in lbls) >= FEAT_LIMIT:
                break
    return np.concatenate(feats)[:FEAT_LIMIT], np.concatenate(lbls)[:FEAT_LIMIT]


def sil(f, l):
    return float(silhouette_score(f, l, sample_size=min(3000, len(f)), random_state=SEED))


def tsne(f):
    return TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca").fit_transform(f)


def finetune(model, tr):
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.classifier[1].parameters(), lr=1e-3)
    model.train()
    for ep in range(EPOCHS):
        for X, y in tr:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(X), y).backward(); opt.step()
    return model


def main():
    rows = []
    keep = {}
    for init in ("champion", "imagenet"):
        torch.manual_seed(SEED); np.random.seed(SEED)
        tr, te, classes = loaders(0.15)
        model = build(init, len(classes))
        acc_zs = evaluate(model, te)
        f_b, l_b = features(model, te)
        sil_b = sil(f_b, l_b)
        tsne_b = tsne(f_b) if init == "champion" else None
        model = finetune(model, tr)
        acc_ft = evaluate(model, te)
        f_a, l_a = features(model, te)
        sil_a = sil(f_a, l_a)
        tsne_a = tsne(f_a) if init == "champion" else None
        # 100% condition
        torch.manual_seed(SEED)
        tr100, te100, _ = loaders(1.0)
        m100 = finetune(build(init, len(classes)), tr100)
        acc_100 = evaluate(m100, te100)
        rows.append({"init": init, "acc_zero_shot": f"{acc_zs:.4f}",
                     "acc_ft_15pct": f"{acc_ft:.4f}", "acc_full_100pct": f"{acc_100:.4f}",
                     "sil_before": f"{sil_b:.4f}", "sil_after": f"{sil_a:.4f}"})
        print(f"{init:9s} ZS={acc_zs:.2f}%  FT15={acc_ft:.2f}%  FT100={acc_100:.2f}%  "
              f"silB={sil_b:.4f}  silA={sil_a:.4f}", flush=True)
        if init == "champion":
            keep = dict(labels=l_b, tsne_b=tsne_b, tsne_a=tsne_a,
                        acc_b=acc_zs, acc_a=acc_ft, sil_b=sil_b, sil_a=sil_a,
                        class_names=np.array(classes, dtype=object))
        del model, m100; torch.cuda.empty_cache()

    with open("Results/exp10C_neudet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.savez("Results/exp10R_tsne_NEUDET.npz", **keep)
    print("saved CSV + npz")


if __name__ == "__main__":
    main()
