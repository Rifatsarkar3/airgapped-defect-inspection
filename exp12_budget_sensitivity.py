# -*- coding: utf-8 -*-
"""
exp12_budget_sensitivity.py — data-budget sensitivity for cross-domain
fine-tuning of the casting-pretrained champion backbone.

Budgets {5, 10, 15, 20}% on NEU-DET / DAGM / Kolektor, CE-only head fine-tune
(5 epochs, frozen backbone), mirroring exp10's pipeline with one protocol
difference: the held-out test set is FIXED across budgets (the last 15% of the
SEED-42 permutation, identical to exp10's 100%-data condition), so accuracies
are comparable across the budget axis.

Outputs: Results/exp12_budget_sensitivity.csv, Results/exp12_report.txt
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import csv, os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS  = os.path.join("Models", "PGML_Champion_EfficientNetB0.pth")
SEED     = 42
EPOCHS   = 5
BATCH    = 32
BUDGETS  = [0.05, 0.10, 0.15, 0.20]
DATASETS = [
    ("NEU-DET",  "NEU-DET_ByClass"),
    ("DAGM",     "DAGM_Cleaned"),
    ("Kolektor", "Kolektor"),
]

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def loaders(root, fraction):
    ds  = datasets.ImageFolder(root, transform=TF)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(SEED)).tolist()
    n_te = max(1, int(0.15 * len(ds)))
    test_idx  = idx[-n_te:]                      # fixed across all budgets
    pool      = idx[:-n_te]
    n_tr      = max(1, int(fraction * len(ds)))
    train_idx = pool[:n_tr]
    tr = DataLoader(Subset(ds, train_idx), batch_size=BATCH, shuffle=True,
                    num_workers=4, pin_memory=True)
    te = DataLoader(Subset(ds, test_idx), batch_size=BATCH, shuffle=False,
                    num_workers=4, pin_memory=True)
    return tr, te, len(ds.classes)


def champion(n_cls):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    m.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE, weights_only=True))
    for p in m.parameters():
        p.requires_grad_(False)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    return m.to(DEVICE)


def evaluate(model, loader):
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            ok  += (model(X).argmax(1) == y).sum().item()
            tot += y.numel()
    return ok / tot * 100


def main():
    rows = []
    for name, root in DATASETS:
        for b in BUDGETS:
            t0 = time.time()
            torch.manual_seed(SEED)
            tr, te, n_cls = loaders(root, b)
            model = champion(n_cls)
            crit = nn.CrossEntropyLoss()
            opt  = optim.Adam(model.classifier[1].parameters(), lr=1e-3)
            model.train()
            for _ in range(EPOCHS):
                for X, y in tr:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    loss = crit(model(X), y)
                    loss.backward(); opt.step()
            acc = evaluate(model, te)
            rows.append({"dataset": name, "budget": b, "acc": acc})
            print(f"{name:9s} budget={int(b*100):3d}%  acc={acc:.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del model; torch.cuda.empty_cache()

    with open("Results/exp12_budget_sensitivity.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "budget", "acc"])
        w.writeheader(); w.writerows(rows)

    lines = ["exp12 — budget sensitivity (champion init, CE head FT, 5 epochs,",
             "fixed 15% held-out test per dataset, seed 42)", ""]
    for name, _ in DATASETS:
        vals = [r for r in rows if r["dataset"] == name]
        lines.append(f"  {name:9s}: " + "  ".join(
            f"{int(r['budget']*100)}%={r['acc']:.2f}%" for r in vals))
    rep = "\n".join(lines)
    open("Results/exp12_report.txt", "w", encoding="utf-8").write(rep)
    print("\n" + rep)


if __name__ == "__main__":
    main()
