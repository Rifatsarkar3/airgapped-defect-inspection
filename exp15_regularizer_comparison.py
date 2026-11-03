"""
exp15_regularizer_comparison.py — SSF versus standard training regularizers under the
Experiment 5 paired protocol (n = 30 shared-seed cycles).

Every arm reuses exp11_paired_validation_v2.py verbatim for the data split, sampler order,
initialization, optimizer, epochs, validation-F1 epoch selection and the single test
evaluation. Each arm changes exactly one thing relative to the cross-entropy (CE) arm:

    label_smoothing : CE with label smoothing 0.1
    focal           : focal loss, gamma = 2 (no alpha term: class balance is already
                      handled by the weighted sampler)
    weight_decay    : AdamW with decoupled weight decay 0.01 (AdamW with zero decay is
                      the Adam optimizer of the CE arm)
    ce_repro        : the CE arm itself, re-run on the first 5 seeds, to measure run-to-run
                      noise against the original CE arm (Results/exp11v2_paired_results.csv);
                      seeds fix split, init and sampler, but GPU training is not bit-deterministic

Hyperparameters are standard literature defaults fixed before the run; none was tuned.
New arms are paired with the CE ("baseline") and SSF ("pgml") arms of
Results/exp11v2_paired_results.csv by seed (same initialization and data order).

Outputs (resume-safe: completed (cycle, arm) pairs are skipped on restart):
    Results/exp15_regularizer_results.csv
    Results/exp15_epoch_log.csv
    Results/exp15_report.txt

Run from the project root:
    python exp15_regularizer_comparison.py            # full run, ~11 h on an RTX 5070
    python exp15_regularizer_comparison.py --smoke    # 1 cycle, 2 epochs, 25 batches
    python exp15_regularizer_comparison.py --report   # rebuild the report from the CSVs
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import csv
import gc
import os
import time
import numpy as np
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import exp11_paired_validation_v2 as E   # same split, sampler, init, evaluate()

ARMS         = ["ce_repro", "label_smoothing", "focal", "weight_decay"]
N_REPRO      = 5
LABEL_SMOOTH = 0.1
FOCAL_GAMMA  = 2.0
WEIGHT_DECAY = 0.01

SUMMARY_CSV = os.path.join(E.RESULT_DIR, f"exp15_regularizer_results{E.SUFFIX}.csv")
EPOCH_CSV   = os.path.join(E.RESULT_DIR, f"exp15_epoch_log{E.SUFFIX}.csv")
REPORT_TXT  = os.path.join(E.RESULT_DIR, f"exp15_report{E.SUFFIX}.txt")
BASE_CSV    = os.path.join(E.RESULT_DIR, "exp11v2_paired_results.csv")


def focal_loss(logits, y, gamma=FOCAL_GAMMA):
    logp = F.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
    return (-(1.0 - logp.exp()) ** gamma * logp).mean()


# ============================================================================
# ONE ARM — exp11v2's run_arm with the arm's single change, identical seeding
# ============================================================================
def run_arm(arm, cycle, seed, epoch_writer, epoch_file):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, val, te = E.make_loaders(seed)
    torch.manual_seed(seed)
    model = E.build_model().to(E.DEVICE)

    if arm == "label_smoothing":
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    elif arm == "focal":
        criterion = focal_loss
    else:
        criterion = nn.CrossEntropyLoss()
    if arm == "weight_decay":
        opt = optim.AdamW(model.parameters(), lr=E.LR, weight_decay=WEIGHT_DECAY)
    else:
        opt = optim.Adam(model.parameters(), lr=E.LR)

    best_val_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, E.EPOCHS + 1):
        model.train()
        for n_batches, (X, y) in enumerate(tr, 1):
            X, y = X.to(E.DEVICE), y.to(E.DEVICE)
            opt.zero_grad()
            criterion(model(X), y).backward()
            opt.step()
            if E.MAX_BATCHES and n_batches >= E.MAX_BATCHES:
                break

        v_acc, v_rc, v_f1 = E.evaluate(model, val)
        if v_f1 > best_val_f1:
            best_val_f1, best_epoch = v_f1, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        epoch_writer.writerow([cycle, seed, arm, epoch,
                               f"{v_acc:.4f}", f"{v_rc:.4f}", f"{v_f1:.4f}"])
        epoch_file.flush()
        print(f"    [{arm:15s}] ep{epoch:02d}  val F1={v_f1:.4f}%", flush=True)

    model.load_state_dict(best_state)
    t_acc, t_rc, t_f1 = E.evaluate(model, te)
    del model, opt
    torch.cuda.empty_cache()
    gc.collect()
    return {"acc": t_acc, "recall": t_rc, "f1": t_f1,
            "best_epoch": best_epoch, "val_f1": best_val_f1}


# ============================================================================
# REPORT — paired against the exp11v2 CE arm, Holm-adjusted across enhancements
# ============================================================================
def holm(pvals):
    order, m = np.argsort(pvals), len(pvals)
    adj, running = np.empty(m), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[i]))
        adj[i] = running
    return adj


def write_report():
    if not os.path.exists(SUMMARY_CSV):
        print("[Report] no exp15 results yet")
        return
    arms = {}
    for path, keep in [(BASE_CSV, {"baseline": "CE", "pgml": "SSF"}),
                       (SUMMARY_CSV, {a: a for a in ARMS})]:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["arm"] in keep:
                arms.setdefault(keep[r["arm"]], {})[int(r["cycle"])] = r

    def vec(arm, key, cycles):
        return np.array([float(arms[arm][c][key]) for c in cycles])

    lines = ["=" * 78,
             "  exp15 — SSF vs standard regularizers, paired against the exp11v2 CE arm",
             f"  label_smoothing={LABEL_SMOOTH}  focal_gamma={FOCAL_GAMMA}  "
             f"AdamW weight_decay={WEIGHT_DECAY}  epochs={E.EPOCHS}",
             "  SD = sample SD (ddof=1), as in Tables 8-9; recall = defect catch rate",
             "=" * 78, ""]

    rep = sorted(set(arms.get("ce_repro", {})) & set(arms["CE"]))
    if rep:
        lines.append(f"  Reproducibility check: CE re-run vs original CE arm (cycles {rep})")
        for key in ("acc", "recall", "f1"):
            d = vec("ce_repro", key, rep) - vec("CE", key, rep)
            lines.append(f"    {key:6s} mean |delta| = {np.abs(d).mean():.4f} pp, "
                         f"max |delta| = {np.abs(d).max():.4f} pp")
        lines.append("")

    enh = [a for a in ("SSF", "label_smoothing", "focal", "weight_decay") if a in arms]
    for key, label in [("acc", "Accuracy"), ("recall", "Recall"), ("f1", "F1")]:
        rows, pvals = [], []
        for a in enh:
            cyc = sorted(set(arms[a]) & set(arms["CE"]))
            if len(cyc) < 3:
                continue
            x, b = vec(a, key, cyc), vec("CE", key, cyc)
            d = x - b
            t, p = stats.ttest_rel(x, b)
            try:
                pw = stats.wilcoxon(x, b).pvalue
            except ValueError:
                pw = float("nan")
            ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
            pm = stats.pearsonr(x + b, x - b)[1]          # Pitman-Morgan paired variance
            rows.append((a, len(cyc), x.mean(), x.std(ddof=1), b.mean(), b.std(ddof=1),
                         d.mean(), ci, t, p, pw, pm))
            pvals.append(p)
        if not rows:
            continue
        adj = holm(np.array(pvals))
        lines.append(f"  {label}")
        for (a, n, xm, xs, bm, bs, dm, ci, t, p, pw, pm), ph in zip(rows, adj):
            lines.append(f"    {a:16s} n={n:2d}  {xm:.3f}% ± {xs:.3f}  vs CE {bm:.3f}% ± {bs:.3f}"
                         f"  delta {dm:+.4f} pp [95% CI {ci[0]:+.3f}, {ci[1]:+.3f}]"
                         f"  t={t:.3f} p={p:.3f} (Holm {ph:.3f}, Wilcoxon {pw:.3f})"
                         f"  Pitman-Morgan p={pm:.3f}")
        lines.append("")

    report = "\n".join(lines + ["=" * 78])
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report + f"\n[Report saved] {REPORT_TXT}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    if "--report" in sys.argv:
        write_report()
        return

    print("=" * 78)
    print(f"  exp15_regularizer_comparison.py  arms={ARMS}  cycles={E.N_CYCLES}  "
          f"epochs={E.EPOCHS}" + ("  [SMOKE]" if E.SMOKE else ""))
    print("=" * 78, flush=True)

    fresh = not os.path.exists(SUMMARY_CSV) or os.path.getsize(SUMMARY_CSV) == 0
    done = set() if fresh else {(int(r["cycle"]), r["arm"])
                                for r in csv.DictReader(open(SUMMARY_CSV, encoding="utf-8"))}
    if done:
        print(f"[Resume] {len(done)} completed (cycle, arm) entries found")
    summary_file = open(SUMMARY_CSV, "a", newline="", encoding="utf-8")
    summary_writer = csv.writer(summary_file)
    if fresh:
        summary_writer.writerow(["cycle", "seed", "arm", "acc", "recall", "f1",
                                 "best_epoch", "best_val_f1"])
    epoch_fresh = not os.path.exists(EPOCH_CSV) or os.path.getsize(EPOCH_CSV) == 0
    if not epoch_fresh:   # an interrupted arm restarts at epoch 1; drop its partial rows
        rows = list(csv.reader(open(EPOCH_CSV, newline="", encoding="utf-8")))
        with open(EPOCH_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows[:1] + [r for r in rows[1:] if (int(r[0]), r[2]) in done])
    epoch_file = open(EPOCH_CSV, "a", newline="", encoding="utf-8")
    epoch_writer = csv.writer(epoch_file)
    if epoch_fresh:
        epoch_writer.writerow(["cycle", "seed", "arm", "epoch",
                               "val_acc", "val_recall", "val_f1"])

    t_wall = time.time()
    for i, seed in enumerate(E.CYCLE_SEEDS):
        cycle = i + 1
        print(f"\n--- Cycle {cycle:02d}/{E.N_CYCLES}  seed={seed} ---", flush=True)
        for arm in ARMS:
            if (arm == "ce_repro" and cycle > N_REPRO) or (cycle, arm) in done:
                continue
            t0 = time.time()
            res = run_arm(arm, cycle, seed, epoch_writer, epoch_file)
            summary_writer.writerow([cycle, seed, arm, f"{res['acc']:.4f}",
                                     f"{res['recall']:.4f}", f"{res['f1']:.4f}",
                                     res["best_epoch"], f"{res['val_f1']:.4f}"])
            summary_file.flush()
            print(f"    [{arm:15s}] test Acc={res['acc']:.4f}%  Recall={res['recall']:.4f}%  "
                  f"F1={res['f1']:.4f}%  (best ep{res['best_epoch']}, "
                  f"{(time.time() - t0) / 60:.1f} min)", flush=True)

    summary_file.close()
    epoch_file.close()
    write_report()
    print(f"\n[Done] Wall time: {(time.time() - t_wall) / 3600:.2f}h")


if __name__ == "__main__":
    main()
