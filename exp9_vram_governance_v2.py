"""
exp9_vram_governance_v2.py — VRAM Governance Ablation, factorial redesign (Item 5 fix)
=======================================================================================
Supersedes exp9_vram_governance.py. The original 3-condition ablation confounded
four factors at once (precision, batch size 128->64->32, CUDA Graph, and the
SCADA-contention mechanism), so FPS/VRAM deltas between conditions could not be
attributed to any single governance stage.

Fix: 4 conditions, batch size FIXED at 32 (the deployment batch size) throughout,
changing exactly one factor per step:

  A  FP32,  no CUDA Graph, dynamic SCADA alloc      (baseline)
  B  FP16,  no CUDA Graph, dynamic SCADA alloc       <- isolates precision   (A->B)
  C  FP16,  CUDA Graph,    dynamic SCADA alloc       <- isolates CUDA Graph  (B->C)
  D  FP16,  CUDA Graph,    pre-allocated 900-frame ring, no dynamic SCADA
                                                       <- isolates the SCADA memory-
                                                          management strategy (C->D):
                                                          dynamic allocation vs.
                                                          pre-allocated ring buffer
                                                          (mutually exclusive by
                                                          construction, so this is one
                                                          factor, not two)

Each condition runs for CONDITION_MINUTES (default 60 min); pass minutes as argv[1]
for a quick validation run, e.g. `python exp9_vram_governance_v2.py 2`.

Outputs:
  Results/exp9v2_governance_raw.csv
  Results/exp9v2_governance_report.txt
  Figures/Figure_Exp9v2_VRAM_Governance.png
  pgml_agent_progress.log  (appended)
"""

import sys
import json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import gc
import csv
import time
import math
import threading
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR       = "Models"
RESULT_DIR      = "Results"
FIG_DIR         = "Figures"
LOG_FILE        = "pgml_agent_progress.log"

PGML_WEIGHTS    = os.path.join(MODEL_DIR, "PGML_Champion_EfficientNetB0.pth")
FP16_WEIGHTS    = os.path.join(MODEL_DIR, "PGML_EfficientNetB0_fp16.pth")

CONDITION_MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 60
LOG_INTERVAL      = 1000
VRAM_LIMIT_MB     = 4066
FPS_TARGET        = 3029.6
IMAGE_SIZE        = 224

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# ============================================================================
# NVML (optional)
# ============================================================================
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_hdl      = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

def gpu_temp_c():
    if NVML_AVAILABLE:
        try:
            return pynvml.nvmlDeviceGetTemperature(_nvml_hdl, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            return -1
    return -1

# ============================================================================
# CONDITIONS — batch size fixed at 32 throughout; one factor changes per step
# ============================================================================
CONDITIONS = [
    {
        "label":       "A — FP32 Baseline",
        "batch":       32,
        "fp16":        False,
        "cuda_graph":  False,
        "ring_frames": 0,
        "scada_alloc": True,
        "description": "float32, batch=32, no CUDA Graph, dynamic SCADA alloc",
        "color":       "#E24B4A",
    },
    {
        "label":       "B — + FP16",
        "batch":       32,
        "fp16":        True,
        "cuda_graph":  False,
        "ring_frames": 0,
        "scada_alloc": True,
        "description": "float16, batch=32, no CUDA Graph, dynamic SCADA alloc",
        "color":       "#BA7517",
    },
    {
        "label":       "C — + CUDA Graph",
        "batch":       32,
        "fp16":        True,
        "cuda_graph":  True,
        "ring_frames": 0,
        "scada_alloc": True,
        "description": "float16, batch=32, CUDA Graph, dynamic SCADA alloc",
        "color":       "#1F7A8C",
    },
    {
        "label":       "D — + Ring Buffer (Full Governance)",
        "batch":       32,
        "fp16":        True,
        "cuda_graph":  True,
        "ring_frames": 900,
        "scada_alloc": False,
        "description": "float16, batch=32, CUDA Graph, 900-frame pre-alloc ring",
        "color":       "#2E7D32",
    },
]

# ============================================================================
# HELPERS
# ============================================================================
def log_append(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_model(fp16, device):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    src = FP16_WEIGHTS if (fp16 and os.path.exists(FP16_WEIGHTS)) else PGML_WEIGHTS
    if not os.path.exists(src):
        raise FileNotFoundError(f"Model not found: {src}")
    state = torch.load(src, map_location=device, weights_only=True)
    m.load_state_dict(state)
    if fp16:
        m = m.half()
    return m.to(device).eval()


def build_cuda_graph(model, static_input):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        with torch.no_grad():
            for _ in range(3):
                model(static_input)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            model(static_input)
    return g


def make_scada_worker(stop_event):
    """SCADA stress: dynamically allocate and free large CUDA tensors."""
    def _worker():
        while not stop_event.is_set():
            try:
                t = torch.zeros((256, 512, 512), device=DEVICE)
                for _ in range(20):
                    math.factorial(200)
                del t
            except RuntimeError:
                pass
            time.sleep(0.02)
    return _worker

# ============================================================================
# RUN ONE CONDITION
# ============================================================================
def run_condition(cond, duration_s, csv_writer):
    label, bs = cond["label"], cond["batch"]
    fp16, use_graph, n_ring = cond["fp16"], cond["cuda_graph"], cond["ring_frames"]

    print(f"\n  {'='*58}")
    print(f"  {label}")
    print(f"  {cond['description']}")
    print(f"  {'='*58}")

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)
    gc.collect()

    model = load_model(fp16, DEVICE)

    dtype   = torch.float16 if fp16 else torch.float32
    x_input = torch.randn(bs, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE, dtype=dtype)

    graph_obj  = None
    ring_buf   = None
    ring_alloc = n_ring

    if use_graph and DEVICE.type == "cuda":
        graph_obj = build_cuda_graph(model, x_input)
        print(f"  [CUDA Graph] Built")

    if n_ring > 0 and DEVICE.type == "cuda":
        while ring_alloc >= 100:
            try:
                ring_buf = torch.zeros((ring_alloc, 1024, 1024), dtype=torch.float32, device=DEVICE)
                vram_now = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
                print(f"  [Ring buffer] {ring_alloc} frames | VRAM so far: {vram_now:.0f} MB")
                if vram_now < VRAM_LIMIT_MB:
                    break
                del ring_buf; torch.cuda.empty_cache(); ring_alloc -= 50
            except RuntimeError:
                ring_alloc -= 50
        if ring_buf is None:
            print("  [Ring buffer] Allocation failed — running without buffer.")

    stop_event = threading.Event()
    scada_t    = None
    if cond["scada_alloc"] and DEVICE.type == "cuda":
        scada_t = threading.Thread(target=make_scada_worker(stop_event), daemon=True)
        scada_t.start()
        print("  [SCADA thread] Started (dynamic allocator)")

    fps_log   = []
    oom_count = 0
    cycle     = 0
    t_start   = time.time()

    try:
        while (time.time() - t_start) < duration_s:
            t0 = time.time()
            try:
                if graph_obj is not None:
                    graph_obj.replay()
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                else:
                    with torch.no_grad():
                        model(x_input)
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                fps_log.append(bs / (time.time() - t0))
                cycle += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_count += 1
                    torch.cuda.empty_cache()
                    print(f"  [OOM #{oom_count}] cycle {cycle}")
                else:
                    raise

            if cycle > 0 and cycle % LOG_INTERVAL == 0:
                w  = fps_log[-LOG_INTERVAL:]
                el = (time.time() - t_start) / 60
                vc = torch.cuda.memory_allocated(DEVICE) / (1024**2) if DEVICE.type == "cuda" else 0.0
                csv_writer.writerow([label, cycle, f"{el:.2f}", f"{np.mean(w):.1f}",
                                     f"{np.std(w):.1f}", f"{vc:.1f}", gpu_temp_c(), oom_count])
    finally:
        stop_event.set()
        if scada_t and scada_t.is_alive():
            scada_t.join(timeout=8)

    peak_vram = (torch.cuda.max_memory_allocated(DEVICE) / (1024**2)) if DEVICE.type == "cuda" else 0.0
    fps_arr   = np.array(fps_log) if fps_log else np.array([0.0])
    result    = {
        "label":        label,
        "description":  cond["description"],
        "color":        cond["color"],
        "batch":        bs,
        "fp16":         fp16,
        "cuda_graph":   use_graph,
        "ring_frames":  ring_alloc if ring_buf is not None else 0,
        "oom_events":   oom_count,
        "fps_mean":     float(fps_arr.mean()),
        "fps_std":      float(fps_arr.std()),
        "fps_cv":       float(fps_arr.std() / fps_arr.mean() * 100) if fps_arr.mean() > 0 else 0.0,
        "peak_vram_mb": peak_vram,
        "cycles":       cycle,
    }
    print(f"  Result: OOM={oom_count}  FPS={result['fps_mean']:.1f}+/-{result['fps_std']:.1f}"
          f"  VRAM={peak_vram:.0f}MB")

    for obj in [model, x_input, ring_buf, graph_obj, fps_log]:
        try:
            del obj
        except Exception:
            pass
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result

# ============================================================================
# FIGURE
# ============================================================================
def generate_figure(results):
    labels   = [r["label"]        for r in results]
    colors   = [r["color"]        for r in results]
    oom      = [r["oom_events"]   for r in results]
    fps_mean = [r["fps_mean"]     for r in results]
    fps_std  = [r["fps_std"]      for r in results]
    fps_cv   = [r["fps_cv"]       for r in results]
    vram     = [r["peak_vram_mb"] for r in results]
    x        = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "VRAM Governance Ablation v2 — EfficientNet-B0 (batch=32 throughout)\n"
        "A (FP32) -> B (+FP16) -> C (+CUDA Graph) -> D (+Ring Buffer, Full Governance)",
        fontsize=13, fontweight="bold"
    )

    def bar_panel(ax, vals, title, ylabel, fmt="{:.1f}", y_margin=1.3):
        bars = ax.bar(x, vals, color=colors, edgecolor="black", lw=0.7, width=0.5)
        ax.set_title(title, fontweight="bold", fontsize=11, pad=8)
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", ls="--", alpha=0.4); ax.set_axisbelow(True)
        ymax = max(vals) if max(vals) > 0 else 1
        ax.set_ylim(0, ymax * y_margin + 0.01)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ymax * 0.02,
                    fmt.format(val), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        bars[-1].set_edgecolor("black"); bars[-1].set_linewidth(2.5)

    bar_panel(axes[0, 0], oom, f"OOM Events ({CONDITION_MINUTES:.0f}-min window)", "OOM count", "{:.0f}")

    bars = axes[0, 1].bar(x, fps_mean, color=colors, edgecolor="black", lw=0.7, width=0.5,
                          yerr=fps_std, capsize=6, ecolor="black", error_kw={"elinewidth":1.2})
    axes[0, 1].set_title("Mean FPS +/- SD", fontweight="bold", fontsize=11, pad=8)
    axes[0, 1].set_ylabel("Frames per second", fontweight="bold", fontsize=10)
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(labels, fontsize=8)
    axes[0, 1].axhline(FPS_TARGET, color="darkred", ls="--", lw=1.2, alpha=0.7,
                        label=f"Target={FPS_TARGET:.0f}")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(axis="y", ls="--", alpha=0.4); axes[0, 1].set_axisbelow(True)
    for bar, m, s in zip(bars, fps_mean, fps_std):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, m + s + max(fps_mean)*0.02,
                        f"{m:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    bars[-1].set_edgecolor("black"); bars[-1].set_linewidth(2.5)

    bar_panel(axes[1, 0], fps_cv, "FPS Coefficient of Variation",
              "CV = SD / Mean x 100 (%)", "{:.2f}%")
    axes[1, 0].axhline(5.0, color="darkred", ls="--", lw=1.2, alpha=0.7, label="CV limit=5%")
    axes[1, 0].legend(fontsize=9)

    bar_panel(axes[1, 1], vram, f"Peak VRAM  (limit: {VRAM_LIMIT_MB} MB)", "Peak VRAM (MB)", "{:.0f} MB")
    axes[1, 1].axhline(VRAM_LIMIT_MB, color="darkred", ls="--", lw=1.2, alpha=0.7,
                        label=f"Limit={VRAM_LIMIT_MB} MB")
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "Figure_Exp9v2_VRAM_Governance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure] {out}")

# ============================================================================
# MAIN
# ============================================================================
def run():
    duration_s = CONDITION_MINUTES * 60

    print("=" * 65)
    print("  exp9_vram_governance_v2.py — Factorial VRAM Governance Ablation")
    print(f"  DEVICE: {DEVICE}  |  Each condition: {CONDITION_MINUTES} min  |  batch=32 fixed")
    print("=" * 65)

    csv_path  = os.path.join(RESULT_DIR, "exp9v2_governance_raw.csv")
    ckpt_path = os.path.join(RESULT_DIR, "exp9v2_conditions_checkpoint.json")

    # ---- Resume support ---------------------------------------------------
    # A condition's peak-VRAM figure comes from an in-process CUDA counter
    # (reset per condition) that cannot be reliably reconstructed after the
    # fact from the periodic CSV samples, so the unit of resume is a whole
    # condition: a condition only counts as "done" once run_condition() has
    # returned and its exact result dict has been checkpointed to disk. Any
    # raw CSV rows left over from a condition that was killed mid-run (i.e.
    # not in the checkpoint) belong to an incomplete attempt and are dropped
    # before that condition is re-run from scratch.
    done_results = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done_results = json.load(f)
        print(f"[Resume] {len(done_results)} condition(s) already complete: "
              f"{', '.join(done_results)}")

    kept_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            kept_rows = [row for row in csv.DictReader(f) if row["condition"] in done_results]

    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["condition", "cycle", "elapsed_min", "fps_mean", "fps_std",
                    "vram_mb", "temp_c", "oom_total"])
        for row in kept_rows:
            w.writerow([row["condition"], row["cycle"], row["elapsed_min"], row["fps_mean"],
                        row["fps_std"], row["vram_mb"], row["temp_c"], row["oom_total"]])
        cf.flush()

        all_results = []
        for cond in CONDITIONS:
            label = cond["label"]
            if label in done_results:
                r = done_results[label]
                print(f"  {label}... [skipped — already complete: "
                      f"FPS={r['fps_mean']:.1f} VRAM={r['peak_vram_mb']:.0f}MB]")
            else:
                r = run_condition(cond, duration_s, w)
                cf.flush()
                done_results[label] = r
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(done_results, f, indent=2)
            all_results.append(r)

    # Report -------------------------------------------------------------------
    lines = [
        "=" * 70,
        "  exp9v2 — FACTORIAL VRAM GOVERNANCE ABLATION REPORT (batch=32 fixed)",
        "=" * 70,
        f"  {'Condition':<38} {'OOM':>5} {'FPS mean':>10} {'FPS SD':>8} {'CV%':>7} {'Peak VRAM':>12}",
        "  " + "-" * 76,
    ]
    for r in all_results:
        vp   = r["peak_vram_mb"]
        vstr = f"{vp:.0f} MB {'PASS' if vp < VRAM_LIMIT_MB else 'FAIL'}"
        lines.append(f"  {r['label']:<38} {r['oom_events']:>5} {r['fps_mean']:>10.1f}"
                     f" {r['fps_std']:>8.1f} {r['fps_cv']:>7.2f} {vstr:>20}")

    lines += ["", "  Isolated single-factor deltas:"]
    A, B, C, D = all_results
    lines.append(f"    A->B  (+FP16)        : FPS {B['fps_mean']-A['fps_mean']:+.1f}   VRAM {B['peak_vram_mb']-A['peak_vram_mb']:+.1f} MB")
    lines.append(f"    B->C  (+CUDA Graph)   : FPS {C['fps_mean']-B['fps_mean']:+.1f}   VRAM {C['peak_vram_mb']-B['peak_vram_mb']:+.1f} MB")
    lines.append(f"    C->D  (+Ring Buffer)  : FPS {D['fps_mean']-C['fps_mean']:+.1f}   VRAM {D['peak_vram_mb']-C['peak_vram_mb']:+.1f} MB")

    lines += ["", "  Compliance (Condition D — Full Governance):"]
    d = all_results[3]
    lines.append(f"    OOM events : {d['oom_events']}  [{'PASS' if d['oom_events']==0 else 'FAIL'}]")
    lines.append(f"    Peak VRAM  : {d['peak_vram_mb']:.0f} MB  [{'PASS' if d['peak_vram_mb']<VRAM_LIMIT_MB else 'FAIL'}]")
    lines.append(f"    Mean FPS   : {d['fps_mean']:.1f}  [{'PASS' if d['fps_mean']>=FPS_TARGET else 'FAIL'}]")
    lines.append(f"    FPS CV     : {d['fps_cv']:.2f}%  [{'PASS' if d['fps_cv']<5.0 else 'FAIL'}]")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print("\n" + report)

    rp = os.path.join(RESULT_DIR, "exp9v2_governance_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Report] {rp}")

    log_append(
        f"\n[exp9v2 factorial governance — {time.strftime('%Y-%m-%d %H:%M')}]\n"
        + "\n".join(
            f"  {r['label']}: OOM={r['oom_events']} FPS={r['fps_mean']:.1f}"
            f" CV={r['fps_cv']:.2f}% VRAM={r['peak_vram_mb']:.0f}MB"
            for r in all_results
        )
    )

    generate_figure(all_results)
    print("\n[Done]")


if __name__ == "__main__":
    run()
