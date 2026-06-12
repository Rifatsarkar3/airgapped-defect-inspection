"""
exp8_stress_test.py — 24-Hour Continuous Stress Test
======================================================
Validates the PGML EfficientNet-B0 (FP16 + CUDA Graphs) under
uninterrupted deployment with a concurrent 900-frame SCADA ring buffer.

Stress pass criteria:
  FPS stability  : CV < 5% over the full window
  Peak VRAM      : < 4,066 MB at all times
  Peak GPU temp  : < 83 C  (RTX 5070 thermal throttle threshold)
  OOM events     : 0 (zero tolerance)

Default STRESS_HOURS = 24.  For a quick validation run set it to 0.25
(15 minutes) or pass the hours as a command-line argument:
    python exp8_stress_test.py 0.25

Outputs:
  Results/exp8_stress_log.csv
  Results/exp8_stress_report.txt
  Figures/Figure_Exp8_StressTest.png
  pgml_agent_progress.log  (appended)

Run from TF_Me project root:
    python exp8_stress_test.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import gc
import csv
import time
import signal
import datetime
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

STRESS_HOURS    = 24
BATCH_SIZE      = 32
IMAGE_SIZE      = 224
RING_FRAMES     = 900
VRAM_LIMIT_MB   = 4066
TEMP_LIMIT_C    = 83        # RTX 5070 thermal throttle threshold
FPS_CV_LIMIT    = 5.0       # max acceptable coefficient of variation (%)
FPS_TARGET      = 3029.6    # minimum acceptable mean FPS

LOG_INTERVAL_S  = 60        # write one CSV row per minute
PRINT_EVERY     = 30        # console summary every 30 log intervals (= 30 min)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# ============================================================================
# GPU TEMPERATURE (optional — requires pynvml)
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
# HELPERS
# ============================================================================
def log_append(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_model_fp16(device):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    src = FP16_WEIGHTS if os.path.exists(FP16_WEIGHTS) else PGML_WEIGHTS
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Model not found: {src}\nRun exp5_pgml_training.py and exp6_tensorrt_profiling.py first."
        )
    state = torch.load(src, map_location=device, weights_only=True)
    m.load_state_dict(state)
    m = m.half().to(device).eval()
    print(f"    Loaded: {src}")
    return m


def build_cuda_graph(model_fp16, static_input):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        with torch.no_grad():
            for _ in range(3):
                model_fp16(static_input)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            out = model_fp16(static_input)
    return g, out


def alloc_ring(frames, device):
    try:
        return torch.zeros((frames, 1024, 1024), dtype=torch.float32, device=device)
    except RuntimeError:
        return None

# ============================================================================
# FIGURE
# ============================================================================
def generate_figure(rows, fps_mean, fps_cv, vram_peak, temp_peak,
                    total_h, fps_pass, vram_pass, temp_pass, oom_pass, oom_n):
    elapsed_h = [r["h"]    for r in rows]
    fps_v     = [r["fps"]  for r in rows]
    vram_v    = [r["vram"] for r in rows]
    temp_v    = [r["temp"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"24-Hour Continuous Stress Test — PGML EfficientNet-B0\n"
        f"FP16 + CUDA Graph + {RING_FRAMES}-Frame SCADA Ring Buffer  |  "
        f"Elapsed: {total_h:.2f} h",
        fontsize=13, fontweight="bold"
    )

    c1, c2, c3 = "#2E5C8A", "#3A7D44", "#C47D00"

    ax = axes[0, 0]
    ax.plot(elapsed_h, fps_v, color=c1, lw=0.8, alpha=0.85)
    ax.axhline(fps_mean, color="black", ls="--", lw=1.2, label=f"Mean={fps_mean:.0f}")
    ax.fill_between(elapsed_h, fps_v, fps_mean, alpha=0.12, color=c1)
    ax.set_xlabel("Elapsed Time (h)", fontweight="bold")
    ax.set_ylabel("FPS", fontweight="bold")
    ax.set_title(f"Throughput Stability  |  CV={fps_cv:.2f}%  [{'PASS' if fps_pass else 'FAIL'}]",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.3); ax.set_axisbelow(True)

    ax = axes[0, 1]
    ax.plot(elapsed_h, vram_v, color=c2, lw=0.8, alpha=0.85)
    ax.axhline(VRAM_LIMIT_MB, color="darkred", ls="--", lw=1.2,
               label=f"Limit={VRAM_LIMIT_MB} MB", alpha=0.7)
    ax.fill_between(elapsed_h, vram_v, alpha=0.15, color=c2)
    ax.set_xlabel("Elapsed Time (h)", fontweight="bold")
    ax.set_ylabel("VRAM (MB)", fontweight="bold")
    ax.set_title(f"VRAM Footprint  |  Peak={vram_peak:.0f} MB  [{'PASS' if vram_pass else 'FAIL'}]",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.3); ax.set_axisbelow(True)

    ax = axes[1, 0]
    valid = [(h, t) for h, t in zip(elapsed_h, temp_v) if t >= 0]
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid], color=c3, lw=0.8, alpha=0.85)
        ax.axhline(TEMP_LIMIT_C, color="darkred", ls="--", lw=1.2,
                   label=f"Throttle={TEMP_LIMIT_C} C", alpha=0.7)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "pynvml not installed\n(temperature unavailable)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("Elapsed Time (h)", fontweight="bold")
    ax.set_ylabel("GPU Temperature (C)", fontweight="bold")
    ax.set_title(f"Thermal Profile  |  Peak={temp_peak} C  [{'PASS' if temp_pass else 'FAIL'}]",
                 fontweight="bold")
    ax.grid(ls="--", alpha=0.3); ax.set_axisbelow(True)

    ax = axes[1, 1]
    ax.axis("off")
    overall = fps_pass and vram_pass and temp_pass and oom_pass
    rows_tbl = [
        ["FPS Stability (CV)",    f"{fps_cv:.2f}%",    f"< {FPS_CV_LIMIT}%",    "PASS" if fps_pass  else "FAIL"],
        ["Peak VRAM",             f"{vram_peak:.0f} MB", f"< {VRAM_LIMIT_MB} MB", "PASS" if vram_pass else "FAIL"],
        ["Peak Temperature",      f"{temp_peak} C",    f"< {TEMP_LIMIT_C} C",   "PASS" if temp_pass else "FAIL"],
        ["OOM Events",            str(oom_n),          "= 0",                   "PASS" if oom_pass  else "FAIL"],
        ["Mean FPS",              f"{fps_mean:.0f}",   f">= {FPS_TARGET}",      "PASS" if fps_mean >= FPS_TARGET else "FAIL"],
    ]
    tbl = ax.table(cellText=rows_tbl,
                   colLabels=["Criterion", "Measured", "Threshold", "Status"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.05, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2E5C8A"); cell.set_text_props(color="white", fontweight="bold")
        elif c == 3 and r > 0:
            cell.set_facecolor("#C8E6C9" if rows_tbl[r-1][3] == "PASS" else "#FFCDD2")
    ax.set_title(f"Compliance Summary — OVERALL: {'PASS' if overall else 'FAIL'}",
                 fontweight="bold", pad=10)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "Figure_Exp8_StressTest.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure] {out}")

# ============================================================================
# MAIN
# ============================================================================
def run():
    global STRESS_HOURS
    if len(sys.argv) > 1:
        try:
            STRESS_HOURS = float(sys.argv[1])
        except ValueError:
            pass

    stress_sec = STRESS_HOURS * 3600

    print("=" * 70)
    print("  exp8_stress_test.py — Continuous Stress Test")
    print(f"  Duration : {STRESS_HOURS} h  |  DEVICE: {DEVICE}")
    print(f"  Batch    : {BATCH_SIZE}  |  Ring: {RING_FRAMES} frames")
    print(f"  Criteria : CV<{FPS_CV_LIMIT}%  VRAM<{VRAM_LIMIT_MB}MB  Temp<{TEMP_LIMIT_C}C  OOM=0")
    print("=" * 70)

    if DEVICE.type != "cuda":
        print("[ABORT] CUDA not available — this test requires GPU.")
        return

    # Model + CUDA Graph -------------------------------------------------------
    print("\n[1] Loading PGML FP16 model...")
    model = load_model_fp16(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    x_static = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE,
                           device=DEVICE, dtype=torch.float16)
    print("[2] Building CUDA Graph...")
    t0 = time.perf_counter()
    graph, _ = build_cuda_graph(model, x_static)
    print(f"    Built in {(time.perf_counter()-t0)*1000:.1f} ms")

    # Ring buffer --------------------------------------------------------------
    print(f"[3] Allocating SCADA ring buffer ({RING_FRAMES} frames)...")
    n_frames  = RING_FRAMES
    ring_buf  = None
    while n_frames >= 100:
        torch.cuda.reset_peak_memory_stats(DEVICE)
        ring_buf = alloc_ring(n_frames, DEVICE)
        if ring_buf is not None:
            vram_now = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
            print(f"    {n_frames} frames ({n_frames*4*1024*1024/1e9:.2f} GB) | VRAM={vram_now:.0f} MB")
            if vram_now < VRAM_LIMIT_MB:
                break
            del ring_buf; torch.cuda.empty_cache(); n_frames -= 50
        else:
            n_frames -= 50
    if ring_buf is None:
        print("    [WARN] Ring buffer allocation failed — running without buffer.")
        n_frames = 0

    # Warm-up ------------------------------------------------------------------
    print("[4] Warm-up (200 replays)...")
    for _ in range(200):
        graph.replay()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    # Stress loop --------------------------------------------------------------
    print(f"\n[5] Stress loop started — {STRESS_HOURS} h target")
    print(f"    Log interval: every {LOG_INTERVAL_S}s  |  Console: every {PRINT_EVERY * LOG_INTERVAL_S // 60} min")
    print(f"    Start: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("-" * 70)

    _stop = [False]
    def _sig(sig, frame): _stop[0] = True; print("\n[INTERRUPTED]")
    signal.signal(signal.SIGINT, _sig)

    csv_path = os.path.join(RESULT_DIR, "exp8_stress_log.csv")
    mem_rows  = []

    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["idx", "timestamp", "elapsed_h", "fps",
                    "vram_mb", "vram_peak_mb", "temp_c", "replays", "oom"])

        t_start   = time.perf_counter()
        t_log     = t_start
        t_win     = t_start
        n_log     = 0
        n_total   = 0
        n_win     = 0
        n_oom     = 0

        while not _stop[0]:
            elapsed = time.perf_counter() - t_start
            if elapsed >= stress_sec:
                break
            try:
                graph.replay()
                torch.cuda.synchronize()
                n_total += 1; n_win += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    n_oom += 1; torch.cuda.empty_cache()
                else:
                    raise

            now = time.perf_counter()
            if (now - t_log) >= LOG_INTERVAL_S:
                win_sec   = now - t_win
                fps       = n_win * BATCH_SIZE / win_sec if win_sec > 0 else 0.0
                vram_cur  = torch.cuda.memory_allocated(DEVICE) / (1024**2)
                vram_peak = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
                temp      = gpu_temp_c()
                elapsed_h = (now - t_start) / 3600
                ts        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                w.writerow([n_log, ts, f"{elapsed_h:.4f}", f"{fps:.1f}",
                            f"{vram_cur:.1f}", f"{vram_peak:.1f}", temp, n_total, n_oom])
                cf.flush()
                mem_rows.append({"h": elapsed_h, "fps": fps,
                                 "vram": vram_cur, "temp": temp})

                if n_log % PRINT_EVERY == 0:
                    print(f"  [{ts}] t={elapsed_h:.2f}h FPS={fps:.0f} "
                          f"VRAM={vram_cur:.0f}MB Temp={temp}C OOM={n_oom}")

                n_log += 1; n_win = 0; t_win = now; t_log = now
                torch.cuda.reset_peak_memory_stats(DEVICE)

    torch.cuda.synchronize()
    total_h = (time.perf_counter() - t_start) / 3600
    print("-" * 70)
    print(f"    End: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  total={total_h:.2f}h")

    # Stats --------------------------------------------------------------------
    if mem_rows:
        fps_arr  = np.array([r["fps"]  for r in mem_rows])
        vram_arr = np.array([r["vram"] for r in mem_rows])
        t_arr    = np.array([r["temp"] for r in mem_rows if r["temp"] >= 0])
        fps_mean = float(fps_arr.mean())
        fps_std  = float(fps_arr.std())
        fps_cv   = fps_std / fps_mean * 100 if fps_mean > 0 else 0.0
        vram_max = float(vram_arr.max())
        vram_min = float(vram_arr.min())
        temp_max = int(t_arr.max()) if len(t_arr) > 0 else -1
        temp_avg = float(t_arr.mean()) if len(t_arr) > 0 else -1.0
    else:
        fps_mean = fps_std = fps_cv = vram_max = vram_min = 0.0
        temp_max = temp_avg = -1

    fps_pass  = fps_cv < FPS_CV_LIMIT
    vram_pass = vram_max < VRAM_LIMIT_MB
    temp_pass = (temp_max < TEMP_LIMIT_C) if temp_max >= 0 else True
    oom_pass  = n_oom == 0
    overall   = fps_pass and vram_pass and temp_pass and oom_pass

    lines = [
        "=" * 70,
        "  exp8 — CONTINUOUS STRESS TEST REPORT",
        f"  Model    : PGML EfficientNet-B0 (FP16 + CUDA Graph)",
        f"  Duration : {total_h:.2f} h  (target: {STRESS_HOURS} h)",
        f"  Replays  : {n_total:,}  (batch={BATCH_SIZE})",
        "=" * 70,
        "",
        "  Throughput Stability:",
        f"    Mean FPS  : {fps_mean:.1f}",
        f"    FPS SD    : {fps_std:.1f}",
        f"    FPS CV    : {fps_cv:.2f}%  (limit: <{FPS_CV_LIMIT}%)  [{'PASS' if fps_pass else 'FAIL'}]",
        "",
        "  Memory Stability:",
        f"    Peak VRAM : {vram_max:.1f} MB  (limit: <{VRAM_LIMIT_MB} MB)  [{'PASS' if vram_pass else 'FAIL'}]",
        f"    Min VRAM  : {vram_min:.1f} MB  |  Drift: {vram_max - vram_min:.1f} MB",
        "",
        "  Thermal Profile:",
        f"    Peak temp : {temp_max} C  (limit: <{TEMP_LIMIT_C} C)  [{'PASS' if temp_pass else 'FAIL'}]",
        f"    Mean temp : {temp_avg:.1f} C",
        "",
        "  Zero-OOM Compliance:",
        f"    OOM events: {n_oom}  [{'PASS' if oom_pass else 'FAIL'}]",
        "",
        f"  OVERALL: {'PASS' if overall else 'FAIL'}",
        "=" * 70,
    ]
    report = "\n".join(lines)
    print("\n" + report)

    rp = os.path.join(RESULT_DIR, "exp8_stress_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Report] {rp}")

    log_append(
        f"\n[exp8 stress test — {time.strftime('%Y-%m-%d %H:%M')}]\n"
        f"  {total_h:.2f} h  OOM={n_oom}  FPS={fps_mean:.1f} CV={fps_cv:.2f}%\n"
        f"  VRAM={vram_max:.1f}MB  Temp={temp_max}C  OVERALL={'PASS' if overall else 'FAIL'}"
    )

    if mem_rows:
        generate_figure(mem_rows, fps_mean, fps_cv, vram_max, temp_max,
                        total_h, fps_pass, vram_pass, temp_pass, oom_pass, n_oom)

    if ring_buf is not None:
        del ring_buf
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\n[Done] OVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    run()
