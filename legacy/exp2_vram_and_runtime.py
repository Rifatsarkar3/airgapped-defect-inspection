"""
exp2_vram_and_runtime.py  —  Paper 1 Final Pipeline
=====================================================
24-hour continuous deployment validation.
Focal architecture: EfficientNet-B0  (the deployment champion)

Changes from v2
---------------
- Model swapped from ResNet-18 to MobileNet-V3 Large.
- Model updated to EfficientNet-B0 (excelled in every test across all datasets).
- Figure title and summary text updated to reflect the pivot.
- All logging/figure logic retained from v2 (pynvml, 3-panel figure,
  FPS / VRAM / temperature CSVs).

Install pynvml before running:
    pip install pynvml

Run from TF_Me project root:
    python exp2_vram_and_runtime.py
"""

import torch
import time
import threading
import math
import numpy as np
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv, os

# =====================================================================
# CONFIGURATION
# =====================================================================
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DURATION_HOURS = 24
BATCH_SIZE     = 32
LOG_INTERVAL   = 5000
FIG_DIR        = "Figures"
RESULT_DIR     = "Results"

os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

stop_stress = False

# =====================================================================
# pynvml SETUP
# =====================================================================
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle   = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_AVAILABLE = True
    print("[pynvml] GPU temperature monitoring active.")
except Exception as e:
    NVML_AVAILABLE = False
    print(f"[pynvml] Not available ({e}). Temperature column will be -1.")


def gpu_temp_c():
    if NVML_AVAILABLE:
        try:
            return pynvml.nvmlDeviceGetTemperature(
                _nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            return -1
    return -1


def current_vram_mb():
    if DEVICE.type == "cuda":
        return torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)
    return 0.0


# =====================================================================
# INDUSTRIAL STRESS WORKER
# =====================================================================
def industrial_stress_worker():
    """4 GB camera ring buffer + continuous SCADA CPU polling."""
    global stop_stress
    try:
        ring_buffer = torch.zeros(
            (1000, 1024, 1024), dtype=torch.float32, device=DEVICE)
        print(f"[Stress] Ring buffer allocated: "
              f"{ring_buffer.element_size() * ring_buffer.nelement() / 1e9:.2f} GB")
    except RuntimeError as e:
        print(f"[Stress] Ring buffer allocation failed: {e}")
        return

    while not stop_stress:
        for _ in range(50):
            math.factorial(200)
        time.sleep(0.01)


# =====================================================================
# FIGURE GENERATION
# =====================================================================
def generate_deployment_figure(log_path):
    try:
        import pandas as pd
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"[Figure] Could not read {log_path}: {e}")
        return

    if df.empty or len(df) < 2:
        print("[Figure] Not enough data.")
        return

    elapsed_h   = df["elapsed_h"].values
    fps_rolling = df["fps_rolling_mean"].values
    vram_mb     = df["vram_mb"].values
    temp_c      = df["gpu_temp_c"].values

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "24-Hour Continuous Deployment Validation — RTX 5070\n"
        "(EfficientNet-B0, Batch=32, 4 GB Simulated Ring Buffer + SCADA Load)",
        fontsize=13, fontweight="bold"
    )

    # Panel A — FPS
    axes[0].plot(elapsed_h, fps_rolling, color="#C47D00", linewidth=1.5)
    axes[0].axhline(y=60, color="gray", linestyle="--",
                    linewidth=1.0, alpha=0.7, label="Real-time threshold (60 FPS)")
    axes[0].set_ylabel("FPS (5k rolling mean)", fontweight="bold", fontsize=11)
    axes[0].set_title("Inference Throughput Stability — EfficientNet-B0",
                      fontweight="bold", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(linestyle="--", alpha=0.4)
    axes[0].set_axisbelow(True)
    mean_fps = fps_rolling[fps_rolling > 0].mean() if fps_rolling.any() else 0
    axes[0].text(0.98, 0.05, f"Global mean: {mean_fps:.1f} FPS",
                 transform=axes[0].transAxes, ha="right",
                 fontsize=9, color="#C47D00", fontweight="bold")

    # Panel B — VRAM
    axes[1].plot(elapsed_h, vram_mb, color="#3A7D44", linewidth=1.5)
    axes[1].axhline(y=12 * 1024, color="#B33A3A", linestyle="--",
                    linewidth=1.0, alpha=0.7, label="12 GB VRAM limit")
    axes[1].set_ylabel("VRAM in use (MB)", fontweight="bold", fontsize=11)
    axes[1].set_title("VRAM Footprint — Fragmentation Monitor",
                      fontweight="bold", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].set_axisbelow(True)
    peak_vram = vram_mb.max() if len(vram_mb) > 0 else 0
    axes[1].text(0.98, 0.92, f"Peak: {peak_vram:.0f} MB",
                 transform=axes[1].transAxes, ha="right",
                 fontsize=9, color="#3A7D44", fontweight="bold")

    # Panel C — Temperature
    if (temp_c > 0).any():
        axes[2].plot(elapsed_h, temp_c, color="#B33A3A", linewidth=1.5)
        axes[2].axhline(y=83, color="darkred", linestyle="--",
                        linewidth=1.0, alpha=0.7,
                        label="RTX 5070 thermal throttle (83°C)")
        axes[2].set_ylabel("GPU core temp (°C)", fontweight="bold", fontsize=11)
        axes[2].set_title("Thermal Stability", fontweight="bold", fontsize=11)
        axes[2].legend(fontsize=9)
        axes[2].grid(linestyle="--", alpha=0.4)
        axes[2].set_axisbelow(True)
        max_t = temp_c[temp_c > 0].max()
        axes[2].text(0.98, 0.92, f"Peak: {max_t:.0f}°C",
                     transform=axes[2].transAxes, ha="right",
                     fontsize=9, color="#B33A3A", fontweight="bold")
    else:
        axes[2].text(0.5, 0.5,
                     "Temperature data unavailable\n(install pynvml)",
                     ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=11, color="gray")
        axes[2].set_ylabel("GPU core temp (°C)", fontweight="bold", fontsize=11)

    axes[2].set_xlabel("Elapsed time (hours)", fontweight="bold", fontsize=11)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "Figure_Exp2_24H_Deployment_Validation.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Figure saved] {out}")


# =====================================================================
# MAIN 24-HOUR RUNTIME TEST
# =====================================================================
def run_continuous_inference():
    print("=" * 60)
    print("  exp2 — 24-Hour Deployment Validation")
    print(f"  Architecture  : EfficientNet-B0 (Deployment Champion)")
    print(f"  Device        : {DEVICE}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Log interval  : every {LOG_INTERVAL} cycles")
    print(f"  pynvml active : {NVML_AVAILABLE}")
    print("=" * 60)

    # ---- Build EfficientNet-B0 and load champion frozen weights ----
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, 2)
    champion_path = os.path.join("Models", "Champion_EfficientNetB0_Exp1.pth")
    if os.path.exists(champion_path):
        model.load_state_dict(torch.load(champion_path, map_location=DEVICE, weights_only=True))
        print(f"[Model] Loaded champion weights from {champion_path}")
    else:
        print(f"[Model] WARNING: {champion_path} not found — running with random weights.")
    model = model.to(DEVICE)
    model.eval()

    input_tensor = torch.randn(BATCH_SIZE, 3, 224, 224).to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    fps_log    = []
    oom_events = 0
    start_time = time.time()

    csv_path = os.path.join(RESULT_DIR, "exp2_deployment_log.csv")
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["elapsed_h", "cycle_count",
                     "fps_rolling_mean", "fps_rolling_std",
                     "vram_mb", "peak_vram_mb",
                     "gpu_temp_c", "oom_total"])

    stress_thread = threading.Thread(
        target=industrial_stress_worker, daemon=True)
    stress_thread.start()

    print(f"\n[Run] Inference loop started — {DURATION_HOURS} hours.\n")

    try:
        cycle_count = 0
        while (time.time() - start_time) < (DURATION_HOURS * 3600):
            t0 = time.time()
            try:
                with torch.no_grad():
                    _ = model(input_tensor)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                fps = BATCH_SIZE / (time.time() - t0)
                fps_log.append(fps)
                cycle_count += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_events += 1
                    torch.cuda.empty_cache()
                    print(f"[!] OOM #{oom_events} at cycle {cycle_count}")
                else:
                    raise

            if cycle_count > 0 and cycle_count % LOG_INTERVAL == 0:
                window    = fps_log[-LOG_INTERVAL:]
                elapsed_h = (time.time() - start_time) / 3600
                fps_mean  = float(np.mean(window))
                fps_std   = float(np.std(window))
                vram_cur  = current_vram_mb()
                vram_peak = (torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
                             if DEVICE.type == "cuda" else 0.0)
                temp      = gpu_temp_c()

                writer.writerow([
                    f"{elapsed_h:.4f}", cycle_count,
                    f"{fps_mean:.2f}", f"{fps_std:.2f}",
                    f"{vram_cur:.1f}", f"{vram_peak:.1f}",
                    temp, oom_events
                ])
                csv_file.flush()
                print(f"  t={elapsed_h:.2f}h | FPS={fps_mean:.1f}±{fps_std:.1f}"
                      f" | VRAM={vram_cur:.0f} MB | Temp={temp}°C"
                      f" | OOM={oom_events}")

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving partial results...")
    finally:
        stop_stress = True
        csv_file.close()
        if stress_thread.is_alive():
            stress_thread.join(timeout=5)

    elapsed_total = time.time() - start_time
    global_fps    = float(np.mean(fps_log)) if fps_log else 0
    global_std    = float(np.std(fps_log))  if fps_log else 0
    final_temp    = gpu_temp_c()
    peak_vram_mb  = (torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
                     if DEVICE.type == "cuda" else 0.0)

    summary = (
        f"\n{'='*60}\n"
        f"  exp2 — EfficientNet-B0 DEPLOYMENT VALIDATION RESULTS\n"
        f"{'='*60}\n"
        f"  Architecture       : EfficientNet-B0\n"
        f"  Total runtime      : {elapsed_total/3600:.2f} h\n"
        f"  Total cycles       : {cycle_count:,}\n"
        f"  OOM events         : {oom_events}  "
        f"{'(PASS)' if oom_events == 0 else '(FAIL)'}\n"
        f"  Global mean FPS    : {global_fps:.2f} (±{global_std:.2f})\n"
        f"  Peak VRAM used     : {peak_vram_mb:.1f} MB\n"
        f"  Final GPU temp     : {final_temp}°C\n"
        f"  Log                : {csv_path}\n"
        f"{'='*60}"
    )
    print(summary)

    with open(os.path.join(RESULT_DIR, "exp2_deployment_summary.txt"), "w") as f:
        f.write(summary)

    generate_deployment_figure(csv_path)


if __name__ == "__main__":
    run_continuous_inference()
