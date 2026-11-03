"""
exp16_edge_backbone_profiling.py — preliminary latency and memory profiling of lightweight
attention-based (hybrid) edge backbones under the Experiment 6 deployment protocol.

Every backbone is profiled exactly as the champion in Experiment 6: FP16 weights, batch 32,
224x224 input, side-stream warm-up, CUDA Graph capture (eager FP16 if capture fails),
50 warm-up and 1,000 timed replays (repeated 3 times). Two allocator footprints are
recorded with torch.cuda.max_memory_allocated():

  isolated : model + activations + graph pool, nothing else resident
  governed : the same replay with the 900-frame ring-buffer proxy pre-allocated — the
             quantity that must stay below the 4,066 MB budget (only ~466 MB of which is
             left for the model once the 3,600 MiB proxy is reserved)

The isolated footprint is also measured at 448x448 to show how activation memory grows
with input resolution, which is what a static pre-allocation has to budget for.

Weights are random: throughput and allocator footprint depend on architecture and dtype,
not on weight values. This is checked directly by also profiling EfficientNet-B0 with the
two trained casting checkpoints. Accuracy is not measured here. EdgeViT has no timm
implementation; EdgeNeXt, EfficientFormer and FastViT represent the same lightweight
hybrid class.

Outputs: Results/exp16_edge_backbone_profiling.csv, Results/exp16_report.txt
Each backbone and resolution is measured in a fresh process, as Exp 6 profiles one model
per process. Run from the project root:
    python exp16_edge_backbone_profiling.py      (~10 min, idle GPU)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import csv
import gc
import json
import os
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import timm
from timm.utils.model import reparameterize_model

from exp6_tensorrt_profiling import (build_cuda_graph, allocate_ring_buffer, BATCH_SIZE,
                                     N_WARMUP, N_ITER, RING_FRAMES, VRAM_LIMIT_MB)

DEVICE  = torch.device("cuda")
REPEATS = 3
MIB     = 1024 ** 2
OUT_CSV = os.path.join("Results", "exp16_edge_backbone_profiling.csv")
OUT_TXT = os.path.join("Results", "exp16_report.txt")


def effnet(weights=None):
    m = tvm.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    if weights:
        m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return m


def mobilenet_v3():
    m = tvm.mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    return m


def hybrid(name):
    # inference form: fuse train-time branches (FastViT/RepViT-style); no-op elsewhere
    return lambda: reparameterize_model(
        timm.create_model(name, pretrained=False, num_classes=2).eval())


BACKBONES = [
    ("EfficientNet-B0 (random init)",         "CNN",    effnet),
    ("EfficientNet-B0 (Exp 6/8/9 checkpoint)", "CNN",
     lambda: effnet(os.path.join("Models", "PGML_Champion_EfficientNetB0.pth"))),
    ("EfficientNet-B0 (Gate-1 champion)",     "CNN",
     lambda: effnet(os.path.join("Models", "Gate1_Champion_EfficientNetB0.pth"))),
    ("MobileNet-V3-Large",                    "CNN",    mobilenet_v3),
    ("MobileViT-XXS",                         "hybrid", hybrid("mobilevit_xxs")),
    ("MobileViT-XS",                          "hybrid", hybrid("mobilevit_xs")),
    ("MobileViT-S",                           "hybrid", hybrid("mobilevit_s")),
    ("MobileViTv2-0.5",                       "hybrid", hybrid("mobilevitv2_050")),
    ("EdgeNeXt-XS",                           "hybrid", hybrid("edgenext_x_small")),
    ("EdgeNeXt-S",                            "hybrid", hybrid("edgenext_small")),
    ("EfficientFormer-L1",                    "hybrid", hybrid("efficientformer_l1")),
    ("FastViT-T8",                            "hybrid", hybrid("fastvit_t8")),
]


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def make_step(model, x):
    """CUDA Graph replay as in Exp 6; eager FP16 forward if capture is unsupported."""
    try:
        graph, _ = build_cuda_graph(model, x, DEVICE)
        return graph.replay, ""
    except Exception as e:
        free()

        def eager():
            with torch.no_grad():
                model(x)
        return eager, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"


def timed(step, n_iter=N_ITER):
    for _ in range(N_WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_one(index, res):
    """One backbone at one resolution, in its own process (as Exp 6 profiles one model
    per process): per-stream library workspaces and cached blocks from earlier models
    would otherwise accumulate in max_memory_allocated()."""
    label, family, build = BACKBONES[index]
    row = {"backbone": label, "family": family, "res": res}
    model = build().half().to(DEVICE).eval()
    row["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
    free()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    x = torch.randn(BATCH_SIZE, 3, res, res, device=DEVICE, dtype=torch.float16)
    step, row["capture_error"] = make_step(model, x)
    row["cuda_graph"] = not row["capture_error"]
    if res != 224:
        timed(step, n_iter=20)
        row["isolated_peak_mb"] = round(torch.cuda.max_memory_allocated(DEVICE) / MIB, 1)
        return row
    secs = [timed(step) for _ in range(REPEATS)]
    fps = np.array([BATCH_SIZE * N_ITER / s for s in secs])
    row["fps_mean"], row["fps_sd"] = round(float(fps.mean()), 1), round(float(fps.std()), 1)
    row["latency_ms_per_batch"] = round(float(np.mean(secs)) / N_ITER * 1000, 3)
    row["isolated_peak_mb"] = round(torch.cuda.max_memory_allocated(DEVICE) / MIB, 1)

    # governed footprint: ring-buffer proxy resident during replay, as in Exp 6
    ring = allocate_ring_buffer(RING_FRAMES, DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    timed(step, n_iter=200)
    gov = torch.cuda.max_memory_allocated(DEVICE) / MIB if ring is not None else float("nan")
    row["governed_peak_mb"] = round(gov, 1)
    row["governed_pass"] = bool(gov < VRAM_LIMIT_MB)
    return row


def gpu_state():
    q = "name,driver_version,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,memory.used"
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"nvidia-smi unavailable ({e})"


def child(index, res):
    cmd = [sys.executable, "-W", "ignore", os.path.abspath(__file__), "--one", str(index), str(res)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    tail = (out.stderr or out.stdout).strip().splitlines()[-1:] or ["no output"]
    return {"backbone": BACKBONES[index][0], "family": BACKBONES[index][1], "res": res,
            "error": tail[0][:200]}


def main():
    if "--one" in sys.argv:
        k = sys.argv.index("--one")
        print("RESULT " + json.dumps(run_one(int(sys.argv[k + 1]), int(sys.argv[k + 2]))))
        return

    env = [f"torch {torch.__version__}, timm {timm.__version__}, "
           f"cuDNN {torch.backends.cudnn.version()}", f"GPU at start: {gpu_state()}"]
    print("\n".join(env), flush=True)
    child(0, 224)                               # warm-up: GPU clocks ramp, result discarded

    rows, by_res = [], {}
    for i, (label, _, _) in enumerate(BACKBONES):
        for res in (224, 448):
            r = child(i, res)
            print(f"  {label:40s} {res}: {r}", flush=True)
            by_res[(label, res)] = r
        r = dict(by_res[(label, 224)])
        hi = by_res[(label, 448)]
        r["isolated_peak_mb_448"] = hi.get("isolated_peak_mb", hi.get("error", ""))
        r["capture_error_448"] = hi.get("capture_error", "")
        rows.append(r)
    env.append(f"GPU at end:   {gpu_state()}")

    fields = ["backbone", "family", "params_M", "cuda_graph", "fps_mean", "fps_sd",
              "latency_ms_per_batch", "isolated_peak_mb", "governed_peak_mb", "governed_pass",
              "isolated_peak_mb_448", "capture_error", "capture_error_448", "error"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    lines = ["exp16 — edge backbone profiling (FP16, batch 32, CUDA Graph as in Exp 6;",
             "one fresh process per backbone and resolution)", *env,
             f"budget {VRAM_LIMIT_MB} MB, ring buffer {RING_FRAMES} frames", ""]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['backbone']:40s} ERROR {r['error']}")
            continue
        lines.append(f"{r['backbone']:40s} {r['params_M']:6.2f}M  graph={r['cuda_graph']!s:5s} "
                     f"{r['fps_mean']:9.1f} ± {r['fps_sd']:6.1f} img/s  "
                     f"{r['latency_ms_per_batch']:7.3f} ms/batch  "
                     f"iso224 {r['isolated_peak_mb']:8.1f} MB  "
                     f"governed {r['governed_peak_mb']:8.1f} MB "
                     f"({'PASS' if r['governed_pass'] else 'FAIL'})  "
                     f"iso448 {r['isolated_peak_mb_448']}")
        for k in ("capture_error", "capture_error_448"):
            if r.get(k):
                lines.append(f"{'':40s} {k}: {r[k]}")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
