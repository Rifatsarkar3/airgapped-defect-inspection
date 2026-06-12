"""
dagm_prepare.py
================
Restructures the DAGM 2007 dataset into a PyTorch ImageFolder-compatible
binary layout:

  DAGM_Cleaned/
    ok/          — defect-free images (Train + Test, labels excluded)
    defective/   — images that have a matching segmentation mask in Label/

Detection logic
  A `*_label.PNG` file in any Label/ sub-folder means the image whose
  stem matches (e.g. '0013_label.PNG' → '0013.PNG') is defective.
  All other images are ok.

Performance
  shutil.copy2 via a thread pool saturates the 6000 MT/s NVMe.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ──────────────────────────────────────────────────────────────
DAGM_ROOT   = "DAGM 2007-Industrial Defect Detection Dataset"
OUTPUT_DIR  = "DAGM_Cleaned"
MAX_WORKERS = 16
# ──────────────────────────────────────────────────────────────

_LABEL_RE = re.compile(r"^(.+)_label\.png$", re.IGNORECASE)


def collect_label_stems(dagm_root):
    """
    Walk every Label/ subfolder and return a set of stems that are defective.
    e.g.  'Label/0013_label.PNG'  →  stem '0013'
    """
    stems = set()
    for root, dirs, files in os.walk(dagm_root):
        if os.path.basename(root).lower() == "label":
            for f in files:
                m = _LABEL_RE.match(f)
                if m:
                    stems.add(m.group(1).lower())
    return stems


def collect_images(dagm_root):
    """
    Collect every .PNG that is NOT inside a Label/ folder.
    Returns a list of absolute paths.
    """
    images = []
    for root, dirs, files in os.walk(dagm_root):
        # Prune Label dirs so os.walk never descends into them
        dirs[:] = [d for d in dirs if d.lower() != "label"]
        for f in files:
            if f.lower().endswith(".png"):
                images.append(os.path.join(root, f))
    return images


def build_tasks(dagm_root, output_dir, label_stems):
    """
    Assign each image to ok/ or defective/ and return (src, dst) pairs.
    Files from Train/ and Test/ are prefixed with their split name so
    names from different splits can never collide.
    """
    tasks = {"ok": [], "defective": []}
    used  = set()

    for src in sorted(collect_images(dagm_root)):
        rel    = Path(src).relative_to(dagm_root)
        parts  = rel.parts                   # e.g. ('Train', '1158.PNG')
        prefix = parts[0] if len(parts) > 1 else ""
        fname  = parts[-1]
        stem   = Path(fname).stem.lower()

        # Unique destination filename
        dest_name = f"{prefix}_{fname}" if prefix else fname
        base, n = dest_name, 1
        while dest_name in used:
            dest_name = f"{Path(base).stem}_{n}{Path(base).suffix}"
            n += 1
        used.add(dest_name)

        cls = "defective" if stem in label_stems else "ok"
        tasks[cls].append((src, os.path.join(output_dir, cls, dest_name)))

    return tasks


def copy_file(src, dst):
    shutil.copy2(src, dst)
    return dst


def copy_class(cls_name, task_list):
    os.makedirs(os.path.dirname(task_list[0][1]), exist_ok=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(copy_file, s, d): d for s, d in task_list}
        copied  = sum(1 for _ in as_completed(futures))
    return copied


def main():
    print("=" * 65)
    print("  DAGM 2007  →  DAGM_Cleaned  (binary ImageFolder layout)")
    print("=" * 65)

    if not os.path.isdir(DAGM_ROOT):
        print(f"\n[ERROR] DAGM root not found:\n  {DAGM_ROOT}")
        raise SystemExit(1)

    print(f"\n[Source]   {DAGM_ROOT}")
    print(f"[Output]   {OUTPUT_DIR}")
    print(f"[Threads]  {MAX_WORKERS}  (shutil.copy2)\n")

    print("[Step 1/3] Collecting segmentation mask stems from Label/ folders …")
    label_stems = collect_label_stems(DAGM_ROOT)
    print(f"           {len(label_stems)} defective image stems identified")

    print("[Step 2/3] Building copy task list …")
    tasks = build_tasks(DAGM_ROOT, OUTPUT_DIR, label_stems)
    print(f"           ok={len(tasks['ok'])}  defective={len(tasks['defective'])}")

    print("[Step 3/3] Copying files …")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 0
    for cls in ("ok", "defective"):
        n = copy_class(cls, tasks[cls])
        print(f"  {cls:<12}  {n:>5} images  →  {OUTPUT_DIR}\\{cls}")
        total += n

    # ── Validation ────────────────────────────────────────────
    print("\n[Validation]")
    all_ok = True
    for cls in ("ok", "defective"):
        cls_path = os.path.join(OUTPUT_DIR, cls)
        n = len([f for f in os.listdir(cls_path) if f.lower().endswith(".png")])
        match = "OK" if n == len(tasks[cls]) else "MISMATCH"
        print(f"  {cls:<12}: {n:>5} images  [{match}]")
        if match != "OK":
            all_ok = False

    status = "PASSED" if all_ok else "FAILED — check warnings above"
    print(f"\n[Done]  {total} files copied  |  validation {status}")

    # ── Integration guidance ──────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  Update DATASET_REGISTRY in exp3_cross_domain_transfer.py:")
    print(f"{'─' * 65}")
    print(f"""
    DATASET_REGISTRY = [
        {{"name":"NEU-DET",  "root":"NEU-DET",  "color":"#8716D2"}},
        {{"name":"DAGM",     "root":r"{OUTPUT_DIR}",  "color":"#EA0CD0"}},
        {{"name":"Kolektor", "root":"Kolektor", "color":"#D41212"}},
    ]
""")
    print(f"{'=' * 65}")
    print("  ImageFolder will see 2 classes: ['defective', 'ok']")
    print("  This directly mirrors the exp1 binary casting-defect task.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
