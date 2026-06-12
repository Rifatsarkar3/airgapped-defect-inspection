# -*- coding: utf-8 -*-
"""Restructure the NEU-DET distribution into a 6-class ImageFolder layout.

The official NEU-DET download nests class folders under train/ and
validation/ split directories. Pointing torchvision.datasets.ImageFolder at
the distribution root silently yields a 2-class split-membership task, so
this script flattens both splits into NEU-DET_ByClass/<defect_class>/ before
any experiment touches the data.
"""
import os, shutil

SRC = [os.path.join("NEU-DET", "train", "images"),
       os.path.join("NEU-DET", "validation", "images")]
DST = "NEU-DET_ByClass"

n = 0
for root in SRC:
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        os.makedirs(os.path.join(DST, cls), exist_ok=True)
        for f in os.listdir(cdir):
            tgt = os.path.join(DST, cls, f)
            if not os.path.exists(tgt):
                shutil.copy2(os.path.join(cdir, f), tgt)
                n += 1
print(f"copied {n} images")
for cls in sorted(os.listdir(DST)):
    print(f"  {cls}: {len(os.listdir(os.path.join(DST, cls)))}")
