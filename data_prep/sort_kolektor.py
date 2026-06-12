import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# --- Master Destination ---
DST_DIR = "Kolektor"
DEFECT_DIR = os.path.join(DST_DIR, "Defect")
OK_DIR = os.path.join(DST_DIR, "OK")

# Create the clean PyTorch ImageFolder structure
os.makedirs(DEFECT_DIR, exist_ok=True)
os.makedirs(OK_DIR, exist_ok=True)

print("🚀 Starting Dataset Reorganization...")

# ---------------------------------------------------------
# 1. Process KolektorSDD2 (train & test folders)
# ---------------------------------------------------------
print("\nScanning KolektorSDD2...")
ksdd2_base = Path("KolektorSDD2")

if ksdd2_base.exists():
    for split in ["train", "test"]:
        split_dir = ksdd2_base / split
        if not split_dir.exists(): 
            continue
            
        for img_path in split_dir.glob("*.png"):
            # Skip the mask files during the main loop
            if "_GT" in img_path.name:
                continue
                
            # Find the matching mask
            mask_name = img_path.name.replace(".png", "_GT.png")
            mask_path = split_dir / mask_name
            
            if not mask_path.exists(): 
                continue
                
            # Check mask: if any pixel is > 0 (white), it's a defect
            mask = np.array(Image.open(mask_path).convert("L"))
            label = "Defect" if mask.max() > 0 else "OK"
            
            # Prefix filename to prevent overwrites (e.g., K2_train_0.png)
            new_name = f"K2_{split}_{img_path.name}"
            shutil.copy(img_path, os.path.join(DST_DIR, label, new_name))
else:
    print("⚠️ KolektorSDD2 folder not found.")

# ---------------------------------------------------------
# 2. Process KolektorSDD-boxes (kos01, kos02, etc.)
# ---------------------------------------------------------
print("\nScanning KolektorSDD-boxes...")
ksdd_boxes_base = Path("KolektorSDD-boxes")

if ksdd_boxes_base.exists():
    # Iterate through kos01, kos02, etc.
    for kos_dir in ksdd_boxes_base.iterdir():
        if not kos_dir.is_dir(): 
            continue
            
        for img_path in kos_dir.glob("*.jpg"):
            # Find the matching .bmp mask
            mask_name = img_path.name.replace(".jpg", "_label.bmp")
            mask_path = kos_dir / mask_name
            
            if not mask_path.exists(): 
                continue
                
            # Check mask for defects
            mask = np.array(Image.open(mask_path).convert("L"))
            label = "Defect" if mask.max() > 0 else "OK"
            
            # Prefix filename to prevent overwrites (e.g., K1_kos01_Part0.jpg)
            new_name = f"K1_{kos_dir.name}_{img_path.name}"
            shutil.copy(img_path, os.path.join(DST_DIR, label, new_name))
else:
    print("⚠️ KolektorSDD-boxes folder not found.")

# ---------------------------------------------------------
print("\n✅ Sorting Complete!")
print(f"Total Defects: {len(os.listdir(DEFECT_DIR))}")
print(f"Total OK: {len(os.listdir(OK_DIR))}")
print("You can now use 'root': 'Kolektor' in your training registry.")