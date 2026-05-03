import os
import shutil
import random
import numpy as np
from collections import defaultdict

random.seed(42)

BASE = "/home/vidhi/VisDrone2019-MOT-val/The-Aerial-Guardian"
out_base = f"{BASE}/data2"


img_dir_train = f"{BASE}/data/images/train"
lbl_dir_train = f"{BASE}/data/labels/train"
img_dir_val   = f"{BASE}/data/images/val"
lbl_dir_val   = f"{BASE}/data/labels/val"

all_imgs = (
    [(f, img_dir_train, lbl_dir_train)
     for f in os.listdir(img_dir_train) if f.endswith((".jpg",".png"))] +
    [(f, img_dir_val, lbl_dir_val)
     for f in os.listdir(img_dir_val)   if f.endswith((".jpg",".png"))]
)

seq_map = defaultdict(list)
for item in all_imgs:
    seq = item[0].split("_")[0]
    seq_map[seq].append(item)

print("Sequence sizes:")
for s in sorted(seq_map):
    print(f"  {s}: {len(seq_map[s])} images")

train_imgs, val_imgs = [], []
for seq, items in seq_map.items():
    random.shuffle(items)
    n_val = max(1, int(len(items) * 0.15))
    val_imgs.extend(items[:n_val])
    train_imgs.extend(items[n_val:])

print(f"\nStratified split — Train: {len(train_imgs)}, Val: {len(val_imgs)}")

for d in [f"{out_base}/images/train", f"{out_base}/images/val",
          f"{out_base}/labels/train", f"{out_base}/labels/val"]:
    os.makedirs(d, exist_ok=True)

def copy_pair(fname, src_img_dir, src_lbl_dir, dest_split):
    shutil.copy2(
        os.path.join(src_img_dir, fname),
        f"{out_base}/images/{dest_split}/{fname}"
    )
    lbl = os.path.splitext(fname)[0] + ".txt"
    src_lbl = os.path.join(src_lbl_dir, lbl)
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, f"{out_base}/labels/{dest_split}/{lbl}")

for fname, img_dir, lbl_dir in train_imgs:
    copy_pair(fname, img_dir, lbl_dir, "train")
for fname, img_dir, lbl_dir in val_imgs:
    copy_pair(fname, img_dir, lbl_dir, "val")

for split in ["train", "val"]:
    img_count = len(os.listdir(f"{out_base}/images/{split}"))
    lbl_count = len(os.listdir(f"{out_base}/labels/{split}"))
    
    widths, heights = [], []
    for f in os.listdir(f"{out_base}/labels/{split}"):
        with open(f"{out_base}/labels/{split}/{f}") as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) == 5:
                    widths.append(float(parts[3]) * 1024)
                    heights.append(float(parts[4]) * 1024)
    
    print(f"\n{split}: {img_count} images, {lbl_count} labels")
    print(f"  median box width:  {np.median(widths):.1f}px")
    print(f"  median box height: {np.median(heights):.1f}px")
