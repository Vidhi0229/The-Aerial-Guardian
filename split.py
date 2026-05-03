import cv2
import os
import shutil
import random
import numpy as np
from collections import defaultdict

SEQUENCE_NAMES = [
    "uav0000086_00000_v",
    "uav0000117_02622_v",
    "uav0000137_00458_v",
    "uav0000339_00001_v",
]

SEQUENCES_ROOT   = "sequences"
ANNOTATIONS_ROOT = "annotations"
LABELS_ROOT      = "labels"

TILES_IMG_DIR    = "tiled/images"
TILES_LBL_DIR    = "tiled/labels"

TILE_SIZE        = 640
OVERLAP          = 0.20           

BASE             = "/home/vidhi/VisDrone2019-MOT-val/The-Aerial-Guardian"
DATA2_DIR        = f"{BASE}/data"

RANDOM_SEED      = 42
random.seed(RANDOM_SEED)

def stage1_convert_annotations():
    print("\n" + "=" * 60)
    print("STAGE 1 — Converting annotations to YOLO format")
    print("=" * 60)

    os.makedirs(LABELS_ROOT, exist_ok=True)

    for sequence_name in SEQUENCE_NAMES:
        annotation_file = os.path.join(ANNOTATIONS_ROOT, f"{sequence_name}.txt")
        image_folder    = os.path.join(SEQUENCES_ROOT, sequence_name)

        annotations: dict[int, list] = {}
        with open(annotation_file, "r") as f:
            for line in f:
                data     = list(map(int, line.strip().split(",")))
                frame_id = data[0]
                x, y, w, h = data[2], data[3], data[4], data[5]
                cls      = data[7]

                if cls not in [1, 2]:
                    continue

                annotations.setdefault(frame_id, []).append((x, y, w, h))

        written = 0
        for frame_id, boxes in annotations.items():
            img_path = os.path.join(image_folder, f"{frame_id:07d}.jpg")
            img      = cv2.imread(img_path)
            if img is None:
                print(f"  [MISSING IMAGE] {img_path}")
                continue

            h_img, w_img = img.shape[:2]
            label_path   = os.path.join(LABELS_ROOT, f"{sequence_name}_{frame_id:07d}.txt")

            with open(label_path, "w") as f:
                for (x, y, w, h) in boxes:
                    cx     = (x + w / 2) / w_img
                    cy     = (y + h / 2) / h_img
                    w_norm = w / w_img
                    h_norm = h / h_img
                    f.write(f"0 {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            written += 1

        print(f"  {sequence_name}: {written} label files → {LABELS_ROOT}/")

    print("Stage 1 complete.\n")


def _tile_one(img_path: str, lbl_path: str, stem: str) -> int:
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [MISSING IMAGE] {img_path}")
        return 0

    H, W   = img.shape[:2]
    stride = int(TILE_SIZE * (1 - OVERLAP))   

    abs_boxes = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            abs_boxes.append((cx * W, cy * H, bw * W, bh * H))

    def tile_starts(length):
        starts = list(range(0, length - TILE_SIZE, stride))
        starts.append(length - TILE_SIZE)
        return sorted(set(max(s, 0) for s in starts))

    saved = 0
    for y0 in tile_starts(H):
        for x0 in tile_starts(W):
            x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE

            # Re-project boxes into tile coordinate space
            tile_boxes = []
            for (cx_abs, cy_abs, bw_abs, bh_abs) in abs_boxes:
                bx0 = cx_abs - bw_abs / 2
                by0 = cy_abs - bh_abs / 2
                bx1 = cx_abs + bw_abs / 2
                by1 = cy_abs + bh_abs / 2

                clipped_x0 = max(bx0, x0)
                clipped_y0 = max(by0, y0)
                clipped_x1 = min(bx1, x1)
                clipped_y1 = min(by1, y1)

                if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
                    continue

                new_cx = (clipped_x0 + clipped_x1) / 2 - x0
                new_cy = (clipped_y0 + clipped_y1) / 2 - y0
                new_w  = clipped_x1 - clipped_x0
                new_h  = clipped_y1 - clipped_y0

                tile_boxes.append((
                    new_cx / TILE_SIZE,
                    new_cy / TILE_SIZE,
                    new_w  / TILE_SIZE,
                    new_h  / TILE_SIZE,
                ))

            if not tile_boxes:
                continue

            tile_stem = f"{stem}_tile_{y0}_{x0}"
            cv2.imwrite(os.path.join(TILES_IMG_DIR, f"{tile_stem}.jpg"), img[y0:y1, x0:x1])

            with open(os.path.join(TILES_LBL_DIR, f"{tile_stem}.txt"), "w") as f:
                for (cx, cy, bw, bh) in tile_boxes:
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            saved += 1

    return saved


def stage2_tile():
    print("=" * 60)
    print(f"STAGE 2 — Tiling ({TILE_SIZE}×{TILE_SIZE}, {int(OVERLAP*100)}% overlap, annotated tiles only)")
    print("=" * 60)

    os.makedirs(TILES_IMG_DIR, exist_ok=True)
    os.makedirs(TILES_LBL_DIR, exist_ok=True)

    total_tiles   = 0
    skipped_frames = 0

    for sequence_name in SEQUENCE_NAMES:
        image_folder = os.path.join(SEQUENCES_ROOT, sequence_name)
        seq_tiles    = 0

        for img_name in sorted(os.listdir(image_folder)):
            if not img_name.endswith(".jpg"):
                continue

            frame_stem = img_name.replace(".jpg", "")
            label_stem = f"{sequence_name}_{frame_stem}"
            lbl_path   = os.path.join(LABELS_ROOT, f"{label_stem}.txt")

            # ── KEY CHANGE: skip frames with no label file or empty label file ──
            if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
                skipped_frames += 1
                continue

            img_path   = os.path.join(image_folder, img_name)
            seq_tiles += _tile_one(img_path, lbl_path, label_stem)

        print(f"  {sequence_name}: {seq_tiles} tiles saved")
        total_tiles += seq_tiles

    print(f"  Frames skipped (no annotation): {skipped_frames}")
    print(f"  Total tiles saved: {total_tiles}")
    print("Stage 2 complete.\n")


def stage3_stratified_split():
    print("=" * 60)
    print("STAGE 3 — Stratified 85/15 per-sequence split → data2/")
    print("=" * 60)

    all_imgs = [
        f for f in os.listdir(TILES_IMG_DIR)
        if f.endswith((".jpg", ".png"))
    ]

    seq_map: dict[str, list] = defaultdict(list)
    for fname in all_imgs:
        seq = fname.split("_tile_")[0]
        seq_map[seq].append(fname)

    print("Sequence tile counts:")
    for s in sorted(seq_map):
        print(f"  {s}: {len(seq_map[s])} tiles")

    train_files, val_files = [], []
    for seq, files in seq_map.items():
        random.shuffle(files)
        n_val = max(1, int(len(files) * 0.15))
        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])

    print(f"\nStratified split — Train: {len(train_files)} | Val: {len(val_files)}")

    for d in [
        f"{DATA2_DIR}/images/train", f"{DATA2_DIR}/images/val",
        f"{DATA2_DIR}/labels/train", f"{DATA2_DIR}/labels/val",
    ]:
        os.makedirs(d, exist_ok=True)

    def copy_pair(fname: str, dest_split: str):
        shutil.copy2(
            os.path.join(TILES_IMG_DIR, fname),
            f"{DATA2_DIR}/images/{dest_split}/{fname}",
        )
        lbl     = os.path.splitext(fname)[0] + ".txt"
        src_lbl = os.path.join(TILES_LBL_DIR, lbl)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, f"{DATA2_DIR}/labels/{dest_split}/{lbl}")

    for fname in train_files:
        copy_pair(fname, "train")
    for fname in val_files:
        copy_pair(fname, "val")

    print("\nFinal statistics:")
    for split in ["train", "val"]:
        img_count = len(os.listdir(f"{DATA2_DIR}/images/{split}"))
        lbl_dir   = f"{DATA2_DIR}/labels/{split}"
        lbl_count = len(os.listdir(lbl_dir))

        widths, heights = [], []
        for f in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, f)) as fp:
                for line in fp:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        widths.append(float(parts[3]) * TILE_SIZE)
                        heights.append(float(parts[4]) * TILE_SIZE)

        print(f"  {split}: {img_count} images | {lbl_count} labels")
        if widths:
            print(f"    median box width : {np.median(widths):.1f} px")
            print(f"    median box height: {np.median(heights):.1f} px")
        else:
            print(f"    (no annotated boxes)")

    print("Stage 3 complete.\n")


if __name__ == "__main__":
    stage1_convert_annotations()
    stage2_tile()
    stage3_stratified_split()
    print("Pipeline finished.")