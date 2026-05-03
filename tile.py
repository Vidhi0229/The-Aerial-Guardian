import os
from PIL import Image

images_dir = "dataset/images/val"
labels_dir = "dataset/labels/val"

out_img_dir = "data/images/val"
out_lbl_dir = "data/labels/val"

TILE_SIZE = 640
OVERLAP = 0.3         
MIN_VISIBILITY = 0.2  
SAVE_EMPTY = False    

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

def load_yolo_labels(label_path, W, H):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, cx, cy, bw, bh = map(float, parts)

            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H

            boxes.append((int(cls), x1, y1, x2, y2))

    return boxes


def get_positions(size, tile_size, stride):
    """Ensure full coverage including edges"""
    positions = list(range(0, size - tile_size + 1, stride))
    if positions[-1] != size - tile_size:
        positions.append(size - tile_size)
    return positions


def tile_image(img_path):
    img = Image.open(img_path)
    W, H = img.size

    name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_dir, name + ".txt")

    boxes = load_yolo_labels(label_path, W, H)

    stride = int(TILE_SIZE * (1 - OVERLAP))
    tile_count = 0

    x_positions = get_positions(W, TILE_SIZE, stride)
    y_positions = get_positions(H, TILE_SIZE, stride)

    for y0 in y_positions:
        for x0 in x_positions:

            x1_tile = x0 + TILE_SIZE
            y1_tile = y0 + TILE_SIZE

            tile_labels = []

            for (cls, bx1, by1, bx2, by2) in boxes:

                # Intersection (clipped box)
                ix1 = max(bx1, x0)
                iy1 = max(by1, y0)
                ix2 = min(bx2, x1_tile)
                iy2 = min(by2, y1_tile)

                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                box_area = (bx2 - bx1) * (by2 - by1)
                inter_area = (ix2 - ix1) * (iy2 - iy1)

                if box_area == 0 or (inter_area / box_area) < MIN_VISIBILITY:
                    continue

                cx = ((ix1 + ix2) / 2 - x0) / TILE_SIZE
                cy = ((iy1 + iy2) / 2 - y0) / TILE_SIZE
                bw = (ix2 - ix1) / TILE_SIZE
                bh = (iy2 - iy1) / TILE_SIZE

                # Clamp
                cx = min(max(cx, 0), 1)
                cy = min(max(cy, 0), 1)
                bw = min(max(bw, 0), 1)
                bh = min(max(bh, 0), 1)

                tile_labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            has_labels = len(tile_labels) > 0

            if has_labels or SAVE_EMPTY:
                tile_name = f"{name}_x{x0}_y{y0}"

                tile_img = img.crop((x0, y0, x1_tile, y1_tile))
                tile_img.save(os.path.join(out_img_dir, tile_name + ".jpg"))

                with open(os.path.join(out_lbl_dir, tile_name + ".txt"), "w") as f:
                    f.write("\n".join(tile_labels))

                tile_count += 1

    return tile_count


# Run
total_tiles = 0

for img_name in os.listdir(images_dir):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(images_dir, img_name)
    tiles = tile_image(img_path)
    total_tiles += tiles

print(f"\nTotal tiles created: {total_tiles}")