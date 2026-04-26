import cv2
import os

sequence_name = "uav0000339_00001_v"

image_folder = f"sequences/{sequence_name}"
annotation_file = f"annotations/{sequence_name}.txt"
label_output_folder = "labels"

os.makedirs(label_output_folder, exist_ok=True)

annotations = {}

with open(annotation_file, "r") as f:
    for line in f:
        data = list(map(int, line.strip().split(',')))

        frame_id = data[0]
        x, y, w, h = data[2], data[3], data[4], data[5]
        cls = data[7]

        if cls not in [1, 2]:
            continue

        if frame_id not in annotations:
            annotations[frame_id] = []

        annotations[frame_id].append((x, y, w, h))

for frame_id in annotations:

    img_name = f"{frame_id:07d}.jpg"
    img_path = os.path.join(image_folder, img_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"missing: {img_path}")
        continue

    h_img, w_img = img.shape[:2]

    new_name = f"{sequence_name}_{frame_id:07d}.txt"
    label_file_path = os.path.join(label_output_folder, new_name)

    with open(label_file_path, "w") as f:

        for (x, y, w, h) in annotations[frame_id]:

            cx = (x + w / 2) / w_img
            cy = (y + h / 2) / h_img
            w_norm = w / w_img
            h_norm = h / h_img

            f.write(f"0 {cx} {cy} {w_norm} {h_norm}\n")

print("YOLO labels")