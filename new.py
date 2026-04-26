import cv2
import os

image_folder = "sequences/uav0000339_00001_v"
annotation_file = "annotations/uav0000339_00001_v.txt"
output = "output339"

os.makedirs(output_folder, exist_ok=True)

annotations = {}

with open(annotation_file, "r") as f:
    for line in f:
        data = list(map(int, line.strip().split(',')))

        frame_id = data[0]
        target_id = data[1]
        x, y, w, h = data[2], data[3], data[4], data[5]
        cls = data[7]   # ✅ class index

        if frame_id not in annotations:
            annotations[frame_id] = []

        annotations[frame_id].append((target_id, x, y, w, h, cls))

for frame_id in annotations:

    img_name = f"{frame_id:07d}.jpg"
    img_path = os.path.join(image_folder, img_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping missing: {img_path}")
        continue

    for (target_id, x, y, w, h, cls) in annotations[frame_id]:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # draw class label
        label = f"ID:{target_id} C:{cls}"
        cv2.putText(img, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, img)

print("Bounding box visualize:", output)