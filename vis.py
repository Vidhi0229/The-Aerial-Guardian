import cv2
import os


image_folder = "dataset/images/val"
label_folder = "dataset/labels/val"
output_folder = "val_output"  

os.makedirs(output_folder, exist_ok=True)

drawn_total = 0
saved = 0

for img_name in sorted(os.listdir(image_folder)):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(image_folder, img_name)
    label_path = os.path.join(label_folder, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping {img_name}")
        continue

    h_img, w_img = img.shape[:2]
    boxes_drawn = 0

    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path, "r") as f:
            for line in f:
                data = line.strip().split()
                if len(data) < 5:
                    continue

                cx, cy, w, h = map(float, data[1:5])

                x = int((cx - w/2) * w_img)
                y = int((cy - h/2) * h_img)
                w_box = int(w * w_img)
                h_box = int(h * h_img)

                # clamp to image bounds
                x = max(0, x)
                y = max(0, y)

                cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(img, "HUMAN", (x, max(y - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                boxes_drawn += 1

    drawn_total += boxes_drawn
    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, img)
    saved += 1

print(f"Saved {saved} images with {drawn_total} total boxes")
print(f"Check: {output_folder}")