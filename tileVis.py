import os
import cv2
import random

IMAGE_DIR = "data/images/train"   
LABEL_DIR = "data/labels/train"
NUM_SAMPLES = 100  

def draw_boxes(image, label_path):
    h, w = image.shape[:2]

    if not os.path.exists(label_path):
        return image

    with open(label_path, "r") as f:
        for line in f:
            cls, cx, cy, bw, bh = map(float, line.strip().split())

            
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

          
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        
            cv2.putText(image, str(int(cls)), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return image


images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
samples = random.sample(images, min(NUM_SAMPLES, len(images)))

for img_name in samples:
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    img = draw_boxes(img, label_path)

    cv2.imshow("Tile Visualization", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()