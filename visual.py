import cv2
import os
import random

IMG_DIR = "data2/images/train"
LBL_DIR = "data2/labels/train"

NUM_SAMPLES = 20   

def visualize_random_samples():

    images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    random.shuffle(images)

    for img_name in images[:NUM_SAMPLES]:

        img_path = os.path.join(IMG_DIR, img_name)
        lbl_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    cls, cx, cy, bw, bh = map(float, line.strip().split())

                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)

        
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Visualization", img)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_random_samples()