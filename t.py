
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import time

MODEL       = "The-Aerial-Guardian/models/best.pt"                  
SOURCE      = "sequences/uav0000073_04464_v"      
OUTPUT      = "out.mp4"                     # output video path

CONF        = 0.10                        
IOU         = 0.30                          
IMGSZ       = 1280                      
DEVICE      = "cpu"                         
HIGH_THRESH = 0.10                         
MIN_HITS    = 2                

FRAME_SKIP  = 1                             
TAIL        = 12                           


class TorchDetector:
    def __init__(self, model_path, imgsz=640, conf=0.10, iou=0.30, device="cpu"):
        self.imgsz  = imgsz
        self.conf   = conf
        self.iou    = iou
        self.device = device

        self.model = YOLO(model_path)

        # warm-up: first inference is always slow (graph build / CUDA init)
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=imgsz, conf=conf, iou=iou,
                            device=device, verbose=False)

        print(f"Model  : {model_path}")
        print(f"Conf   : {conf}  |  IOU : {iou}")
        print(f"Device : {device}")

    def detect(self, frame):
        t0 = time.perf_counter()
        result = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False
        )[0]

        boxes, scores = [], []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy  = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(float(c))

        inf_ms = (time.perf_counter() - t0) * 1000
        return boxes, scores, inf_ms


class SimpleTracker:
    def __init__(self, max_age=15, min_hits=3, iou_thresh=0.10, high_thresh=0.25):
        self.max_age    = max_age
        self.min_hits   = min_hits
        self.iou_thresh = iou_thresh
        self.high_thresh = high_thresh
        self.tracks   = []
        self.next_id  = 1

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def _match(self, dets):
        """Greedy best-IoU match, detections -> existing tracks."""
        matched = {}
        used    = set()
        for di, (box, _) in enumerate(dets):
            best_iou, best_ti = 0.0, -1
            for ti, t in enumerate(self.tracks):
                if ti in used:
                    continue
                score = self._iou(box, t["box"])
                if score > best_iou:
                    best_iou, best_ti = score, ti
            if best_iou >= self.iou_thresh and best_ti >= 0:
                matched[di] = best_ti
                used.add(best_ti)
        return matched

    def update(self, boxes, scores):
        high = [(b, s) for b, s in zip(boxes, scores) if s >= self.high_thresh]

        matches = self._match(high)

        # update matched tracks
        for di, ti in matches.items():
            box, score = high[di]
            self.tracks[ti].update({
                "box":   box,
                "score": score,
                "age":   0,
                "hits":  self.tracks[ti]["hits"] + 1,
            })

        # age unmatched tracks
        matched_tracks = set(matches.values())
        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks:
                t["age"] += 1

        # spawn new tracks for unmatched detections
        for di, (box, score) in enumerate(high):
            if di not in matches:
                self.tracks.append({
                    "id":    self.next_id,
                    "box":   box,
                    "score": score,
                    "age":   0,
                    "hits":  1,
                })
                self.next_id += 1

        # drop tracks that have aged out
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        return [t for t in self.tracks if t["hits"] >= self.min_hits]


# ── drawing ──────────────────────────────────────────────────────────

def get_color(tid):
    np.random.seed(int(tid) * 7 + 13)
    return tuple(int(x) for x in np.random.randint(80, 255, 3))


def draw_tracks(frame, tracks, history, tail):
    for t in tracks:
        x1, y1, x2, y2 = t["box"]
        tid   = t["id"]
        color = get_color(tid)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        history[tid].append((cx, cy))
        if len(history[tid]) > tail:
            history[tid].pop(0)

        pts = history[tid]
        n   = len(pts)
        for i in range(1, n):
            alpha     = i / n
            thickness = max(1, int(alpha * 3))
            blend     = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pts[i-1], pts[i], blend, thickness)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{tid}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame,
                      (x1, y1 - th - baseline - 4),
                      (x1 + tw + 2, y1),
                      color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


# ── main pipeline ────────────────────────────────────────────────────

def run_tracking():
    if not os.path.exists(SOURCE):
        raise ValueError(f"Source folder not found: {SOURCE}")

    files = sorted([f for f in os.listdir(SOURCE)
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if not files:
        raise ValueError("No images found in source folder")

    paths  = [os.path.join(SOURCE, f) for f in files]
    sample = cv2.imread(paths[0])
    if sample is None:
        raise ValueError("Failed to read first image")

    h, w = sample.shape[:2]
    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)

    writer = cv2.VideoWriter(
        OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    print(f"Source : {SOURCE}  ({len(paths)} frames)")
    print(f"Output : {OUTPUT}")
    print(f"Frame  : {w}x{h}")

    detector = TorchDetector(
        model_path = MODEL,
        imgsz      = IMGSZ,
        conf       = CONF,
        iou        = IOU,
        device     = DEVICE,
    )

    tracker = SimpleTracker(
        max_age    = 15,
        min_hits   = MIN_HITS,
        iou_thresh = 0.10,
        high_thresh = HIGH_THRESH,
    )

    history   = defaultdict(list)
    inf_times = []

    for i, path in enumerate(paths):
        frame = cv2.imread(path)
        if frame is None:
            print(f"  skip: {path}")
            continue

        if FRAME_SKIP > 1 and i % FRAME_SKIP != 0:
            tracks = tracker.update([], [])
            draw_tracks(frame, tracks, history, TAIL)
            writer.write(frame)
            continue

        boxes, scores, inf = detector.detect(frame)
        inf_times.append(inf)

        tracks = tracker.update(boxes, scores)
        draw_tracks(frame, tracks, history, TAIL)

        avg = sum(inf_times[-30:]) / min(len(inf_times), 30)
        fps = 1000 / avg * FRAME_SKIP

        cv2.putText(
            frame,
            f"FPS:{fps:.1f}  Tracks:{len(tracks)}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA
        )

        writer.write(frame)

        if len(inf_times) % 50 == 0:
            print(f"  [{len(inf_times)}/{len(paths)}] "
                  f"avg {avg:.0f} ms/frame  FPS≈{fps:.1f}  tracks={len(tracks)}")

    writer.release()

    if inf_times:
        avg_inf = sum(inf_times) / len(inf_times)
        print("\nDone.")
        print(f"Avg inference : {avg_inf:.1f} ms")
        print(f"Avg FPS       : {1000/avg_inf*FRAME_SKIP:.2f}")
        print(f"Video saved   : {OUTPUT}")


if __name__ == "__main__":
    run_tracking()