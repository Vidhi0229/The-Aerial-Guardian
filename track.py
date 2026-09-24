import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict
import os
import time


# ══════════════════════════════════════════════════════════════════════
#  CONFIG — edit these values directly instead of passing CLI flags
# ══════════════════════════════════════════════════════════════════════

MODEL       = "best.onnx"                  # path to the .onnx weights
SOURCE      = "./test-images/test2/"        # folder of input images
OUTPUT      = "out.mp4"                     # output video path

CONF        = 0.10                          # detection confidence threshold
IOU         = 0.30                          # NMS IoU threshold
IMGSZ       = 640                           # model input size

FRAME_SKIP  = 1                             # run detector every N frames (1 = every frame)
TAIL        = 12                            # length of the fading trail behind each track


class ONNXDetector:
    """Single full-frame ONNX inference — no tiling, no thread pool."""

    def __init__(self, model_path, imgsz=640, conf=0.10, iou=0.30):
        self.imgsz = imgsz
        self.conf  = conf
        self.iou   = iou

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # warm-up: first inference is always slow due to JIT
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

        print(f"Model : {model_path}")
        print(f"Conf  : {conf}  |  IOU : {iou}")

    def _preprocess(self, frame):
        oh, ow = frame.shape[:2]
        scale  = min(self.imgsz / ow, self.imgsz / oh)
        nw, nh = int(ow * scale), int(oh * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas  = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top  = (self.imgsz - nh) // 2
        left = (self.imgsz - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]
        return np.ascontiguousarray(img), scale, left, top

    def _postprocess(self, output, ow, oh, scale, left, top):
        pred = output[0][0].T
        mask = pred[:, 4] >= self.conf
        pred = pred[mask]
        if len(pred) == 0:
            return [], []

        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        cx = (cx - left) / scale
        cy = (cy - top)  / scale
        bw /= scale
        bh /= scale

        x1 = np.clip(cx - bw / 2, 0, ow).astype(int)
        y1 = np.clip(cy - bh / 2, 0, oh).astype(int)
        x2 = np.clip(cx + bw / 2, 0, ow).astype(int)
        y2 = np.clip(cy + bh / 2, 0, oh).astype(int)

        scores     = pred[:, 4].tolist()
        boxes_xywh = [[x1[i], y1[i], x2[i]-x1[i], y2[i]-y1[i]] for i in range(len(scores))]

        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.conf, self.iou)
        if len(indices) == 0:
            return [], []
        indices = indices.flatten()

        return [[x1[i], y1[i], x2[i], y2[i]] for i in indices], \
               [scores[i] for i in indices]

    def detect(self, frame):
        t0 = time.perf_counter()
        oh, ow = frame.shape[:2]
        inp, scale, left, top = self._preprocess(frame)
        out = self.session.run([self.output_name], {self.input_name: inp})
        boxes, scores = self._postprocess(out, ow, oh, scale, left, top)
        inf_ms = (time.perf_counter() - t0) * 1000
        return boxes, scores, inf_ms


class SimpleTracker:
    """
    Minimal IoU-only tracker.

    Each track: {id, box, score, age, hits}
    No velocity, no graveyard/backtracking — a track that ages out past
    max_age is simply dropped, and a re-appearing person gets a new ID.
    """

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

    detector = ONNXDetector(
        model_path = MODEL,
        imgsz      = IMGSZ,
        conf       = CONF,
        iou        = IOU,
    )

    tracker = SimpleTracker(
        max_age    = 15,
        min_hits   = 3,
        iou_thresh = 0.10,
        high_thresh = 0.25,
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
