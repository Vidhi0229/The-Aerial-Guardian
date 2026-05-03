import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict
import argparse
import os
import time


class ONNXDetector:
    def __init__(self, model_path, imgsz=640, conf=0.15, iou=0.45):
        self.imgsz = imgsz
        self.conf  = conf
        self.iou   = iou

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"Model   : {model_path}")
        print(f"Threads : 3 | CPUExecutionProvider")

    def preprocess(self, frame):
        oh, ow = frame.shape[:2]

        scale = min(self.imgsz / ow, self.imgsz / oh)
        nw, nh = int(ow * scale), int(oh * scale)

        resized = cv2.resize(frame, (nw, nh))

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)

        top  = (self.imgsz - nh) // 2
        left = (self.imgsz - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized

        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]

        return np.ascontiguousarray(img), ow, oh, scale, left, top

    def postprocess(self, output, ow, oh, scale, left, top):
        pred = output[0][0].T

        mask = pred[:, 4] >= self.conf
        pred = pred[mask]

        if len(pred) == 0:
            return [], []

        cx, cy, bw, bh = pred[:,0], pred[:,1], pred[:,2], pred[:,3]

        cx = (cx - left) / scale
        cy = (cy - top)  / scale
        bw = bw / scale
        bh = bh / scale

        x1 = np.clip(cx - bw/2, 0, ow).astype(int)
        y1 = np.clip(cy - bh/2, 0, oh).astype(int)
        x2 = np.clip(cx + bw/2, 0, ow).astype(int)
        y2 = np.clip(cy + bh/2, 0, oh).astype(int)

        scores = pred[:, 4].tolist()

        boxes_xywh = [[x1[i], y1[i], x2[i]-x1[i], y2[i]-y1[i]]
                      for i in range(len(scores))]

        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.conf, self.iou)

        if len(indices) == 0:
            return [], []

        indices = indices.flatten()
        boxes  = [[x1[i], y1[i], x2[i], y2[i]] for i in indices]
        scores = [scores[i] for i in indices]

        return boxes, scores

    def detect(self, frame):
        t0 = time.perf_counter()

        inp, ow, oh, scale, left, top = self.preprocess(frame)
        out = self.session.run([self.output_name], {self.input_name: inp})

        inf_ms = (time.perf_counter() - t0) * 1000

        boxes, scores = self.postprocess(out, ow, oh, scale, left, top)

        return boxes, scores, inf_ms


class ByteTracker:
    def __init__(self, max_age=25, min_hits=2, iou_thresh=0.20, high_thresh=0.30):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.high_thresh = high_thresh
        self.tracks = []
        self.next_id = 1

    def iou(self, a, b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        ix1=max(ax1,bx1); iy1=max(ay1,by1)
        ix2=min(ax2,bx2); iy2=min(ay2,by2)
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        union=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/union if union>0 else 0

    def _match(self, dets, track_indices):
        matched = {}
        used = set()
        for di, (box, _) in enumerate(dets):
            best_iou, best_ti = 0, -1
            for ti in track_indices:
                if ti in used:
                    continue
                v = self.iou(box, self.tracks[ti]["box"])
                if v > best_iou:
                    best_iou, best_ti = v, ti
            if best_iou >= self.iou_thresh and best_ti >= 0:
                matched[di] = best_ti
                used.add(best_ti)
        return matched

    def update(self, boxes, scores):
        high = [(b,s) for b,s in zip(boxes,scores) if s>=self.high_thresh]
        active = list(range(len(self.tracks)))

        m1 = self._match(high, active)
        for di, ti in m1.items():
            self.tracks[ti].update({
                "box": high[di][0],
                "score": high[di][1],
                "age": 0,
                "hits": self.tracks[ti]["hits"] + 1
            })

        unmatched_t = [i for i in active if i not in m1.values()]
        unmatched_d = [i for i in range(len(high)) if i not in m1]

        for ti in unmatched_t:
            self.tracks[ti]["age"] += 1

        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        for di in unmatched_d:
            self.tracks.append({
                "id": self.next_id,
                "box": high[di][0],
                "score": high[di][1],
                "age": 0,
                "hits": 1
            })
            self.next_id += 1

        return [t for t in self.tracks if t["hits"] >= self.min_hits]


def get_color(tid):
    np.random.seed(int(tid)*7+13)
    return tuple(int(x) for x in np.random.randint(80,255,3))


def run_tracking(model, source, output, conf, iou, imgsz, tail):

    if not os.path.exists(source):
        raise ValueError(f"Source folder not found: {source}")

    files = sorted([f for f in os.listdir(source)
                    if f.lower().endswith((".jpg",".png",".jpeg"))])

    if len(files) == 0:
        raise ValueError("No images found in source folder")

    paths = [os.path.join(source,f) for f in files]

    sample = cv2.imread(paths[0])
    if sample is None:
        raise ValueError("Failed to read first image")

    h, w = sample.shape[:2]

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"Saving video to: {output}")
    print(f"Total frames: {len(paths)}")

    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (w, h)
    )

    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    detector = ONNXDetector(model, imgsz, conf, iou)
    tracker  = ByteTracker()

    history = defaultdict(list)
    inf_times = []

    for path in paths:
        frame = cv2.imread(path)

        if frame is None:
            print(f"Skipping bad frame: {path}")
            continue

        boxes, scores, inf = detector.detect(frame)
        inf_times.append(inf)

        tracks = tracker.update(boxes, scores)

        for t in tracks:
            x1,y1,x2,y2 = t["box"]
            tid = t["id"]
            color = get_color(tid)

            cx,cy = (x1+x2)//2,(y1+y2)//2
            history[tid].append((cx,cy))

            if len(history[tid]) > tail:
                history[tid].pop(0)

            for i in range(1, len(history[tid])):
                cv2.line(frame, history[tid][i-1], history[tid][i], color, 2)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        avg = sum(inf_times[-30:]) / min(len(inf_times), 30)
        fps = 1000 / avg

        cv2.putText(frame, f"FPS:{fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        writer.write(frame)

    writer.release()

    print(f"\nVideo saved {output}")
    print(f"Avg FPS: {1000/(sum(inf_times)/len(inf_times)):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="best.onnx")
    p.add_argument("--source", required=True)
    p.add_argument("--output", default="output.mp4")
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--tail", type=int, default=20)

    args = p.parse_args()

    run_tracking(args.model, args.source, args.output,
                 args.conf, args.iou, args.imgsz, args.tail)