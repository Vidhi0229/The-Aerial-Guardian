import cv2
import os
import time
import argparse
import numpy as np
from collections import defaultdict
import onnxruntime as ort

class ONNXDetector:
    def __init__(self, model_path, imgsz=640, conf=0.15, iou=0.45):
        self.imgsz = imgsz
        self.conf  = conf
        self.iou   = iou

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"Model   : {model_path}")
        print(f"Threads : 3 | CPUExecutionProvider")

    def preprocess(self, frame):
        h, w = frame.shape[:2]

        scale = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (nw, nh))

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top  = (self.imgsz - nh) // 2
        left = (self.imgsz - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized

        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]

        return np.ascontiguousarray(img), w, h, scale, left, top

    def postprocess(self, output, w, h, scale, left, top):
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

        x1 = np.clip(cx - bw/2, 0, w).astype(int)
        y1 = np.clip(cy - bh/2, 0, h).astype(int)
        x2 = np.clip(cx + bw/2, 0, w).astype(int)
        y2 = np.clip(cy + bh/2, 0, h).astype(int)

        scores = pred[:, 4].tolist()

        boxes_xywh = [[x1[i], y1[i], x2[i]-x1[i], y2[i]-y1[i]]
                      for i in range(len(scores))]

        idx = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.conf, self.iou)

        if len(idx) == 0:
            return [], []

        idx = idx.flatten()
        boxes = [[x1[i], y1[i], x2[i], y2[i]] for i in idx]
        scores = [scores[i] for i in idx]

        return boxes, scores

    def detect(self, frame):
        inp, w, h, scale, left, top = self.preprocess(frame)
        out = self.session.run([self.output_name], {self.input_name: inp})
        return self.postprocess(out, w, h, scale, left, top)

class ByteTracker:
    def __init__(self, max_age=25, min_hits=2,
                 iou_thresh=0.2, high_thresh=0.3):
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
        for di,(box,_) in enumerate(dets):
            best_iou, best_t = 0, -1
            for ti in track_indices:
                if ti in used: continue
                v = self.iou(box, self.tracks[ti]["box"])
                if v > best_iou:
                    best_iou, best_t = v, ti
            if best_iou >= self.iou_thresh:
                matched[di] = best_t
                used.add(best_t)
        return matched

    def update(self, boxes, scores):
        high = [(b,s) for b,s in zip(boxes,scores) if s>=self.high_thresh]
        low  = [(b,s) for b,s in zip(boxes,scores) if s< self.high_thresh]

        active = list(range(len(self.tracks)))

        m1 = self._match(high, active)
        for di,ti in m1.items():
            self.tracks[ti].update({"box":high[di][0],"score":high[di][1],
                                    "age":0,"hits":self.tracks[ti]["hits"]+1})

        unmatched = [i for i in active if i not in m1.values()]

        m2 = self._match(low, unmatched)
        for di,ti in m2.items():
            self.tracks[ti].update({"box":low[di][0],"score":low[di][1],
                                    "age":0})

        for ti in unmatched:
            if ti not in m2.values():
                self.tracks[ti]["age"] += 1

        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        for di in range(len(high)):
            if di not in m1:
                self.tracks.append({
                    "id": self.next_id,
                    "box": high[di][0],
                    "score": high[di][1],
                    "age": 0,
                    "hits": 1
                })
                self.next_id += 1

        return [t for t in self.tracks if t["hits"] >= self.min_hits]

def run(model, source, imgsz, conf, iou, show, save):

    detector = ONNXDetector(model, imgsz, conf, iou)
    tracker  = ByteTracker()

    files = sorted([f for f in os.listdir(source) if f.endswith((".jpg",".png"))],
                   key=lambda x: int(os.path.splitext(x)[0]))
    paths = [os.path.join(source,f) for f in files]

    sample = cv2.imread(paths[0])
    h,w = sample.shape[:2]

    writer = None
    if save:
        writer = cv2.VideoWriter(save,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 25,(w,h))

    times = []

    for p in paths:
        t0 = time.perf_counter()

        frame = cv2.imread(p)
        boxes, scores = detector.detect(frame)
        tracks = tracker.update(boxes, scores)

        for t in tracks:
            x1,y1,x2,y2 = t["box"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)

        t1 = time.perf_counter()
        times.append(t1 - t0)

        fps = 1.0 / (sum(times[-20:]) / min(len(times),20))

        cv2.putText(frame,f"FPS:{fps:.1f}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        if writer:
            writer.write(frame)

        if show:
            cv2.imshow("Aerial Guardian", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Avg FPS: {1.0/(sum(times)/len(times)):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="best.onnx")
    p.add_argument("--source", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--show", action="store_true")
    p.add_argument("--save", default="output.mp4")

    args = p.parse_args()

    run(args.model, args.source, args.imgsz,
        args.conf, args.iou, args.show, args.save)