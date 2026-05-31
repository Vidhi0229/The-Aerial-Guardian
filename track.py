#python track.py   --model /home/vidhi/VisDrone2019-MOT-val/The-Aerial-Guardian/drone_human_v5-4/weights/best.onnx   --source /home/vidhi/VisDrone2019-MOT-val/The-Aerial-Guardian/test-images/test2/   --output out.mp4   --conf 0.10 --iou 0.30   --slice --slice-size 960 --slice-overlap 0.15   --threads 4 --frame-skip 2   --tail 12 --backtrack-window 25 --backtrack-thresh 0.30

import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
import time


class ONNXDetector:
    def __init__(self, model_path, imgsz=640, conf=0.10, iou=0.30,
                 slice_infer=True, slice_size=960, slice_overlap=0.15,
                 num_threads=4):
        self.imgsz         = imgsz
        self.conf          = conf
        self.iou           = iou
        self.slice_infer   = slice_infer
        self.slice_size    = slice_size
        self.slice_overlap = slice_overlap
        self.num_threads   = num_threads

        opts = ort.SessionOptions()
        opts.intra_op_num_threads        = num_threads
        opts.inter_op_num_threads        = num_threads
        opts.graph_optimization_level    = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode              = ort.ExecutionMode.ORT_PARALLEL

        self.session = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # warm-up: first inference is always slow due to JIT
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

        print(f"Model        : {model_path}")
        print(f"Conf         : {conf}  |  IOU : {iou}")
        print(f"Slice infer  : {slice_infer}  size={slice_size}  overlap={slice_overlap}")
        print(f"ORT threads  : {num_threads}")

    # ── single-patch helpers ─────────────────────────────────────────

    def _preprocess(self, frame):
        oh, ow = frame.shape[:2]
        scale  = min(self.imgsz / ow, self.imgsz / oh)
        nw, nh = int(ow * scale), int(oh * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas  = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top  = (self.imgsz - nh) // 2
        left = (self.imgsz - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]
        return np.ascontiguousarray(img), ow, oh, scale, left, top

    def _postprocess(self, output, ow, oh, scale, left, top):
        pred = output[0][0].T
        mask = pred[:, 4] >= self.conf
        pred = pred[mask]
        if len(pred) == 0:
            return [], []
        cx, cy, bw, bh = pred[:,0], pred[:,1], pred[:,2], pred[:,3]
        cx = (cx - left) / scale
        cy = (cy - top)  / scale
        bw /= scale;  bh /= scale
        x1 = np.clip(cx - bw/2, 0, ow).astype(int)
        y1 = np.clip(cy - bh/2, 0, oh).astype(int)
        x2 = np.clip(cx + bw/2, 0, ow).astype(int)
        y2 = np.clip(cy + bh/2, 0, oh).astype(int)
        scores     = pred[:, 4].tolist()
        boxes_xywh = [[x1[i], y1[i], x2[i]-x1[i], y2[i]-y1[i]]
                      for i in range(len(scores))]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.conf, self.iou)
        if len(indices) == 0:
            return [], []
        indices = indices.flatten()
        return [[x1[i], y1[i], x2[i], y2[i]] for i in indices], \
               [scores[i] for i in indices]

    def _run_patch(self, args):
        """
        Inference on one (x_offset, y_offset, patch) tuple.
        Returns boxes already shifted to full-frame coordinates.
        ORT releases the GIL during session.run(), so threads
        genuinely run in parallel on separate CPU cores.
        """
        x_off, y_off, patch = args
        ph, pw = patch.shape[:2]
        inp, ow, oh, scale, left, top = self._preprocess(patch)
        out    = self.session.run([self.output_name], {self.input_name: inp})
        boxes, scores = self._postprocess(out, pw, ph, scale, left, top)
        shifted = [[x_off+bx1, y_off+by1, x_off+bx2, y_off+by2]
                   for bx1,by1,bx2,by2 in boxes]
        return shifted, scores

    # ── sliced inference with thread pool ────────────────────────────

    def _slice_detect(self, frame):
        oh, ow = frame.shape[:2]
        sz     = self.slice_size
        step   = int(sz * (1 - self.slice_overlap))

        ys = list(range(0, oh - sz + 1, step))
        xs = list(range(0, ow - sz + 1, step))
        if not ys or ys[-1] + sz < oh:
            ys.append(max(0, oh - sz))
        if not xs or xs[-1] + sz < ow:
            xs.append(max(0, ow - sz))

        # build tile list — slicing is O(1) in numpy (view, no copy)
        tiles = [
            (x, y, frame[y:y+sz, x:x+sz])
            for y in ys for x in xs
        ]

        all_boxes, all_scores = [], []

        # ORT releases the GIL during inference → real CPU parallelism
        with ThreadPoolExecutor(max_workers=self.num_threads) as exe:
            for boxes, scores in exe.map(self._run_patch, tiles):
                all_boxes.extend(boxes)
                all_scores.extend(scores)

        if not all_boxes:
            return [], []

        # single global NMS to remove cross-tile duplicates
        boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in all_boxes]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, all_scores, self.conf, self.iou)
        if len(indices) == 0:
            return [], []
        indices = indices.flatten()
        return [all_boxes[i] for i in indices], [all_scores[i] for i in indices]

    # ── full-frame inference (fallback / --no-slice) ─────────────────

    def _full_detect(self, frame):
        oh, ow = frame.shape[:2]
        inp, _, _, scale, left, top = self._preprocess(frame)
        out = self.session.run([self.output_name], {self.input_name: inp})
        return self._postprocess(out, ow, oh, scale, left, top)

    # ── public API ───────────────────────────────────────────────────

    def detect(self, frame):
        t0 = time.perf_counter()
        if self.slice_infer:
            boxes, scores = self._slice_detect(frame)
        else:
            boxes, scores = self._full_detect(frame)
        inf_ms = (time.perf_counter() - t0) * 1000
        return boxes, scores, inf_ms


class ByteTracker:
    """
    Aerial-optimised ByteTracker with custom backtracking.

    Active matching  : IoU + centre-distance composite score.
    Backtracking     : Graveyard resurrection via IoU + velocity
                       proximity + motion-angle consistency.
    Velocity         : Exponentially-smoothed per-track (vx, vy).
    """

    def __init__(
        self,
        max_age          = 15,
        min_hits         = 3,
        iou_thresh       = 0.10,
        high_thresh      = 0.25,
        backtrack_window = 25,
        backtrack_thresh = 0.30,
    ):
        self.max_age          = max_age
        self.min_hits         = min_hits
        self.iou_thresh       = iou_thresh
        self.high_thresh      = high_thresh
        self.backtrack_window = backtrack_window
        self.backtrack_thresh = backtrack_thresh

        self.tracks    = []
        self.graveyard = []
        self.next_id   = 1
        self.frame_id  = 0

    # ── geometry helpers ─────────────────────────────────────────────

    def _iou(self, a, b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        ix1 = max(ax1,bx1);  iy1 = max(ay1,by1)
        ix2 = min(ax2,bx2);  iy2 = min(ay2,by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def _center(self, box):
        return ((box[0]+box[2]) / 2.0, (box[1]+box[3]) / 2.0)

    def _box_diag(self, box):
        return max(1.0, ((box[2]-box[0])**2 + (box[3]-box[1])**2) ** 0.5)

    def _predict_center(self, track):
        cx, cy = self._center(track["box"])
        vx, vy = track.get("vx", 0.0), track.get("vy", 0.0)
        age    = track.get("age", 0)
        dead   = self.frame_id - track.get("dead_frame", self.frame_id)
        steps  = age + dead + 1
        return cx + vx * steps, cy + vy * steps

    # ── velocity ─────────────────────────────────────────────────────

    def _update_velocity(self, track, new_box, alpha=0.6):
        old_cx, old_cy = self._center(track["box"])
        new_cx, new_cy = self._center(new_box)
        track["vx"] = alpha*(new_cx-old_cx) + (1-alpha)*track.get("vx", 0.0)
        track["vy"] = alpha*(new_cy-old_cy) + (1-alpha)*track.get("vy", 0.0)

    # ── active-track matching ────────────────────────────────────────

    def _match(self, dets, track_indices):
        matched = {}
        used    = set()
        for di, (box, _) in enumerate(dets):
            best_score, best_ti = 0.0, -1
            det_cx, det_cy = self._center(box)
            for ti in track_indices:
                if ti in used:
                    continue
                t      = self.tracks[ti]
                iou_v  = self._iou(box, t["box"])
                tcx, tcy = self._center(t["box"])
                dist   = ((det_cx-tcx)**2 + (det_cy-tcy)**2) ** 0.5
                diag   = self._box_diag(t["box"])
                dist_s = max(0.0, 1.0 - dist / diag)
                score  = 0.4 * iou_v + 0.6 * dist_s
                if score > best_score:
                    best_score, best_ti = score, ti
            if best_score >= self.iou_thresh and best_ti >= 0:
                matched[di] = best_ti
                used.add(best_ti)
        return matched

    # ── backtracking ─────────────────────────────────────────────────

    def _backtrack_score(self, det_box, grave):
        iou_s = self._iou(det_box, grave["box"])

        pred_cx, pred_cy = self._predict_center(grave)
        det_cx,  det_cy  = self._center(det_box)
        dist  = ((det_cx-pred_cx)**2 + (det_cy-pred_cy)**2) ** 0.5
        norm  = self._box_diag(grave["box"])
        vel_s = max(0.0, 1.0 - dist / norm)

        vx, vy = grave.get("vx", 0.0), grave.get("vy", 0.0)
        speed  = (vx**2 + vy**2) ** 0.5
        if speed > 0.5:
            gcx, gcy = self._center(grave["box"])
            dx, dy   = det_cx - gcx, det_cy - gcy
            d_len    = (dx**2 + dy**2) ** 0.5 + 1e-6
            dot      = (dx*vx + dy*vy) / (d_len * speed)
            angle_s  = max(0.0, dot)
        else:
            angle_s  = 0.5

        return 0.25*iou_s + 0.50*vel_s + 0.25*angle_s

    def _try_backtrack(self, unmatched_dets, high_dets):
        resurrected = {}
        used_graves = set()
        for di in unmatched_dets:
            det_box = high_dets[di][0]
            best_score, best_gi = 0.0, -1
            for gi, grave in enumerate(self.graveyard):
                if gi in used_graves:
                    continue
                s = self._backtrack_score(det_box, grave)
                if s > best_score:
                    best_score, best_gi = s, gi
            if best_score >= self.backtrack_thresh and best_gi >= 0:
                resurrected[di] = best_gi
                used_graves.add(best_gi)
        return resurrected, used_graves

    # ── main update ──────────────────────────────────────────────────

    def update(self, boxes, scores):
        self.frame_id += 1

        high   = [(b, s) for b, s in zip(boxes, scores) if s >= self.high_thresh]
        active = list(range(len(self.tracks)))

        # Stage 1 — match detections → active tracks
        m1 = self._match(high, active)
        for di, ti in m1.items():
            self._update_velocity(self.tracks[ti], high[di][0])
            self.tracks[ti].update({
                "box":   high[di][0],
                "score": high[di][1],
                "age":   0,
                "hits":  self.tracks[ti]["hits"] + 1,
            })

        unmatched_t = [i for i in active if i not in m1.values()]
        unmatched_d = [i for i in range(len(high)) if i not in m1]

        for ti in unmatched_t:
            self.tracks[ti]["age"] += 1

        # expire aged-out tracks → graveyard
        alive = []
        for t in self.tracks:
            if t["age"] <= self.max_age:
                alive.append(t)
            else:
                t["dead_frame"] = self.frame_id
                self.graveyard.append(t)
        self.tracks = alive

        # prune stale graveyard entries
        self.graveyard = [
            g for g in self.graveyard
            if (self.frame_id - g.get("dead_frame", self.frame_id)) <= self.backtrack_window
        ]

        # Stage 2 — backtrack unmatched detections → graveyard
        resurrected, used_graves = self._try_backtrack(unmatched_d, high)

        for di, gi in resurrected.items():
            grave = self.graveyard[gi]
            self._update_velocity(grave, high[di][0])
            grave.update({
                "box":        high[di][0],
                "score":      high[di][1],
                "age":        0,
                "hits":       grave["hits"] + 1,
                "dead_frame": None,
            })
            self.tracks.append(grave)

        for gi in sorted(used_graves, reverse=True):
            self.graveyard.pop(gi)

        # Stage 3 — spawn new tracks for genuinely new detections
        for di in unmatched_d:
            if di not in resurrected:
                self.tracks.append({
                    "id":         self.next_id,
                    "box":        high[di][0],
                    "score":      high[di][1],
                    "age":        0,
                    "hits":       1,
                    "vx":         0.0,
                    "vy":         0.0,
                    "dead_frame": None,
                })
                self.next_id += 1

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
        cx, cy = (x1+x2)//2, (y1+y2)//2

        history[tid].append((cx, cy))
        if len(history[tid]) > tail:
            history[tid].pop(0)

        # faded trail
        pts = history[tid]
        n   = len(pts)
        for i in range(1, n):
            alpha     = i / n
            thickness = max(1, int(alpha * 3))
            blend     = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pts[i-1], pts[i], blend, thickness)

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ID label on filled background strip
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

def run_tracking(args):
    if not os.path.exists(args.source):
        raise ValueError(f"Source folder not found: {args.source}")

    files = sorted([f for f in os.listdir(args.source)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if not files:
        raise ValueError("No images found in source folder")

    paths  = [os.path.join(args.source, f) for f in files]
    sample = cv2.imread(paths[0])
    if sample is None:
        raise ValueError("Failed to read first image")

    h, w = sample.shape[:2]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    print(f"Source  : {args.source}  ({len(paths)} frames)")
    print(f"Output  : {args.output}")
    print(f"Frame   : {w}x{h}")

    detector = ONNXDetector(
        model_path    = args.model,
        imgsz         = args.imgsz,
        conf          = args.conf,
        iou           = args.iou,
        slice_infer   = args.slice,
        slice_size    = args.slice_size,
        slice_overlap = args.slice_overlap,
        num_threads   = args.threads,
    )

    tracker = ByteTracker(
        max_age          = 15,
        min_hits         = 3,
        iou_thresh       = 0.10,
        high_thresh      = 0.25,
        backtrack_window = args.backtrack_window,
        backtrack_thresh = args.backtrack_thresh,
    )

    history   = defaultdict(list)
    inf_times = []

    for i, path in enumerate(paths):
        frame = cv2.imread(path)
        if frame is None:
            print(f"  skip: {path}")
            continue

        # frame-skip: on skipped frames reuse last detections
        # (tracker still steps so velocity/age stay consistent)
        if args.frame_skip > 1 and i % args.frame_skip != 0:
            tracks = tracker.update([], [])
            draw_tracks(frame, tracks, history, args.tail)
            writer.write(frame)
            continue

        boxes, scores, inf = detector.detect(frame)
        inf_times.append(inf)

        tracks = tracker.update(boxes, scores)
        draw_tracks(frame, tracks, history, args.tail)

        avg = sum(inf_times[-30:]) / min(len(inf_times), 30)
        fps = 1000 / avg * args.frame_skip   # adjust displayed FPS for skipping

        cv2.putText(
            frame,
            f"FPS:{fps:.1f}  Tracks:{len(tracks)}  Graves:{len(tracker.graveyard)}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA
        )

        writer.write(frame)

        # progress every 50 detected frames
        if len(inf_times) % 50 == 0:
            print(f"  [{len(inf_times)}/{len(paths)}] "
                  f"avg {avg:.0f} ms/frame  FPS≈{fps:.1f}  "
                  f"tiles/frame≈{len(tiles_count(args))}  "
                  f"tracks={len(tracks)}")

    writer.release()

    if inf_times:
        avg_inf = sum(inf_times) / len(inf_times)
        print(f"\nDone.")
        print(f"Avg inference : {avg_inf:.1f} ms")
        print(f"Avg FPS       : {1000/avg_inf*args.frame_skip:.2f}")
        print(f"Video saved   : {args.output}")


def tiles_count(args):
    """Estimate number of tiles per frame for progress display."""
    from math import ceil
    oh, ow = 1071, 1904   # approximate; good enough for display
    sz   = args.slice_size
    step = int(sz * (1 - args.slice_overlap))
    ny   = ceil((oh - sz) / step) + 1
    nx   = ceil((ow - sz) / step) + 1
    return range(ny * nx)


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Aerial human tracker — ONNX + sliced inference + ByteTrack + backtracking"
    )

    # model / IO
    p.add_argument("--model",            default="best.onnx")
    p.add_argument("--source",           required=True)
    p.add_argument("--output",           default="output.mp4")

    # detection
    p.add_argument("--conf",             type=float, default=0.10)
    p.add_argument("--iou",              type=float, default=0.30)
    p.add_argument("--imgsz",            type=int,   default=640)

    # sliced inference
    p.add_argument("--slice",            action="store_true", default=True)
    p.add_argument("--no-slice",         dest="slice", action="store_false")
    p.add_argument("--slice-size",       type=int,   default=960,
                                         help="Tile size — 960 gives ~4 tiles vs 12 at 640")
    p.add_argument("--slice-overlap",    type=float, default=0.15,
                                         help="Tile overlap — 0.15 is fast, 0.25 catches more edge people")

    # performance
    p.add_argument("--threads",          type=int,   default=4,
                                         help="Thread-pool workers for parallel tile inference")
    p.add_argument("--frame-skip",       type=int,   default=1,
                                         help="Run detector every N frames (1=every frame, 2=every other)")

    # tracking
    p.add_argument("--tail",             type=int,   default=12)
    p.add_argument("--backtrack-window", type=int,   default=25)
    p.add_argument("--backtrack-thresh", type=float, default=0.30)

    args = p.parse_args()
    run_tracking(args)