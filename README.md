# The Aerial Guardian
### Human Detection & Tracking Pipeline for Aerial Drone Footage
**Fine-tuned YOLOv8s + a lightweight IoU-based multi-object tracker for the VisDrone2019 MOT dataset.**

## Results

![Demo GIF](output.gif)

| Metric | Value |
|--------|-------|
| mAP50 | 0.94 |
| mAP50-95 | 0.78 |
| Precision | 0.96 |
| Recall | 0.93 |
| F1 Score | 0.95 |
| Inference FPS | ~2.8 FPS (CPU, ONNX FP32, 1920×1080 source, single-pass, no tiling) |
| Avg inference time | ~357 ms/frame |
| Model Size | 22 MB (ONNX, FP32) |
| Training Tiles | 11,205 (from 4 sequences) |
| Tile Size (training only) | 640×640 with 20% overlap |

> Detection accuracy (mAP/precision/recall/F1) comes from `eval.py` on the tiled validation set and is unaffected by the tracker simplification below. The FPS figure is a real measurement from running the current `track.py` on a 312-frame sequence, replacing earlier estimates that assumed tiled + multithreaded test-time inference, which has since been removed (see "Pipeline Simplifications").

## Setup

```bash
pip install ultralytics onnxruntime opencv-python numpy
```

### Project Structure

```
The-Aerial-Guardian/
├── best.onnx              # Exported inference model (22 MB)
├── best.pt                # Trained weights (86 MB)
├── track.py               # Tracking + visualization pipeline (single-pass, no CLI args — see Usage)
├── eval.py                # Evaluation script
├── split.py               # YOLO labeling + tiling + train/val split
├── train.py                # Training script
├── data.yaml               # Dataset config
├── sequences/               # VisDrone image sequences
│   └── uav0000086_00000_v/
│       ├── 0000001.jpg
│       └── ...
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── output/
```

---

## Pipeline Simplifications

The original implementation used sliced/tiled inference with a thread pool at test time, plus a two-stage ByteTrack-style tracker with velocity estimation and graveyard-based ID resurrection. That version has been replaced with a simpler pipeline:

| Removed | Replaced with |
|---|---|
| Sliced ("SAHI"-style) tiled inference at test time, run across a thread pool | A single full-frame inference pass per image |
| Two-stage association (high/low confidence, graveyard, backtracking) | Single-stage greedy IoU matching |
| Exponentially-smoothed per-track velocity + motion-angle scoring | No velocity — tracks are matched on IoU only |
| CLI arguments (`--model`, `--source`, `--output`, `--imgsz`, `--tail`, etc.) | A `CONFIG` block edited directly at the top of `track.py` |

**Trade-offs to know about:** without tiling, people who are very small/far in a 1920×1080 frame are more likely to be missed, since the whole frame is downscaled to the model's input size before inference. Without the graveyard/backtracking stage, a person who is briefly occluded (e.g. during a camera pan) will be assigned a *new* ID rather than resuming their old one. These were deliberate simplifications made for a smaller, easier-to-read codebase — see `HIGH_THRESH` and `IMGSZ` in `track.py`'s `CONFIG` block if you want to tune the accuracy/robustness trade-off back up.

---

## Dataset

Download the [VisDrone2019 MOT Validation Set](https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view?usp=sharing) and place the extracted sequences under `sequences/`.

The 4 sequences used:

| Sequence | Frames | Altitude |
|----------|--------|---------|
| uav0000086_00000_v | 2,784 | High |
| uav0000117_02622_v | 4,307 | Medium |
| uav0000137_00458_v | 4,096 | Medium |
| uav0000339_00001_v | 1,994 | Low |

---

## Usage

### 1. Prepare Training Data

```bash
python3 split.py
```

This runs three stages in sequence:
- **Stage 1**: Converts VisDrone annotations to YOLO format (normalized cx, cy, w, h)
- **Stage 2**: Tiles each annotated frame into 640×640 patches with 20% overlap
- **Stage 3**: Stratified 85/15 train/val split, sampling 15% from each sequence independently

Note: this tiling is a **training-data preparation step only**. It is not used at inference time in the current `track.py`.

### 2. Train

```bash
python3 train.py
```

### 3. Export to ONNX

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(
    format='onnx',
    imgsz=640,
    half=False,     # CPU inference (CPUExecutionProvider) gets no speed
                     # benefit from fp16, and track.py sends float32
                     # tensors — half=True causes a dtype mismatch
    simplify=True,
    dynamic=True,    # accepts any input size at inference (e.g. if you
                      # change IMGSZ in track.py later); dynamic=False
                      # locks the graph to a fixed 640x640 input
    opset=17,
)
"
```

### 4. Evaluate

```bash
python3 eval.py
```

### 5. Run Tracking on a Sequence

`track.py` no longer takes command-line flags — open the file and edit the `CONFIG` block near the top:

```python
MODEL       = "best.onnx"
SOURCE      = "sequences/uav0000297_02761_v"
OUTPUT      = "output/tracked.mp4"

CONF        = 0.10
IOU         = 0.30
IMGSZ       = 640          # must match what the model was exported/trained for
                            # unless it was exported with dynamic=True

HIGH_THRESH = 0.10          # min detection score to become/update a track
MIN_HITS    = 2             # consecutive matches before a track is drawn

FRAME_SKIP  = 1
TAIL        = 20
```

Then run:

```bash
python3 track.py
```

Output video shows: bounding boxes, unique ID labels, and trajectory tails per tracked person.

---

## Technical Report

### 1. Architecture — YOLOv8s

**Base model:** YOLOv8s (small variant, 11M parameters, 28.6 GFLOPs, 43 MB ONNX)

**Why YOLOv8s specifically:**

| Model | Size | Speed | Decision |
|-------|------|-------|----------|
| YOLOv8n (nano) | ~6 MB | Very fast | Insufficient capacity for dense aerial crowds |
| **YOLOv8s (small)** | **43 MB** | **Fast** | **Sweet spot — chosen** |
| YOLOv8m (medium) | ~100 MB | Moderate | Exceeds 300 MB? No, but too slow on CPU |
| YOLOv8l (large) | ~200 MB | Slow | Too slow for deployment |

**Architecture internals:**

```
Backbone (CSPDarknet with C2f blocks)
│
│  C2f = Cross-Stage Partial with faster flow
│  Each C2f splits features into two paths:
│   - Path 1: passes through n Bottleneck blocks (deep features)
│   - Path 2: bypasses directly (gradient highway)
│  Both paths concatenated → richer gradients, less compute than C3
│
├── P3 (stride 8)  → small object features  ← most critical for aerial
├── P4 (stride 16) → medium object features
└── P5 (stride 32) → large object features

Neck (FPN + PAN)
│  Feature Pyramid Network — top-down: high-level semantics → small scales
│  Path Aggregation Network — bottom-up: fine spatial detail → large scales
│  Result: every detection head sees both fine detail and global context

Head (Decoupled Detect)
│  Separate branches for classification and box regression
│  Eliminates conflict between the two objectives
│  DFL (Distribution Focal Loss) models box edges as distributions
│  Output: [cx, cy, w, h, conf] per anchor point across 8400 candidates
```

**Key training decisions for aerial imagery:**

| Hyperparameter | Value | Reasoning |
|----------------|-------|-----------|
| `imgsz` | 1024px train / 640px infer | Larger training res preserves small human detail; infer at 640 for speed |
| `flipud` | 0.5 | No gravity-up bias in aerial view — humans can appear from any orientation |
| `fliplr` | 0.5 | Standard horizontal flip augmentation |
| `degrees` | 0.0 | Rotation destroys axis-aligned YOLO box labels — disabled entirely |
| `mosaic` | 0.5 | Full mosaic creates unrealistic crowd densities from tiled data |
| `close_mosaic` | 20 | Disables mosaic in last 20 epochs to stabilize convergence |
| `single_cls` | True | Only detecting persons; collapses all person categories to class 0 |
| `optimizer` | AdamW | Better convergence than SGD on this task size |
| `batch` | 16 | Stable gradients — batch=8 caused recall oscillation in early runs |
| `lrf` | 0.1 | Less aggressive LR decay than default 0.01 |
| `cos_lr` | True | Cosine annealing schedule for smoother decay |
| `warmup_epochs` | 5 | Gradual warm-up prevents early instability |
| `erasing` | 0.3 | Random erasing simulates partial occlusion — relevant for drone crowds |

---

### 2. Tiling Pipeline — Small Object Handling (Training Only)

Aerial images from VisDrone are 1920×1080. At full scale, humans are only ~30px wide. Directly resizing to 640×640 shrinks them further, making them nearly undetectable.

**Solution: Sliding window tiling before training** (this happens in `split.py`; it is not repeated at inference time)

```
Original frame (1920×1080)
        │
        |
Sliding window: 640×640 tiles, 20% overlap, stride=512px
        │
        ├── Tile (x=0,    y=0)    → only saved if contains ≥1 annotation
        ├── Tile (x=512,  y=0)    → only saved if contains ≥1 annotation
        ├── Tile (x=1024, y=0)    → only saved if contains ≥1 annotation
        ├── Tile (x=1280, y=0)    → forced last tile (edge coverage)
        └── ... (all y positions)
```

**Tiling parameters:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| Tile size | 640×640 | Matches training resolution |
| Overlap | 20% | Ensures objects near tile borders are not missed |
| Stride | 512 px | Derived from tile size and overlap |
| Save empty tiles | No | Only tiles containing annotations are saved — reduces class imbalance |
| Frame filtering | Enabled | Frames without annotations are skipped entirely |
| Edge handling | Forced last position | Guarantees full image coverage, especially right/bottom edges |

**Label handling during tiling:**

1. Original YOLO-normalized boxes converted to absolute pixel coordinates
2. Each box intersected with the tile boundary
3. Boxes with no overlap discarded
4. Remaining boxes clipped to the tile region
5. Clipped boxes re-centered and resized within tile
6. Normalized back to tile size (640×640)

**Effect on object scale:**

| Stage | Median human width |
|-------|--------------------|
| Full frame (1920px) | ~30px |
| After tiling (640px tile) | ~50–55px |

Tiling effectively increases relative object size by ~75% during training, making small humans significantly easier for the model to learn. Because inference is single-pass, full-frame at 640×640 (see below), objects at test time are shrunk back down to roughly the un-tiled scale — this is the main source of the tiny/far-away miss rate discussed in "Pipeline Simplifications."

---

### 3. Data Split Strategy

The dataset has only 4 sequences. Naive sequence-level splitting caused a **25% scale mismatch** (train 58px vs val 44px median box width) because `uav0000086` was filmed at higher altitude than the others.

This mismatch caused recall to oscillate and collapse across all early training runs regardless of hyperparameters — the model learned 58px features but was evaluated on 44px objects.

**Solution: Stratified image-level split**

```
uav0000086 (2,784 imgs) → 85% train + 15% val   (sampled independently)
uav0000117 (4,307 imgs) → 85% train + 15% val
uav0000137 (4,096 imgs) → 85% train + 15% val
uav0000339 (1,994 imgs) → 85% train + 15% val
────────────────────────────────────────────────
Total:  11,205 train tiles | 1,976 val tiles

Result: train 51px vs val 52px median box width — essentially identical
```

---

### 4. Tracking — IoU-Only Tracker (Custom NumPy Implementation)

**Algorithm:** Single-stage greedy IoU matching, implemented from scratch in NumPy and OpenCV. Zero PyTorch or Ultralytics dependency at inference time (when using the ONNX model).

**Why IoU-only over DeepSORT / appearance-based tracking:**

| Criterion | IoU-only tracker | DeepSORT |
|-----------|-----------|----------|
| Re-ID network | None | +50–100 MB |
| Inference overhead | Negligible | High (CNN forward pass per detection) |
| Aerial resolution | Good — IoU sufficient at 40–60px | Re-ID degrades at low resolution |


At aerial resolution, humans appear at 40–60px — appearance features from a Re-ID network are too low-resolution to add meaningful signal over geometry alone, which is why IoU-only matching is a reasonable choice here. The trade-off is that this implementation does **not** attempt to recover IDs after an occlusion: there is no graveyard, no backtracking, and no velocity prediction. A track that goes unmatched for more than `max_age` frames is simply dropped, and the next detection in that area starts a brand-new ID.

**Matching:**
Detections are first filtered to those meeting `HIGH_THRESH` (default 0.10 — configurable in `CONFIG`). Each detection is then greedily matched to the existing track with the highest IoU above `iou_thresh`; each track can be claimed by at most one detection per frame. Matched tracks have their box updated and age reset to 0. Unmatched tracks have their age incremented; once age exceeds `max_age` (15 frames), the track is dropped permanently. Any detection not matched to an existing track spawns a new track with a fresh ID, which is only drawn once it accumulates `MIN_HITS` consecutive matches.

**Letterbox preprocessing:**
Images are resized with preserved aspect ratio and padded with gray (114) to the model's input size. The padding offsets (left, top) and scale factor are recorded during preprocessing and used to invert the transform in postprocessing, recovering original frame coordinates. Without this, bounding boxes on non-square footage are systematically offset from the actual person.

**Trajectory tail rendering:**
Raw centroids are appended directly to the tail history buffer — no positional averaging or smoothing is applied. Tails are rendered with linearly increasing thickness and opacity from oldest to newest point, which reduces the *visual* impact of detection jitter from drone vibration without altering the underlying track data.

---

### 5. Real-World Performance (CPU, ONNX FP32)

Measured on a 312-frame, 1920×1080 sequence with the current single-pass pipeline (`CONF=0.10`, `IOU=0.30`):

| Resolution | Format | Avg time/frame | FPS |
|------------|--------|-----------------|-----|
| 640×640 | ONNX FP32, single-pass | ~357 ms | ~2.8 |

This is measured end-to-end, including detection + tracking + drawing per frame, on CPU with no threading. It is lower than earlier estimates in this README that assumed the tiled + multithreaded test-time pipeline (since removed). If you need higher throughput, options include: re-introducing tiled inference (recovers small-object recall too, at a speed cost), running on GPU, or increasing `FRAME_SKIP`.

---

## Key Design Choices

| Trade-off | Choice | Reasoning |
|-----------|--------|-----------|
| Accuracy vs Speed | 640px inference | Matches training/export resolution; keep vs. increase per your accuracy/speed needs |
| Model size vs Capacity | YOLOv8s (43 MB) | Nano too weak, medium too slow/large |
| Re-ID vs IoU tracking | IoU only | Re-ID adds 100 MB+ with marginal gain at aerial resolution |
| Training vs inference resolution | 1024px train / 640px infer | Learn fine features, deploy at speed |
| Augmentation (rotation) | Disabled | Rotation destroys axis-aligned YOLO labels |
| Tiling at inference | Disabled | Simpler, faster single-pass pipeline; costs some small-object recall (see "Pipeline Simplifications") |
| Track ID recovery after occlusion | Disabled | Simpler tracker; re-appearing people get a new ID instead of resuming the old one |
| Batch size (training) | 16 | Batch=8 caused noisy gradients and recall oscillation |
| Empty tile saving (training) | Disabled | Reduces class imbalance — empty tiles add no signal |

---

## Enhancements Over Base Model

1. **Sliding window tiling (training only)** — 1920×1080 frames sliced into 640×640 tiles with 20% overlap before training. Increases effective human pixel width from ~30px to ~54px. Edge positions forced to guarantee full image coverage.

2. **Stratified data split** — fixed train/val scale mismatch that caused recall collapse across all baseline runs. Both splits now have matched median box widths (51px vs 52px).

3. **Lightweight IoU tracker in pure NumPy** — no PyTorch/Ultralytics dependency at inference time when using the ONNX model. Single-stage greedy IoU matching; tracks are dropped rather than resurrected after occlusion, trading some ID persistence for a much simpler, easier-to-audit implementation that still runs entirely without a GPU.

4. **Letterbox preprocessing with inverse transform** — aspect-ratio-preserving resize with gray padding. Padding offsets reversed in postprocessing for correct coordinate recovery on non-square footage. Without this, detections are systematically offset.

5. **Aerial-specific augmentation tuning** — disabled rotation (breaks axis-aligned boxes), enabled flipud=0.5 (no orientation bias in aerial view), reduced mosaic to 0.5 (prevents unrealistic crowd densities from tiled data).

6. **Two-resolution training strategy** — train at 1024px for fine feature learning, export and infer at 640px (or higher, if the ONNX model is exported with `dynamic=True`).

7. **Trajectory tail rendering** — raw centroids stored directly (no positional smoothing applied); tails drawn with increasing thickness/opacity from oldest to newest point to visually reduce the perceived impact of frame-to-frame detection jitter.

---

## Reproducing from Scratch

```bash
# Step 1: Prepare data (tiling + labels + split)
python3 split.py

# Step 2: Train
python3 train.py

# Step 3: Export to ONNX
python3 -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', imgsz=640, half=False, simplify=True, dynamic=True, opset=17)
"

# Step 4: Evaluate
python3 eval.py

# Step 5: Run tracking — edit the CONFIG block in track.py first, then:
python3 track.py
```

---

## Authors
**Vidhi Srivastava**

[Linkedin](https://www.linkedin.com/in/vidhisrivastava01/)

[Github](https://github.com/Vidhi0229)