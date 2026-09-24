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
| Model Size | 44.6 MB (ONNX) |
| Training Tiles | 11,205 (from 4 sequences) |
| Tile Size (training only) | 640×640 with 20% overlap |

## Setup

```bash
pip install ultralytics onnxruntime opencv-python numpy
```

### Project Structure

```
The-Aerial-Guardian/
├── best.onnx              # Exported inference model (44.6 MB)
├── best.pt                # Trained weights (89.6 MB)
├── track.py               # Tracking + visualization pipeline 
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
## Inference & Tracking Pipeline

The Aerial Guardian uses a fine-tuned YOLOv8s model for person detection and a lightweight IoU-based tracker for multi-object tracking in aerial drone footage.

### Pipeline Overview

1. **Frame Acquisition**
   - Input frames are loaded from a drone video sequence.

2. **Preprocessing**
   - Frames are resized using letterbox preprocessing while preserving aspect ratio.

3. **Person Detection**
   - YOLOv8s ONNX performs full-frame inference.
   - Low-confidence detections are filtered using a configurable confidence threshold.

4. **Multi-Object Tracking**
   - Detections are associated across frames using greedy IoU matching.
   - Existing tracks are updated when matched with new detections.
   - New detections create new track IDs.
   - Tracks that remain unmatched for a specified number of frames are removed.

5. **Visualization**
   - Bounding boxes, track IDs, and trajectory tails are rendered on each frame.
   - The final tracked video is exported as an MP4 file.


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
python3 compress.py
```

### 4. Evaluate

```bash
python3 eval.py
```

### 5. Run Tracking on a Sequence

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
| **YOLOv8s (small)** | **43 MB** | **Fast** | **chosen** |
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
### 4. Tracking — IoU-Based Multi-Object Tracker

**Algorithm:** A lightweight multi-object tracker implemented in NumPy and OpenCV. The tracker associates detections across consecutive frames using Intersection over Union (IoU) to maintain consistent identities over time.

#### Matching Strategy

Detections are first filtered using the `HIGH_THRESH` confidence threshold. Each detection is then matched to the existing track with the highest IoU above the configured `iou_thresh` value. Matched tracks are updated with the new bounding box and their age is reset.

Tracks that do not receive a matching detection have their age incremented. If a track remains unmatched for more than `max_age` frames, it is removed. Any unmatched detection initializes a new track with a unique ID. Tracks are displayed only after reaching the configured `MIN_HITS` threshold, reducing short-lived false positives.

#### Letterbox Preprocessing

Input frames are resized using aspect-ratio-preserving letterbox preprocessing before inference. Padding offsets and scaling factors are recorded and later used during postprocessing to map detections back to the original image coordinates accurately.

#### Trajectory Visualization

Each track maintains a history of centroid positions across frames. These points are used to render trajectory tails, providing a visual representation of movement over time. Tails are drawn with gradually increasing thickness and opacity from oldest to newest positions, improving motion visualization while preserving the original tracking data.

#### Output

For each frame, the pipeline generates:

- Person bounding boxes
- Unique track IDs
- Detection confidence scores
- Motion trajectory tails

The final output is an annotated video showing tracked individuals throughout the sequence.

---

## Authors
**Vidhi Srivastava**

[Linkedin](https://www.linkedin.com/in/vidhisrivastava01/)

[Github](https://github.com/Vidhi0229)