from ultralytics import YOLO
import time

model = YOLO("models/best.onnx", task="detect")

metrics = model.val(
    data="data.yaml",
    imgsz=640,        
    batch=1,
    conf=0.25,
    iou=0.5,
    device="cpu",
    verbose=True,
)

print(f"mAP50:       {metrics.box.map50:.4f}")
print(f"mAP50-95:    {metrics.box.map:.4f}")
print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")
print(f"F1 Score:    {2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr):.4f}")
print(f"Inference:   {metrics.speed['inference']:.1f}ms/img = {1000/metrics.speed['inference']:.1f} FPS")
