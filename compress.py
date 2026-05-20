
from ultralytics import YOLO

model = YOLO("models/best.pt")

model.export(
    format="onnx",
    imgsz=640,        
    half=True,       
    simplify=True,
    dynamic=False,   
    opset=17,
)