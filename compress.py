
from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    imgsz=640,        
    half=False,       
    simplify=True,
    dynamic=False,   
    opset=17,
)