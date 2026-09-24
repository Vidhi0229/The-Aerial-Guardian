from ultralytics import YOLO

model = YOLO("The-Aerial-Guardian/models/best.pt")

model.export(
    format="onnx",
    imgsz=640,        
    half=False,       
    simplify=True,
    dynamic=True,     # model accept any input size, instead of locking to one fixed shape
    opset=17,
)