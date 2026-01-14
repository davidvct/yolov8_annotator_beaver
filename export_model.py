from ultralytics import YOLO
import os

model_path = r"l:\DataScience_L\David\beaver\data_for_test\model\best.pt"
output_name = "best_fixed.onnx"

print(f"Loading model from {model_path}...")
model = YOLO(model_path)

print("Exporting to ONNX...")
# Export the model
# imgsz=640 to match our inference resolution
# opset=12 is usually stable
path = model.export(format="onnx", imgsz=640, opset=12)

print(f"Export complete: {path}")
