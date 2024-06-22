from PIL import Image
from ultralytics import YOLO
import os

# Export the model to TensorRT

project_dir = os.getcwd()
model_path = os.path.join(project_dir, 'models/FN3-17/weights/best.pt')

model = YOLO(model_path)

model.export(format='engine', imgsz=(640, 640))  # export to TensorRT