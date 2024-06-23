from PIL import Image
from ultralytics import YOLO
import os
import time
import torch

project_dir = os.getcwd()

# PyTorch model file path
model_path = os.path.join(project_dir, 'models/FN3-17/weights/best.pt')

# TensorRT model file path
# model_path = os.path.join(project_dir, 'models/FN3-17/weights/best.engine')

# Try to use the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Set the GPU to use (0, 1, 2, or 3)
    print("Using GPU")
else:
    print("Using CPU")

model = YOLO(model_path)

num = "8"

# Open the image file
img = Image.open(f'TestImages/ss{num}.jpg')
# Resize the image
img = img.resize((640, 640))
# Save the resized image
img.save(f'TestImages/resized_ss{num}.jpg')

results = model(f'TestImages/ss{num}.jpg', device=device, half=True, imgsz=(640, 640))

start_time = time.time()
# Pass the resized image to the model
resresized_results = model(f'TestImages/resized_ss{num}.jpg', device=device, half=True, imgsz=(640, 640))

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

results[0].save(filename=f'TestImages/results{num}.jpg')  # save to file
resresized_results[0].save(filename=f'TestImages/resized_results{num}.jpg')  # save to file