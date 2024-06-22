from PIL import Image
from ultralytics import YOLO
import os
import time

project_dir = os.getcwd()

# PyTorch model file path
model_path = os.path.join(project_dir, 'models/FN3-17/weights/best.pt')

# TensorRT model file path
# model_path = os.path.join(project_dir, 'models/FN3-17/weights/best.engine')

model = YOLO(model_path)

num = "8"

# Open the image file
img = Image.open(f'TestImages/ss{num}.jpg')
# Resize the image
img = img.resize((640, 640))
# Save the resized image
img.save(f'TestImages/resized_ss{num}.jpg')

results = model(f'TestImages/ss{num}.jpg')

start_time = time.time()
# Pass the resized image to the model
resresized_results = model(f'TestImages/resized_ss{num}.jpg')

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

results[0].save(filename=f'TestImages/results{num}.jpg')  # save to file
resresized_results[0].save(filename=f'TestImages/resized_results{num}.jpg')  # save to file