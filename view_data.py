from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os

project_dir = os.getcwd()
data = os.path.join(project_dir, 'dataset/data.yaml')
model_path = os.path.join(project_dir, 'models/FN3-17/weights/best.pt')

# Load your trained YOLOv8 model
model = YOLO(model_path)

# Load an image
img = cv2.imread('./TestImages/ss4.jpg')  # Replace with the path to your image

# Perform inference
results = model.predict(img)

# Annotate the image with bounding boxes
annotator = Annotator(img)
for r in results:
    print(r)
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)])

# Display the annotated image
annotated_img = annotator.result()
cv2.imshow('YOLOv8 Detection', annotated_img)
cv2.waitKey(0)  # Press any key to close the window
cv2.destroyAllWindows()
