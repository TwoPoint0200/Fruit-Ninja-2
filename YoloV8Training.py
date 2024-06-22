from ultralytics import YOLO
import torch
import os

if __name__ == '__main__':
    project_dir = os.getcwd()

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Set to primary GPU
        print("Using GPU")
        

    # Create a new YOLO model from scratch example
    # model = YOLO('yolov8n.yaml')

    # Load examples
    # model = YOLO(os.path.join(project_dir, 'models', 'FN3-8', 'weights', 'best.pt'))
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
    # model = YOLO('yolov8n.pt')

    # Load a pretrained YOLO model
    model = YOLO('yolov8n.pt')


    # Train the model using fruit ninja dataset
    data = os.path.join(project_dir, 'CustomDataset/Fruit Ninja.v12i.yolov8/data.yaml')
    model_path = os.path.join(project_dir, 'models')
    results = model.train(data=data, epochs=1000, imgsz=(640, 640), project=model_path, name='FN3-', device=device, batch=-1) 

    # Evaluate the model's performance on the training dataset
    results = model.val(data=data, imgsz=(640, 640))

    # Save the model to a file
    model.export()

    # Export the model to ONNX format
    # success = model.export(format='onnx')