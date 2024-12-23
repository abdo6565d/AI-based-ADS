import torch
import os

# Load your trained model
model_path = r"C:\Users\habib\PycharmProjects\yolov5directory\yolov5-master\runs\train\exp44\weights\best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Define the folder containing images for prediction
image_folder = r"C:\Users\habib\OneDrive\Pictures\Saved Pictures\droneimg.jpg"  # Change this to your folder path

# Run inference on the folder
results = model(image_folder)  # Run inference on the entire folder

# Save results
results.save(save_dir='C:/Users/habib/Downloads/yolov5_predictions')  # Directory to save predictions
