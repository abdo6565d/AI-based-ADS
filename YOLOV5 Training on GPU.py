import torch
import subprocess

# Check if GPU is available
if torch.cuda.is_available():
   device = 'cuda'  # Use GPU
   print("Using GPU for training.")
else:
   device = 'cpu'  # Use CPU
   print("Using CPU for training.")

# Define your dataset configuration file
data_config = r"C:\Users\habib\Downloads\Drone Detection.v1i.yolov5pytorch\data.yaml"  # Path YAML file

# Define parameters
weights = 'yolov5n.pt'  # Pre-trained weights (can also use yolov5m.pt, etc.)
epochs = 100  # Number of epochs to train

# Construct the training command
train_command = (f"python C:/Users/habib/PycharmProjects/yolov5directory/yolov5-master/train.py --data "
                f"\"{data_config}\" --weights {weights} --epochs {epochs} --device 0")

# Print the command (optional)
print("Training command:", train_command)

# Execute the training command using subprocess
result = subprocess.run(train_command, shell=True)

# Check the result of the command
if result.returncode == 0:
   print("Training started successfully.")
else:
   print("Error occurred during training.")