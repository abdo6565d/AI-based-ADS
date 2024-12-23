import torch
import os

# Check if GPU is available
if torch.cuda.is_available():
   device = '0'  # GPU device ID (use '0' for the first GPU)
   print("Using GPU for prediction.")
else:
   device = 'cpu'  # Use CPU if GPU is not available
   print("Using CPU for prediction.")

# Define your paths
model_weights = r"C:\Users\habib\PycharmProjects\yolov5directory\yolov5-master\runs\train\exp44\weights\best.pt"  # Path to your trained model weights
test_images = r"C:\Users\habib\Downloads\Drone Detection.v1i.yolov5pytorch\actualdata\test\images"  # Path to folder containing test images
output_dir = r"C:\Users\habib\Downloads\yolov5_predictions"  # Directory to save predictions

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Make sure to include all valid image files from the folder
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
image_files = [os.path.join(test_images, f) for f in os.listdir(test_images) if os.path.splitext(f)[1].lower() in valid_extensions]

if not image_files:
   print("No valid image files found in the test directory.")
else:
   print(f"Found {len(image_files)} image(s) for prediction.")

# Construct the prediction command
predict_command = f"""
python C:/Users/habib/PycharmProjects/yolov5directory/yolov5-master/detect.py \
--weights "{model_weights}" \
--source "{test_images}" \
--device {device} \
--save-txt --save-conf --project "{output_dir}"
"""

# Print the command (optional)
print("Prediction command:", predict_command)

# Execute the prediction command
os.system(predict_command)
