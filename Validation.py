import sys

# Add the path to the YOLOv5 repository to the system path
yolo_path = r"C:\Users\habib\PycharmProjects\yolov5directory\yolov5-master"
sys.path.append(yolo_path)
# Import the run function from val.py
from val import run

def main():
    # Load your trained model
    model_path = r"C:\Users\habib\PycharmProjects\yolov5directory\yolov5-master\runs\train\exp44\weights\best.pt"

    # Specify the path to the validation dataset YAML file
    val_data_path = r"C:\Users\habib\Downloads\Drone Detection.v1i.yolov5pytorch\data.yaml"
    # Run validation
    run(weights=model_path, data=val_data_path)
if __name__ == '__main__':
    main()
