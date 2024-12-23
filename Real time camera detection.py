import cv2
import torch

# Load your trained YOLOv5 model
model_path = r"C:\Users\habib\PycharmProjects\yolov5directory\yolov5-master\runs\train\exp44\weights\best.pt"  # Replace with the path to your model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Set frame dimensions if needed (Optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if the camera doesn't provide it
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer to save the output
output_video = cv2.VideoWriter('output_with_predictions.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Perform detection using the YOLOv5 model
    results = model(frame)  # Pass the frame to the model for predictions

    # Draw results on the frame
    for result in results.xyxy[0]:  # Get the predictions
        x1, y1, x2, y2, conf, cls = map(int, result[:4]) + [result[4].item(), int(result[5].item())]  # Unpack values

        label = f'Class {cls}: {conf:.2f}'  # Label to display

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections to the video file
    output_video.write(frame)

    # Display the frame with detections
    cv2.imshow("YOLOv5 Real-Time Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and video writer, and close OpenCV windows
cap.release()
output_video.release()
cv2.destroyAllWindows()
