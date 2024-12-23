import cv2
import torch

# Load your trained YOLOv5 model
model_path = r"C:\Users\habib\PycharmProjects\yolov5directory\yolov5-master\runs\train\exp44\weights\best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Define the video file path
video_path = r"C:\Users\habib\Videos\Captures\onedrone.mp4"  # Change this to your video file path

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    print(f"Video path: {video_path}")
else:
    print("Video opened successfully.")
    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            print("No more frames to read.")
            break  # Exit if there are no more frames

        # Make predictions on the frame
        results = model(frame)  # Pass the frame to the model for predictions

        # Draw predictions on the frame
        annotated_frame = results.render()[0]  # Get the annotated frame with predictions

        # Display the frame with predictions
        cv2.imshow("YOLOv5 Predictions", annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
