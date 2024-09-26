#Pip install all dependencies and packages

from ultralytics import YOLO
import cv2

# Load the desired version of the YOLOv8 model pre-trained for satellite images
model = YOLO("yolov8-satellite-[version].pt")

# Load the video file
video_path = "sample_satellite_video.mp4"
video_capture = cv2.VideoCapture(video_path)

# Get video frame width, height, and fps to save the output video
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Set up the output video writer
output_video_path = "output_video.mp4"
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# Process each frame in the video
while video_capture.isOpened():
    ret, frame = video_capture.read()
    
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Get the annotated frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    video_writer.write(annotated_frame)

# Release video capture and writer objects
video_capture.release()
video_writer.release()

print("Inference completed. Annotated video saved as", output_video_path)

