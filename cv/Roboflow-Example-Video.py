#One example on video
'''
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="yolov8x-1280",
    video_reference="https://media.roboflow.com/inference/people-walking.mp4", #Can be from local computer
    on_prediction=render_boxes
)

pipeline.start()
pipeline.join()
'''
# Or follow the bottom which would be more ideal (less knowledge fo pipeline)


import pickle
import os
import numpy as np
import pandas as pd
import cv2
import base64
import supervision as sv
from roboflow import Roboflow
from inference_sdk import InferenceHTTPClient
import videoUtils # From this github

SOURCE_VIDEO_PATH = "video-path"
TARGET_VIDEO_PATH = "video-path"
API_KEY = ""
VERSION_NUMBER = 1  # version number doesn't matter for you
STUBS_PATH = "stub.pkl"  # Pickle file to save/load detections
FRAMES_STUB_PATH = "frames_stub.pkl";
PROJECT_NAME = "your_project_name"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT_NAME)
model = project.version(VERSION_NUMBER).model

client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)
model_id = f"{PROJECT_NAME}/{VERSION_NUMBER}"

# Function to save detections using pickle
def save_detections(detections, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(detections, f)

# Function to load detections using pickle
def load_detections(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

# Read frames 
if os.path.exists(STUB_FRAMES_PATH):
        try:
            with open(STUB_FRAMES_PATH, 'rb') as f:
                frames = pickle.load(f)
            print("Frames loaded successfully from stub")
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading frames from stub: {e}")
            frames = None
else:
        print("Reading frames from video...")
        frames = videoUtils.read_video(SOURCE_VIDEO_PATH)
        if isinstance(frames, np.ndarray):
            with open(STUB_FRAMES_PATH, 'wb') as f:
                pickle.dump(frames, f)
            print(f"Frames stored in {STUB_FRAMES_PATH}")
        else:
            print("Error: Invalid frames data. Unable to process video.")
            frames = None


# Check if detections have already been saved to avoid rerunning inference
detections = load_detections(STUBS_PATH)
if detections is None:
    
    print("No saved detections found. Running inference...")
    detections = []
    for frame in frames:
        _, encoded_image = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(encoded_image).decode('utf-8')
        response = client.infer(base64_image, model_id=model_id)
        detections.append(response)

    # Save detections after running inference
    save_detections(detections, STUBS_PATH)
    print(f"Detections saved to {STUBS_PATH}.")
else:
    print(f"Loaded detections from {STUBS_PATH}.")

# Now you can process the `detections` list as needed

