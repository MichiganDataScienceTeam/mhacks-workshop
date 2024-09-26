#One example on video

from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="yolov8x-1280",
    video_reference="https://media.roboflow.com/inference/people-walking.mp4", #Can be from local computer
    on_prediction=render_boxes
)

pipeline.start()
pipeline.join()

