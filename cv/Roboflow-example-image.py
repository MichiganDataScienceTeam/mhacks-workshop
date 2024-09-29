#Inference using an API Call to pretrained Roboflow model and dataset
#You must pip install inference-sdk and Ropboflow 

from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=""
)

result = CLIENT.infer(your_image.jpg, model_id="football-players-detection-3zvbc/9")


#Another example 

from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=<ROBOFLOW_API_KEY>
)
with client.use_model(model_id="soccer-players-5fuqs/1"):
    predictions = client.infer("https://media.roboflow.com/inference/soccer.jpg")

#Will be in python dictionary form
