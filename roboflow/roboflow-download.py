import torch
import ultralytics
from ultralytics import YOLO
from roboflow import Roboflow

print(torch.cuda.is_available())
print(ultralytics.checks())

rf = Roboflow(api_key="zWZQsL4B3tBSv4r6RWnj")
project = rf.workspace("school-igyn6").project("models-onsb5")
dataset = project.version(1).download("yolov8")
