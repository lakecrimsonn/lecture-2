import torch
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np

image = cv2.imread(
    'C:\\Users\\user\\PycharmProjects\\lecture\\roboflow\\henry.jpg')
image = cv2.resize(image, (400, 500))
model = YOLO(
    'C:\\Users\\user\\PycharmProjects\\lecture\\roboflow\\runs\detect\\train7\\weights\\best.pt')
# pre = model(source=image, show=False, save=True)
pre = model(image)
for p in pre:
    boxes = p.boxes.data.tolist()  # 텐서를 리스트로 변환
    for box in boxes:
        box[0] = int(box[0])
        box[1] = int(box[1])
        box[2] = int(box[2])
        box[3] = int(box[3])
        print(box[0], box[1])
        cv2.rectangle(image, (box[0], box[1]),
                      (box[2], box[3]), (255, 0, 0), 2)


cv2.imshow('img', image)
cv2.waitKey(0)
