import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(1)


while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        results = model.track(frame, persist=True)
        # model.track(source="youtube url")
        frame = results[0].plot()

    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
