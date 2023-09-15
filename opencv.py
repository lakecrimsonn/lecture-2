import cv2

# img = cv2.imread('image/starry_night.jpg')
# img = cv2.resize(img, (224, 224))
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyWindow()

cap = cv2.VideoCapture(1)  # 카메라의 번호

while True:
    ret, frame = cap.read()  # 리턴 값 2개
    frame = cv2.flip(frame, 1)  # 좌우반전
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:  # esc
        break

cap.release()
cv2.destroyAllWindows()








