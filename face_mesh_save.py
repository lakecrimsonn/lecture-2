import mediapipe as mp
import cv2
import numpy as np
import csv
from os import path

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(1)

hcnt, scnt = 0, 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        image.flags.writeable = False

        result = holistic.process(image)

        image.flags.writeable = True

        mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS, mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1))

        try:
            face = result.face_landmarks.landmark
            face_list = []
            for temp in face:
                face_list.append([temp.x, temp.y, temp.z])
            # 좌표 468개 [[xyz],[xyz],...]
            # 하나로 만들기 [xyz,xyz,xyz]
            face_row = list(np.array(face_list).flatten())

            # 파일이 없으면 생성해준다
            if not path.isfile('facedata.csv'):
                # ['class', 'x1','y1','z1',...]
                # csv의 상단 컬럼명 만들어주기
                landmarks = ['class']
                for val in range(1, len(face)+1):
                    landmarks += ['x{}'.format(val),'y{}'.format(val), 'z{}'.format(val)]
                with open('facedata.csv', mode='w', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)  # 문자열기호 지정
                    csv_writer.writerow(landmarks)
            else:
                if cv2.waitKey(1) & 0xFF == ord('1'):
                    face_row.insert(0, 'happy')  # ['happy',0.2,0.1,...]
                    with open('facedata.csv', mode='a', newline='') as f:  # a 이어서 쓰기
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)  # 문자열기호 지정
                        csv_writer.writerow(face_row)
                        hcnt += 1
                        print('save happy', hcnt)

                elif cv2.waitKey(1) & 0xFF == ord('2'):
                    face_row.insert(0, 'sad')  # ['sad',0.2,0.1,...]
                    with open('facedata.csv', mode='a', newline='') as f:  # a 이어서 쓰기
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)  # 문자열기호 지정
                        csv_writer.writerow(face_row)
                        scnt += 1
                        print('save sad', scnt)

        except:
            pass

        cv2.imshow('face', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
