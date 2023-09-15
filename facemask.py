import numpy as np
import mediapipe as mp
import time
import cv2

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

face_img = cv2.imread('face_mk.png', cv2.IMREAD_UNCHANGED)


def face_overlay(background_img, img_to_overlay, x, y, overlay_size=None):
    try:
        bg_img = background_img.copy()
        ov_img = img_to_overlay.copy()

        # bgr로 들어오는 이미지를 rgba로 수정
        if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        # face 이미지 사이즈가 재설정 되었으면 재설정된 이미지로 변경
        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(ov_img, overlay_size)

        # 마스크 사용을 위해 채널 분리
        b, g, r, a = cv2.split(img_to_overlay_t)

        # 마스크 블러처리
        mask = cv2.medianBlur(a, 5)

        # 얼굴 영역 중심정 설정
        h, w, _ = img_to_overlay_t.shape

        i_s = int(y - h / 2)
        i_e = int(y + h / 2)
        c_s = int(x - w / 2)
        c_e = int(x + w / 2)
        if i_s < 0:
            i_s = 0
        if i_e > bg_img.shape[0]:
            i_e = bg_img.shape[0]
        if c_s < 0:
            c_s = 0
        if c_e > bg_img.shape[1]:
            c_e = bg_img.shape[1]
        roi = bg_img[i_s:i_e, c_s:c_e]

        print(roi.shape, mask.shape)

        img1_bg = cv2.bitwise_and(
            roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(
            img_to_overlay_t, img_to_overlay_t, mask=mask)

        # 캠이미지에 마스크영역과 마스크를 제외해서 더한 결과값을 갱신
        bg_img[i_s:i_e, c_s:c_e] = cv2.add(img1_bg, img2_fg)

        # convert 4 channels to 4 channels
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
        return bg_img
    except:
        return background_img


while True:
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)
    ih, iw, ic = img.shape

    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            xy_point = []

            for c, lm in enumerate(faceLms.landmark):
                xy_point.append([lm.x, lm.y])
                # img = cv2.circle(img,(int(lm.x*iw), int(lm.y*ih)),1,(255,0,0),3)

            top_left = np.min(xy_point, axis=0)
            bottom_right = np.max(xy_point, axis=0)
            mean_xy = np.mean(xy_point, axis=0)

            # img = cv2.circle(img, (int(mean_xy[0]*iw), int(mean_xy[1]*ih)),4,(0,0,255),3)

            face_width = int(bottom_right[0]*iw) - int(top_left[0]*iw)
            face_height = int(bottom_right[1] * ih) - int(top_left[1] * ih)

            if face_width > 0 and face_height > 0:
                result = face_overlay(img, face_img, int(
                    mean_xy[0]*iw), int(mean_xy[1]*ih), (face_width, face_height))

    try:
        cv2.imshow('face', result)
    except:
        cv2.imshow('face', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
