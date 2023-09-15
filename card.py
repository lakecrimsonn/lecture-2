import cv2
import numpy as np
# from google.colab.patches import cv2_imshow  # 코랩에서 돌리는 경우

src = cv2.imread('image\\business-card_640 .jpg')
print('카드 사이즈:', src.shape)

# 카드 사이즈 변환해주기
w, h = 720, 480
srcQuad = np.array([[160, 207], [395, 112], [487, 221], [
                   239, 327]], np.float32)  # 그림판으로 찍어본 현재 좌표값
dstQuad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]],
                   np.float32)  # 이동하는 곳의 좌표값

pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)  # 원근변환

dst = cv2.warpPerspective(src, pers, (w, h))  # 적용
cv2.imshow("img", dst)
cv2.waitKey()
cv2.destroyAllWindows()
