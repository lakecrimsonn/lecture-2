import pandas as pd
import numpy as np
import cv2

df = pd.read_csv('data\hand.csv', header=None)
x = df.iloc[:, :-1].to_numpy().astype(np.float32)
y = df.iloc[:, -1].to_numpy().astype(np.float32)

# cv2의 knn 머신러닝 알고리즘을 사용할 수 있다.
knn = cv2.ml.KNearest.create()
knn.train(x, cv2.ml.ROW_SAMPLE, y)
