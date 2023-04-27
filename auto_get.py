# 本文件将随机截取roi

import cv2
import random

img = cv2.imread('2.jpg')
i = 5000
while True:
    h, w = img.shape[0:2]
    print(h, w)
    c = random.randint(0, w - 108)
    d = 108
    e = random.randint(0, h - 80)
    f = 80
    print(c, c + d)
    print(e, e + f)
    roi = img[e:e + f, c:c + d]
    h, w = roi.shape[0:2]
    if h * w > 0:
        cv2.imwrite('./data/roi' + str(i) + '.jpg', roi)
        i += 1
    cv2.waitKey(1)
