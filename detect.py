#测试模型用代码，可以看到每张图片的输出是什么
import cv2
import numpy as np
import os

if __name__ == '__main__':
    # 读取图片
    winSize = (64, 64)
    blockSize = (16, 16)  # 105
    blockStride = (8, 8)  # 4 cell
    cellSize = (8, 8)
    nBin = 9  # 9 bin 3780
    # 设置文件夹中数量------------需更改
    total_img = os.listdir('data\\4')
    # 载入svm模型----------------需更改模型地址和文件夹地址
    svm = cv2.ml.SVM_load('10svm_100.xml')
    for i in total_img:
        if os.path.getsize('data\\4\\' + str(i)) == 0:
            continue
        img = cv2.imread('data\\4\\' + str(i), 1)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        img_sw = img.copy()
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBin)
        hist = hog.compute(img, (1, 1), (0, 0))
        img = hist.reshape(-1)
        list1 = []
        list1.append(img)
        img = np.array(list1, np.float32)

        # 进行预测
        img_pre = svm.predict(img)
        pre = int(img_pre[1])
        print(pre)
        #cv2.imwrite('E:\python_file\\svm\\data\\'+ str(pre) +'\\'+str(i), img_sw)
        # os.remove('E:\py             thon_file\\auto_labell\svm_data\\'+str(i))

        cv2.imshow('test',img_sw)
        cv2.waitKey(0)
