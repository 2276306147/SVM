# 训练用代码
import cv2
import numpy as np

import os


def get_files(file_dir):
    # 存放图片类别和标签的列表：第0类
    list_1 = []
    label_1 = []

    for file in os.listdir(file_dir):  # 获得file_dir路径下的全部文件名
        # print(file)
        # 拼接出图片文件路径
        image_file_path = os.path.join(file_dir, file)
        for image_name in os.listdir(image_file_path):
            # print('image_name',image_name)
            # 图片的完整路径
            image_name_path = os.path.join(image_file_path, image_name)
            # print('image_name_path',image_name_path)
            # 将图片存放入对应的列表
            image_name_path = cv2.imread(image_name_path)
            image_name_path = cv2.resize(image_name_path, (64, 64))
            image_name_path = cv2.cvtColor(image_name_path, cv2.COLOR_BGR2GRAY)
            hist = hog.compute(image_name_path, (1, 1), (0, 0))
            image_name_path = hist.reshape(-1)
            if image_file_path[-1:] == '1':
                list_1.append(image_name_path)
                label_1.append('1')
            elif image_file_path[-1:] == '2':
                list_1.append(image_name_path)
                label_1.append('2')
            elif image_file_path[-1:] == '3':
                list_1.append(image_name_path)
                label_1.append('3')
            elif image_file_path[-1:] == '4':
                list_1.append(image_name_path)
                label_1.append('4')
            elif image_file_path[-1:] == '5':
                list_1.append(image_name_path)
                label_1.append('5')
            elif image_file_path[-1:] == '6':
                list_1.append(image_name_path)
                label_1.append('6')
            elif image_file_path[-1:] == '7':
                list_1.append(image_name_path)
                label_1.append('7')
            elif image_file_path[-1:] == '8':
                list_1.append(image_name_path)
                label_1.append('8')
            elif image_file_path[-1:] == '9':
                list_1.append(image_name_path)
                label_1.append('9')
            elif image_file_path[-1:] == '0':
                list_1.append(image_name_path)
                label_1.append('0')

    # 合并数据
    image_list = list_1
    label_list = label_1
    # 利用shuffle打乱数据
    image_list = np.array(image_list)
    label_list = np.array(label_list)
    # temp = np.array([image_list, label_list])
    # temp = temp.transpose()  # 转置
    # np.random.shuffle(temp)
    #
    # # 将所有的image和label转换成list
    # image_list = list(temp[:, 0])
    # image_list = [i for i in image_list]
    # label_list = list(temp[:, 1])
    # label_list = [int(float(i)) for i in label_list]
    # print(image_list)
    # print(label_list)
    return image_list, label_list


if __name__ == '__main__':
    winSize = (64, 64)
    blockSize = (16, 16)  # 105
    blockStride = (8, 8)  # 4 cell
    cellSize = (8, 8)
    nBin = 9  # 9 bin 3780
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBin)
    # 变换数据的形状并归一化
    # 训练图片与标签
    print("加载训练图片地址")
    train_dir = 'svm_64x64'
    print("完成")
    print("加载测试图片地址")
    test_dir = 'data'
    print("完成")
    # 测试图片与标签
    print("分类图片与标签")
    train_images, train_labels = get_files(train_dir)
    test_images, test_labels = get_files(test_dir)
    print("完成")
    featureArray_train = np.zeros((16525, 1764), np.float32)
    # train_images = train_images.reshape(train_images.shape[0], -1)  # (60000, 784)
    # train_images = train_images.astype('float32') / 255
    #
    # test_images = test_images.reshape(test_images.shape[0], -1)
    # test_images = test_images.astype('float32') / 255
    #
    # # 将标签数据转为int32 并且形状为(60000,1)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    train_labels = train_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)

    # 创建svm模型
    svm = cv2.ml.SVM_create()
    # 设置类型为SVM_C_SVC代表分类
    svm.setType(cv2.ml.SVM_C_SVC)
    # 设置核函数
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # 设置其它属性
    svm.setGamma(3)
    svm.setDegree(3)
    # 设置迭代终止条件
    # 迭代次数，精度
    svm.setTermCriteria((cv2.TermCriteria_MAX_ITER, 40000, 1e-3))
    # 训练
    print("开始训练")
    svm.train(train_images, cv2.ml.ROW_SAMPLE, train_labels)
    svm.save('10svm_100.xml')
    print("训练完成，已保存")
    print(("开始测试准确度"))

    # 在测试数据上计算准确率
    # 进行模型准确率的测试 结果是一个元组 第一个值为数据1的结果
    test_pre = svm.predict(test_images)
    test_ret = test_pre[1]

    # 计算准确率
    test_ret = test_ret.reshape(-1, )
    test_labels = test_labels.reshape(-1, )
    test_sum = (test_ret == test_labels)
    acc = test_sum.mean()
    print(acc)
    print("完成")
