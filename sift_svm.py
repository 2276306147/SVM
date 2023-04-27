import os
import numpy as np
import cv2
import glob
import sklearn.svm as svm
import joblib

def calcSiftFeature(img):
    #设置图像sift特征关键点最大为200
    sift = cv2.xfeatures2d.SURF_create()
    #计算图片的特征点和特征点描述
    keypoints, features = sift.detectAndCompute(img, None)
    return features

#计算词袋
def learnVocabulary(features):
    wordCnt = 50
    #criteria表示迭代停止的模式   eps---精度0.1，max_iter---满足超过最大迭代次数20
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    #得到k-means聚类的初始中心点
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 标签，中心 = kmeans(输入数据（特征)、聚类的个数K,预设标签，聚类停止条件、重复聚类次数、初始聚类中心点
    compactness, labels, centers = cv2.kmeans(features, wordCnt, None,criteria, 20, flags)
    return centers

#计算特征向量
def calcFeatVec(features, centers):
    featVec = np.zeros((1, 50))
    for i in range(0, features.shape[0]):
        #第i张图片的特征点
        fi = features[i]
        diffMat = np.tile(fi, (50, 1)) - centers
        #axis=1按行求和，即求特征到每个中心点的距离
        sqSum = (diffMat**2).sum(axis=1)
        dist = sqSum**0.5
        #升序排序
        sortedIndices = dist.argsort()
        #取出最小的距离，即找到最近的中心点
        idx = sortedIndices[0]
        #该中心点对应+1
        featVec[0][idx] += 1
    return featVec

#建立词袋
def build_center(path):
    cate=[path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    features = np.float32([]).reshape(0, 64)#存放训练集图片的特征
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            # print(im)
            img=cv2.imread(im)
            #获取图片sift特征点
            img_f = calcSiftFeature(img)
            #特征点加入训练数据
            features = np.append(features, img_f, axis=0)
    #训练集的词袋
    centers = learnVocabulary(features)
    #将词袋保存
    filename = "./svm_centers.npy"
    np.save(filename, centers)
    print('词袋:',centers.shape)

#计算训练集图片特征向量
def cal_vec(path):
    centers = np.load("./svm_centers.npy")
    data_vec = np.float32([]).reshape(0, 50)#存放训练集图片的特征
    labels = np.float32([])
    cate=[path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            #print('reading the images:%s'%(im))#im表示某张图片的路径
            img=cv2.imread(im)
            # print(im)
            #获取图片sift特征点
            img_f = calcSiftFeature(img)
            img_vec = calcFeatVec(img_f, centers)
            data_vec = np.append(data_vec,img_vec,axis=0)
            labels = np.append(labels,idx)
    print('data_vec:',data_vec.shape)
    print('image features vector done!')
    return data_vec,labels

#训练SVM分类器
def SVM_Train(data_vec,labels):
    #设置SVM模型参数
    clf = svm.SVC(decision_function_shape='ovo')
    #利用x_train,y_train训练SVM分类器，获得参数
    clf.fit(data_vec,labels)
    joblib.dump(clf, "./svm_model.m")

#SVM分类器测试测试集正确率
def SVM_Test(path):
    #读取SVM模型
    clf = joblib.load("./svm_model.m")
    #读取词袋
    centers = np.load("./svm_centers.npy")
    #计算每张图片的特征向量
    data_vec,labels = cal_vec(path)
    res = clf.predict(data_vec)
    num_test = data_vec.shape[0]
    print(num_test)
    acc = 0
    for i in range(num_test):
        if labels[i] == res[i]:
            acc = acc+1
    print('acc: ' + str(acc) + '/' + str(num_test) + '=' + str(acc/num_test))
    return acc/num_test,res

if __name__ == "__main__":
    # train_path = './big/train_big'
    # test_path = './big/val_big'
    train_path = './train'
    test_path = './test'
    #建立词袋
    build_center(train_path)
    #构建训练集特征向量
    data_vec,labels = cal_vec(train_path)
    #将特征向量和标签输入到SVM分类器中
    SVM_Train(data_vec,labels)
    # print(x_train.shape)
    # print(y_train)
    #计算测试集的正确率
    acc,res = SVM_Test(test_path)
    # print(acc)
    print(res)
