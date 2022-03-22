from sklearn.decomposition import PCA
from sklearn import decomposition
import numpy as np
import cv2
import os

# path = 'E:/fjj/SeaShips_SMD/JPEGImages/'
# files = ['004091.jpg', '004100.jpg']
# for file in files:
#     img = cv2.imread(os.path.join(path, file),0)
#     img = cv2.resize(img, None, None, 0.5, 0.5)

# 数据中心化
def centere_data(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    meanVal = np.tile(meanVal, (rows, 1))
    newdata = dataMat - meanVal
    return newdata, meanVal

'''
# 协方差矩阵
def Cov(dataMat):
    meanVal = np.mean(dataMat, 0)  # 压缩行，返回1*cols矩阵，对各列求均值
    meanVal = np.tile(meanVal, (rows, 1))  # 返回rows行的均值矩阵
    Z = dataMat - meanVal
    Zcov = (1 / (rows - 1)) * Z.T * Z
    return Zcov
'''

# 最小化降维造成的损失，确定k
def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    temp_Sum = 0
    num = 0
    thresh=arraySum * percentage
    for i in sortArray:
        temp_Sum += i
        num += 1
        if temp_Sum >= thresh:
            return num


# 得到最大的k个特征值和特征向量
def EigDV(covMat, p):
    D, V = np.linalg.eig(covMat)  # 得到特征值和特征向量
    k = Percentage2n(D, p)  # 确定k值
    print("保留{}%信息，降维后的特征个数：".format(p*100) + str(k) + "\n")
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k + 1):-1]
    K_eigenVector = V[:, K_eigenValue]
    return K_eigenValue, K_eigenVector

# PCA算法
def PCA1(data, p):
    dataMat = np.float32(np.mat(data))
    # 数据中心化
    dataMat, meanVal = centere_data(dataMat)
    # 计算协方差矩阵
    covMat = np.cov(dataMat, rowvar=False)
    # 选取最大的k个特征值和特征向量
    D, V = EigDV(covMat, p)
    # 得到降维后的数据
    lowDataMat = dataMat * V
    # 重构数据
    reconDataMat = lowDataMat *V.T+ meanVal
    return reconDataMat

def PCAImg(data, p):
    dataf = np.float32(data)
    pca=PCA()
    pca.fit(dataf)
    # VV=pca.explained_variance_#特征值
    VR=pca.explained_variance_ratio_#特征值占比

    VN = 0
    temp_Sum=0
    #thresh = p*VV.sum()
    for i in VR:
        temp_Sum += i
        VN += 1
        if temp_Sum >= p:#thresh
            break
    VF = pca.components_[:VN,:]

    lowD=np.dot(dataf, VF.T)
    reconD=np.dot(lowD, VF) #+ pca.mean_
    return reconD

def PCAImgs(data, p):
    N,H,W=data.shape
    data=data.reshape((N,-1))
    dataf = np.float32(data)
    pca=PCA()#all features
    pca.fit(dataf)
    # VV=pca.explained_variance_#特征值
    VR=pca.explained_variance_ratio_#特征值占比
    #covMat = np.cov(dataf, rowvar=True)
    VN = 0
    temp_Sum=0
    #thresh = p*VV.sum()
    if p==1.0:
        VN=N
    else:
        for i in VR:
            temp_Sum += i
            VN += 1
            if temp_Sum >= p:#thresh
                break
    VF = pca.components_[:VN,:]
    # for i in range(VN):
    #     FI = VF[i, :].reshape((H,W))
    #     FI=cv2.normalize(FI,None,0,255,cv2.NORM_MINMAX)
    #     cv2.imshow('a'+str(i), FI.astype(np.uint8))
    # cv2.waitKey(0)
    lowD=np.dot(dataf, VF.T)
    reconD=np.dot(lowD, VF) #+ pca.mean_
    return reconD

def pcaImg():
    path = 'E:/fjj/SeaShips_SMD/JPEGImages/'
    files = ['004091.jpg', '004100.jpg', 'MVI_1592_VIS_00224.jpg', '003714.jpg', '000498.jpg']
    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(path, file), 0)
        img = cv2.resize(img, None, None, 0.5, 0.5)
        rows, cols = img.shape
        # pca = decomposition.PCA()
        print("降维前的特征个数：" + str(cols) + "\n")
        # print(img)
        print('----------------------------------------')
        PCA_img = PCA1(img, 0.90)
        # PCA_img = PCAImg(img, 0.98)
        PCA_img = PCA_img.astype(np.uint8)
        cv2.imshow('test' + str(i), PCA_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pcaImgs():
    path = 'E:/fjj/SeaShips_SMD/JPEGImages/'
    files = ['004091.jpg', '004100.jpg', 'MVI_1592_VIS_00224.jpg', '003714.jpg', '000498.jpg']
    imgs=[]
    N=len(files)
    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(path, file), 0)
        img = cv2.resize(img, None, None, 0.5, 0.5)
        imgs.append(img)
        # rows, cols = img.shape
    imgs=np.array(imgs)

    print('----------------------------------------')
    #imgs=imgs.reshape((N,-1))
    PCA_imgs=PCAImgs(imgs, 1.0)
    # PCA_img = PCA1(img, 0.98)
    #PCA_img = PCAF(img, 1.0)
    PCA_imgs = PCA_imgs.astype(np.uint8)
    # print(PCA_img)
    # cv2.imshow('test' + str(i), PCA_img)
    # cv2.imwrite('D:/testimage/dog-pca.jpg',PCA_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pcaImg()
    # pcaImgs()
