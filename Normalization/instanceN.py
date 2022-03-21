import os
import numpy as np
import cv2

def IN(img):
    assert len(img.shape)>2
    meanO=img.mean()
    stdO=img.std()
    H,W,C=img.shape
    imgf = img.astype(np.float32)
    for c in range(C):
        imgT=imgf[:,:,c]
        mean=imgT.mean()
        std=imgT.std()
        imgf[:,:,c]=(imgT-mean)/std
    img1=((imgf*stdO)+meanO).astype(np.uint8)
    return img1,imgf

def demoIN():
    data=np.random.random((2,5))
    mean=np.mean(data,axis=1)
    std=np.std(data,axis=1)
    data1=(data-mean[:,np.newaxis])/std[:,np.newaxis]
    print('before:',data,'after',data1)

if __name__=='__main__':
    path = 'E:/fjj/SeaShips_SMD/JPEGImages/'
    files = ['004091.jpg', '004100.jpg']
    for file in files:
        img = cv2.imread(os.path.join(path, file))
        img = cv2.resize(img, None, None, 0.5, 0.5)
        cv2.imshow('0', img)
        cv2.waitKey(1)
        img1,imgm=IN(img)
        cv2.imshow('1',img1)
        imgT=((imgm*10)+128).astype(np.uint8)
        cv2.imshow('r',imgT)
        cv2.waitKey()
