import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import imgaug
import imgaug.augmenters as iaa

def functional_conv2d(im):
    aug = iaa.EdgeDetect(alpha=(1.0))
    edge_detect=aug(images=im)
    # sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
    # sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # weight = Variable(torch.from_numpy(sobel_kernel))
    # edge_detect = F.conv2d(Variable(im), weight)
    # edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

path="E:/SeaShips_SMD/JPEGImages/000001.jpg"
iaa.Sequential([]).augment_image
#im = Image.open(path)#.convert('L')
im=cv2.imread(path,)#cv2.IMREAD_GRAYSCALE
im=im.astype(np.float32)

dst=cv2.normalize(im,None,0.0,1.0,cv2.NORM_MINMAX)
aug = iaa.EdgeDetect(alpha=(1.0))

aug1 = iaa.DirectedEdgeDetect(alpha=(1.0), direction=(1.0))
a=aug(image=dst)
a2=aug(image=im)
cv2.imshow('a',a)
cv2.imshow('a2',a2)
cv2.waitKey()

# cv2.imshow('a0',a[:,:,0])
# cv2.imshow('a1',a[:,:,1])
# cv2.imshow('a2',a[:,:,2])
a1=aug1(images=im)

cv2.imshow('a1',a1)

sobelx=cv2.Sobel(im,cv2.CV_64F,1,0)
sobelx = cv2.convertScaleAbs(sobelx)
sobely=cv2.Sobel(im,cv2.CV_64F,0,1)
sobely = cv2.convertScaleAbs(sobely)
sobelimg=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
cv2.imshow('sobel',sobelimg)


cv2.waitKey()

# 将图片数据转换为矩阵
#im = np.array(im, dtype='float32')
# 将图片矩阵转换为pytorch tensor,并适配卷积输入的要求
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
im=im.float()
# 边缘检测操作
# edge_detect = nn_conv2d(im)
edge_detect = functional_conv2d(im)
# 将array数据转换为image
im = Image.fromarray(edge_detect)
# image数据转换为灰度模式
im = im.convert('L')








o=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
r1=cv2.Canny(o,128,200)
r2=cv2.Canny(o,32,128)



cv2.imshow("original",o)



cv2.imshow("result1",r1)
cv2.imshow("result2",r2)
cv2.waitKey()
cv2.destroyAllWindows()
