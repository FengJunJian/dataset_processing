import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

predefined_size=[[140,140],[256,128],[512,128]]#w,h
anchora=[(534,662),(1374,557)]#x,y
anchorb=[(1424,662),(716,557)]#x,y
root='../../data_processing_cache/dataAug'
imgnamea='aMVI_1582_VIS_00253.jpg'
imgnameb='bMVI_1582_VIS_00253.jpg'


imagepatha=os.path.join(root,imgnamea)
imga=plt.imread(imagepatha)
plt.figure(1)
plt.imshow(imga)

for i in range(len(anchora)):
    a=anchora[i]
    for j in range(len(predefined_size)):
        size=np.array(predefined_size[j])
        roi=imga[int(a[1]-size[1]/2):int(a[1]+size[1]/2),int(a[0]-size[0]/2):int(a[0]+size[0]/2)]
        plt.imsave(os.path.join(root, str(i) + str(j) + imgnamea), roi)
        # plt.imshow(roi)



imagepathb=os.path.join(root,imgnameb)
imgb=plt.imread(imagepathb)
for i in range(len(anchorb)):
    a = anchorb[i]
    for j in range(len(predefined_size)):
        size = np.array(predefined_size[j])
        roi = imgb[int(a[1] - size[1] / 2):int(a[1] + size[1] / 2), int(a[0] - size[0] / 2):int(a[0] + size[0] / 2)]
        plt.imsave(os.path.join(root,str(i)+str(j)+imgnameb),roi)

        # plt.imshow(roi)
plt.figure(2)
plt.imshow(imgb)

# img=cv2.imread(imagepath)
