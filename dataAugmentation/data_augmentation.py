import cv2
import os
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

root='E:/fjj/SeaShips_SMD/JPEGImages'
imgname='MVI_1582_VIS_00253.jpg'#'MVI_1644_VIS_00203.jpg'
imagepath=os.path.join(root,imgname)

img=Image.open(imagepath)
imgnp=np.array(img)
plt.figure(1)
plt.imshow(imgnp)
#plt.show(1)
transform=transforms.Compose(
                    [
                        #transforms

                        #transforms.RandomCrop((1080,1920), padding=4),
                        # transforms.RandomCrop((1080,1080), padding=4),
                        transforms.RandomResizedCrop((1080,1920), (0.8, 1.0),(3.0/4,1920.0/1080.0)),

                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply(
                            [transforms.ColorJitter(0.3, 0.3, 0.15, 0.1)], p=0.5
                        ),
                        transforms.RandomGrayscale(p=0.2),
                        #transforms.Resize((1080,1920))
                        #transforms.ToTensor(),
                        #transforms.Normalize(mean, std),
                    ]
                )
# transform=transforms.Compose(
#                     [
#                         #transforms
#                         transforms.RandomChoice(
#                             [
#                                 transforms.RandomCrop((1080,1920), padding=4),
#                                 transforms.RandomResizedCrop((1080,1920), (0.8, 1.0)),
#                             ]
#                         ),
#                         transforms.RandomHorizontalFlip(),
#                         transforms.RandomApply(
#                             [transforms.ColorJitter(0.3, 0.3, 0.15, 0.1)], p=0.5
#                         ),
#                         transforms.RandomGrayscale(p=0.15),
#                         #transforms.ToTensor(),
#                         #transforms.Normalize(mean, std),
#                     ]
#                 )

savetmp='../data_processing_cache/dataAug'

if not os.path.exists(savetmp):
    os.mkdir(savetmp)


a=transform(img)
anp=np.array(a)
plt.figure(2)
plt.imshow(anp)
a.save(os.path.join(savetmp,'a'+imgname))
a=transform(img)
anp=np.array(a)
plt.figure(3)
plt.imshow(anp)
a.save(os.path.join(savetmp,'b'+imgname))