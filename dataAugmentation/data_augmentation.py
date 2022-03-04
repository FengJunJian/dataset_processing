import cv2
import os
#import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
def augTorch():
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
                            #transforms.RandomResizedCrop((1080,1920), (0.8, 1.0),(3.0/4,1920.0/1080.0)),

                            #transforms.RandomHorizontalFlip(),
                            # transforms.RandomApply(
                            #     [transforms.ColorJitter(0.3, 0.3, 0.15, 0.1)], p=0.5
                            # ),
                            transforms.RandomErasing(0.5),
                            #transforms.RandomGrayscale(p=0.2),
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

    savetmp='../../data_processing_cache/dataAug'

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

def augImgaug():
    root = 'E:/fjj/SeaShips_SMD/JPEGImages'
    imgname = 'MVI_1582_VIS_00253.jpg'  # 'MVI_1644_VIS_00203.jpg'
    imagepath = os.path.join(root, imgname)

    #img = Image.open(imagepath)
    img=cv2.imread(imagepath)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    # imgnp = np.array(img)
    plt.figure(1)
    plt.imshow(img)
    plt.grid()
    frame = plt.gca()
    # y 轴不可见
    #frame.axes.get_yaxis().set_visible(False)
    #plt.axis('off')


    seq=iaa.Sequential([
        #iaa.Cutout(5,size=0.02),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.07*255), per_channel=0.6),
        iaa.OneOf([
           #iaa.Dropout((0.01, 0.1), per_channel=0.6),
           iaa.CoarseDropout((0.03, 0.031), size_percent=(0.02, 0.05),per_channel=0.3),]),
        iaa.Affine(
            scale={"x": (1.0, 1.0), "y": (1.0, 1.0)},
            translate_percent={"x": (-15.0/1920.0, 15.0/1920.0), "y": (-15.0/1080.0, 15.0/1080.0)},#(1080,1920)
            rotate=(-3, 3),
            ),
                        ])

    imgs=seq(images=[img])
    #imgA = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR)
    plt.figure(2)
    plt.imshow(imgs[0])
    plt.grid()
   # plt.axis('off')



    cv2.imwrite('o'+imgname,img)
    cv2.imwrite('a'+imgname,imgs[0])
    plt.show()

def postprocess():
    root = '../../data_processing_cache\dataAug'
    files=['aMVI_1582_VIS_00253.jpg','aMVI_1644_VIS_00203.jpg','MVI_1582_VIS_00253.jpg','MVI_1644_VIS_00203.jpg']
    for file in files:
        filedir=os.path.join(root,file)
        img=cv2.imread(filedir)
        dst=cv2.resize(img,dsize=None,fx=0.2,fy=0.2)
        cv2.imshow('a',dst)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(root,"resize"+file),dst)
if __name__ == "__main__":
    augImgaug()
    #augTorch()
    # postprocess()