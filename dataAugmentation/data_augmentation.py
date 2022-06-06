import cv2
import os
#import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A

from imgaug import augmenters as iaa
import random
def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area
def bbox2segment(bbox):#xmin,ymin,xmax,ymax
    #n=len(bboxes)
    #segments=[]
    #for bbox in bboxes:
    segment = []
    segment.append([[bbox[0],bbox[1]]])
    segment.append([[bbox[0],bbox[3]]])
    segment.append([[bbox[2],bbox[3]]])
    segment.append([[bbox[2], bbox[1]]])
    #segments.append(np.array(segment))
    return np.array(segment)
def bboxes2segment(bboxes):#xmin,ymin,xmax,ymax
    #n=len(bboxes)
    segments=[]
    for bbox in bboxes:
        segments.append(bbox2segment(bbox))

    return segments
def copy_paste(im, labels,p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    #n = len(segments)
    n=len(labels)
    segments=bboxes2segment(labels[:,1:])
    #segment = bbox2segment(labels[0, 1:])
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            #l, s = labels[j], segments[j]
            l = labels[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.20).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                # segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                segments.append(bbox2segment(box))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
            #cv2.imshow('a',im_new)
        #cv2.circle()
        # imgray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug
    return im, labels
    #return im, labels, segments
def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels

def augTorch():
    from torchvision import transforms
    root='E:/SeaShips_SMD/JPEGImages'
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
    root = 'E:/SeaShips_SMD/JPEGImages'
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

def augImgalbu():
    root = 'E:/SeaShips_SMD/JPEGImages'
    #imgname = 'MVI_1582_VIS_00253.jpg'  # 'MVI_1644_VIS_00203.jpg'
    imgname="000001.jpg"
    imgname1 = "000002.jpg"
    imagepath = os.path.join(root, imgname)
    # img = Image.open(imagepath)
    img = cv2.imread(imagepath)
    img2=cv2.imread(os.path.join(root,imgname1))
    H,W,C=img.shape
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # imgnp = np.array(img)
    plt.figure(3)
    plt.imshow(img[:,:,::-1])
    #plt.grid()

    labels=np.array([[0,633,467,944,510],[1,0,0,10,5]])#(label,xmin,ymin,xmax,ymax)
    labels1=np.array([[0,894,474,1252,525]])
    #labels=np.array([0])
    frame = plt.gca()
    # y 轴不可见
    # frame.axes.get_yaxis().set_visible(False)
    # plt.axis('off')

    seq = A.Compose([
        #A.Resize(int(H/2),int(W/2)),
        A.ShiftScaleRotate(shift_limit=0,rotate_limit=0,scale_limit=0.6,border_mode=cv2.BORDER_CONSTANT  ,p=0.8),
        #A.Downscale(always_apply=True)#下采样
        #A.Cutout(10,p=1.0),
        #A.GaussianBlur(blur_limit=(3,9),p=1.0)
        #A.Blur(blur_limit=10,p=1.0),
        #A.MedianBlur(p=1.0)
        # A.ChannelDropout(p=1.0)
        #A.ShiftScaleRotate(shift_limit=0, rotate_limit=0, scale_limit=0.6, border_mode=cv2.BORDER_CONSTANT,p=1.0)
        A.Cutout(num_holes=16,max_h_size=16,max_w_size=16,p=0.8)
        #A.RandomFog(fog_coef_upper=0.5,p=1.0),#雾True霾
        #A.RandomRain(p=1.0)#下雨
        #A.RandomShadow(p=1.0)#阴影
        # A.RandomScale(p=1.0)
        #A.RandomSunFlare(flare_roi=(0,0,1,0.3),num_flare_circles_lower=1,num_flare_circles_upper=5,src_radius=300,p=1.0)

    ],bbox_params=A.BboxParams("pascal_voc",),)#seq.processors['bboxes'].params._to_dict()

    new={}
    new['image']=None
    new['bboxes']=None
    albuformat="pascal_voc"
    original_bbox_p=seq.processors['bboxes'].params._to_dict()
    original_bbox_p.update({"format":albuformat,"label_fields":['class_labels']})
    seq.processors["bboxes"] = A.BboxProcessor(A.BboxParams(**original_bbox_p))
    new.update(seq(image=img, bboxes=labels[:,1:], class_labels=labels[:,0]))
    #cv2.imshow('a',new['image'])
    # im,la=copy_paste(img,labels,p=1.0)
    # im3, labels3=mixup(img,labels,img2,labels1)
    # new['image']=im
    # new['bboxes']=la
    # imgA = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR)
    plt.figure(2)
    #cv2.imwrite('R'+imgname,new['image'])
    plt.imshow(new['image'][:,:,::-1])
    print(new['image'].shape)
    print(new['bboxes'])
    #plt.grid()
    # plt.axis('off')
    # cv2.imwrite('o' + imgname, img)
    # cv2.imwrite('a' + imgname, imgs[0])
    plt.show()
if __name__ == "__main__":
    augImgalbu()
    # augImgaug()
    #augTorch()
    # postprocess()