import numpy as np
import os
from shutil import copy
import scipy.io as scio
import cv2
from datasetConvert.detection2voc import detection2xml
absroot='E:/fyl/WIDER_val/'
imagePath='images'
annotaPath='ground_truth'
JPEGImages='JPEGImages'
Annotations='Annotations'


CLASSES=('__background__',
         'Face'
         )
# DistanceType=('Far','Near','Other')
# save_path='JPEGImages1'
# anno_path='Annotations'


class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))
# distancetype_to_ind = dict(list(zip(DistanceType, list(range(len(DistanceType))))))
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# if not os.path.exists(annotaPath):
#     os.mkdir(annotaPath)

# Video_path='VIS_Onshore/Videos/MVI_1448_VIS_Haze.avi'
# GT_path='VIS_Onshore/ObjectGT/'#'VIS_Onshore/Videos/*.avi'#VIS_Onboard
# Video_Path='VIS_Onshore/Videos'#VIS_Onboard
#Videos=glob.glob(Video_path)
# GTs=os.listdir(GT_path)
# Videos=[os.path.join(Video_Path,gt.replace('_ObjectGT.mat','.avi')) for gt in GTs]
# GTs=[os.path.join(GT_path,gt) for gt in GTs]
saveXml=os.path.join(absroot,Annotations)
data=scio.loadmat(os.path.join(absroot,annotaPath,'wider_easy_val.mat'))
print(data.keys())
file_lists=data['file_list']
face_bbx_lists=data['face_bbx_list']
invalid_label_lists=data['invalid_label_list']
occlusion_label_lists=data['occlusion_label_list']
i=0
# file_lists[i][0]
sum=0
total_bb=0
for i in range(len(file_lists)):#遍历文件夹
    print('Processing %d:' % i)
    files=file_lists[i][0]
    bboxes_files=face_bbx_lists[i][0]
    #difficults=occlusion_label_lists[i][0]
    sum+=len(files)
    for j in range(len(files)):#遍历文件
        file=files[j,0][0]
        print(file)
        bboxes=bboxes_files[j,0].copy().astype(np.int)
        bboxes[:,2]=bboxes[:,0]+bboxes[:,2]
        bboxes[:,3] = bboxes[:, 1] + bboxes[:, 3]
        img=cv2.imread(os.path.join(absroot,JPEGImages,file+'.jpg'))
        for bb in bboxes:
            img=cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(0,0,255))
        #cv2.imshow('a',img)
        #cv2.waitKey()
        total_bb+=len(bboxes)
        gt=[(bb,CLASSES[1]) for bb in bboxes]
        detection2xml(file, 0, 0, gt, saveXml)
    #sum+=len(file_lists[i][0])
print('total files %d'%sum)
print('total faces %d'%total_bb)

