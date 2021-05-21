import numpy as np
import os
from shutil import copy
import scipy.io as scio
import cv2

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

def detection2xml(imgname,width,height,gt,xmlpath,difficult=0):
    '''
    imgname:basename for xml
    width: width of image
    height: height of image
    gt: <（bbox，class_name）>  bbox:[xmin,ymin,xmax,ymax]
    xmlpath:path of xml
    difficult


    '''
    # write in xml file
    #path=(xmlpath + '/' + imgname + '.xml')
    path=os.path.join(xmlpath,imgname+'.xml')
    with open(path, 'w') as xml_file:
        #xml_file.write('<?xml version=\'1.0\' encoding=\'utf-8\'?>')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC</folder>\n')
        xml_file.write('    <filename>' + imgname + '.jpg' + '</filename>\n')
        xml_file.write('    <path>' + path + '</path>\n')
        xml_file.write('    <source>\n')
        xml_file.write('        <database>' + 'Singapore Maritime Dataset' + '</database>\n')
        xml_file.write('    </source>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of image on xml file
        for img_each_label in gt:#
            spt = img_each_label#img_each_label.split(' ') #这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + spt[1] + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>'+str(int(difficult))+'</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(int(spt[0][0])) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(int(spt[0][1])) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(int(spt[0][2])) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(int(spt[0][3])) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
        xml_file.write('</annotation>')


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

# for i in range(len(GTs)):
#     print('Processing %d:'%i,Videos[i])
#     # cap=cv2.VideoCapture(Videos[i])
#     basename=os.path.splitext(os.path.basename(Videos[i]))[0]
#     Annotations = scio.loadmat(GTs[i])['structXML'][0]
#     #('Motion', 'O'), ('Object', 'O'), ('Distance', 'O'), ('MotionType', 'O'), ('ObjectType', 'O'), ('DistanceType', 'O'), ('BB', 'O')
#     # classes=GT['structXML'][1][4]#class
#     # BBs=GT['structXML'][1][6]#BBs
#     # if cap.isOpened():
#     #     nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     #     fps = cap.get(cv2.CAP_PROP_FPS)
#     #     duration = nFrames / fps  # s
#     #     step=nFrames/250
#         #for i in range(nFrames):
#         pos=0
#         while True:
#             int_pos=int(pos)
#             cap.set(cv2.CAP_PROP_POS_FRAMES,int_pos)
#             ret, frame = cap.read()  # 捕获一帧图像
#
#             frame_num=int_pos+1
#             if ret:
#                 width, height = frame.shape[1], frame.shape[0]
#                 img_temp=frame.copy()
#                 #cv2.imshow('frame', frame)
#                 #cv2.waitKey(25)
#                 img_name=basename+'_%05d.jpg'%frame_num
#                 distance=Annotations[int_pos][5]
#                 classes=Annotations[int_pos][4]
#                 bboxes=Annotations[int_pos][6]
#                 object_num=len(bboxes)
#                 Total_dets=[]
#                 bb_n=0
#                 try:
#                     for n in range(object_num):
#                         if classes.shape[1]>0 and len(classes[n,0])>0:
#                             name=classes[n,0][0]
#                         else:
#                             bb_n+=1
#                             continue
#                             #name='Unknown'
#                         bbox=bboxes[n-bb_n]
#                         bbox[2]=bbox[0]+bbox[2]
#                         bbox[3]=bbox[1]+bbox[3]
#                         #is_diffcult=
#                         Total_dets.append((bbox,name))
#                         cv2.rectangle(img_temp,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,255),5)
#                         cv2.putText(img_temp, '%s' % (name), (int(bbox[0]), int(bbox[1] - 2)),
#                                          cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 8, 8, False)
#                 finally:
#                     print('int_pos:',int_pos,'\nn:',n)
#                 # cv2.imshow('a',img_temp)
#                 # cv2.waitKey(1)
#                 detection2xml(os.path.splitext(img_name)[0], width, height, Total_dets, annotaPath)
#                 # cv2.imwrite(os.path.join(save_path,img_name),frame)
#             else:
#                 break
#             print('pos:',pos)
#             pos+=step
#     # cap.release()  # 关闭相机
#     print('\n')

