import cv2
import scipy.io as scio
import os
import glob
import math
import numpy as np

CLASSES=('__background__',
         'Ferry','Buoy','Vessel/ship','Speed boat','Boat',
         'Kayak','Sail boat','Swimming person','Flying bird/plane','Other'
         )
DistanceType=('Far','Near','Other')
save_path='JPEGImages1'
anno_path='Annotations'

def detection2xml(imgname,width,height,gt,xmlpath):
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
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(int(spt[0][0])) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(int(spt[0][1])) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(int(spt[0][2])) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(int(spt[0][3])) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
        xml_file.write('</annotation>')


class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))
distancetype_to_ind = dict(list(zip(DistanceType, list(range(len(DistanceType))))))
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(anno_path):
    os.mkdir(anno_path)

# Video_path='VIS_Onshore/Videos/MVI_1448_VIS_Haze.avi'
GT_path='VIS_Onshore/ObjectGT/'#'VIS_Onshore/Videos/*.avi'#VIS_Onboard
Video_Path='VIS_Onshore/Videos'#VIS_Onboard
#Videos=glob.glob(Video_path)
GTs=os.listdir(GT_path)
Videos=[os.path.join(Video_Path,gt.replace('_ObjectGT.mat','.avi')) for gt in GTs]
GTs=[os.path.join(GT_path,gt) for gt in GTs]

assert len(GTs)==len(Videos)
for i in range(len(GTs)):
    print('Processing %d:'%i,Videos[i])
    cap=cv2.VideoCapture(Videos[i])
    basename=os.path.splitext(os.path.basename(Videos[i]))[0]
    Annotations = scio.loadmat(GTs[i])['structXML'][0]
    #('Motion', 'O'), ('Object', 'O'), ('Distance', 'O'), ('MotionType', 'O'), ('ObjectType', 'O'), ('DistanceType', 'O'), ('BB', 'O')
    # classes=GT['structXML'][1][4]#class
    # BBs=GT['structXML'][1][6]#BBs
    if cap.isOpened():
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = nFrames / fps  # s
        step=nFrames/250
        #for i in range(nFrames):
        pos=0
        while True:
            int_pos=int(pos)
            cap.set(cv2.CAP_PROP_POS_FRAMES,int_pos)
            ret, frame = cap.read()  # 捕获一帧图像

            frame_num=int_pos+1
            if ret:
                width, height = frame.shape[1], frame.shape[0]
                img_temp=frame.copy()
                #cv2.imshow('frame', frame)
                #cv2.waitKey(25)
                img_name=basename+'_%05d.jpg'%frame_num
                distance=Annotations[int_pos][5]
                classes=Annotations[int_pos][4]
                bboxes=Annotations[int_pos][6]
                object_num=len(bboxes)
                Total_dets=[]
                bb_n=0
                try:
                    for n in range(object_num):
                        if classes.shape[1]>0 and len(classes[n,0])>0:
                            name=classes[n,0][0]
                        else:
                            bb_n+=1
                            continue
                            #name='Unknown'
                        bbox=bboxes[n-bb_n]
                        bbox[2]=bbox[0]+bbox[2]
                        bbox[3]=bbox[1]+bbox[3]
                        #is_diffcult=
                        Total_dets.append((bbox,name))
                        cv2.rectangle(img_temp,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,255),5)
                        cv2.putText(img_temp, '%s' % (name), (int(bbox[0]), int(bbox[1] - 2)),
                                         cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 8, 8, False)
                finally:
                    print('int_pos:',int_pos,'\nn:',n)
                # cv2.imshow('a',img_temp)
                # cv2.waitKey(1)
                detection2xml(os.path.splitext(img_name)[0], width, height, Total_dets, anno_path)
                cv2.imwrite(os.path.join(save_path,img_name),frame)
            else:
                break
            print('pos:',pos)
            pos+=step
    cap.release()  # 关闭相机
    print('\n')

