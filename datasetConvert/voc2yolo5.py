import xml.etree.ElementTree as ET
from os import getcwd
import os

# sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
# sets=[('seaships_smd', 'label0'), ('seaships_smd', 'test1300'),('seaships_smd', 'label3')]
sets=[('seaships_smd', 'label2')]
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["passenger ship",
"ore carrier",
"general cargo ship",
"fishing boat",
"Sail boat",
"Kayak",
"flying bird",
"vessel",
"Buoy",
"Ferry",
"container ship",
"Other",
"Boat",
"Speed boat",
"bulk cargo carrier"]
########################New
#classes=['boat']
sets=[('SMD_SS','train')]

datasetpath='F:/船舶数据整理/SeaShips_SMD'#'E:/fjj/MarineShips2' #'E:/fjj/SeaShips_SMD'

def convert_annotation(datasetname,image_id, list_file,encoding='utf-8'):
    in_file = open(os.path.join(datasetpath,'Annotations/%s.xml'%(image_id)),'r',encoding=encoding)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size=root.find('size')
    width=float(size.find('width').text)
    height=float(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is None:
            difficult='0'
        else:
            difficult=difficult.text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        xmin=int(xmlbox.find('xmin').text)
        ymin=int(xmlbox.find('ymin').text)
        xmax=int(xmlbox.find('xmax').text)
        ymax=int(xmlbox.find('ymax').text)
        b = ((xmin+xmax)/2.0/width, (ymin+ymax)/2.0/height,
             (xmax-xmin)/width, (ymax-ymin)/height)
        list_file.write(str(cls_id)+' ' + " ".join([str(a) for a in b])+'\n')

wd = getcwd()
# save_path='dataset'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
encoding='utf-8'
for datasetname, image_set in sets:
    SaveDir='%s_%s_YOLOv5'%(datasetname,image_set)
    if not os.path.exists(SaveDir):
        os.mkdir(SaveDir)
    f=open(os.path.join(datasetpath, 'ImageSets/Main/%s.txt' % (image_set)),'r',encoding=encoding)
    image_ids = f.readlines()
    f.close()
    image_ids=[im_id.strip() for im_id in image_ids]
    #path=os.path.join(datasetpath,'JPEGImages/%s.jpg')
    for image_id in image_ids:
        print(image_id)
        # with open('s')list_file.write(path%(image_id))
        with open(os.path.join(SaveDir,'%s.txt'%(image_id)), 'w', encoding=encoding) as label_file:
            convert_annotation(datasetname, image_id, label_file,encoding=encoding)
            #list_file.write('\n')
    #list_file.close()

