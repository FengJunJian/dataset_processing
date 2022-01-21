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
sets=[('MarineShip','all')]

datasetpath='E:/fjj/MarineShips2' #'E:/fjj/SeaShips_SMD'

def convert_annotation(datasetname,image_id, list_file,encoding='utf-8'):
    in_file = open(os.path.join(datasetpath,'Annotations/%s.xml'%(image_id)),'r',encoding=encoding)
    tree=ET.parse(in_file)
    root = tree.getroot()

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
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()
# save_path='dataset'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
encoding='utf-8'
for datasetname, image_set in sets:
    with open(os.path.join(datasetpath, 'ImageSets/Main/%s.txt' % (image_set)),'r',encoding=encoding) as f:
        image_ids = f.readlines()
        image_ids=[im_id.strip() for im_id in image_ids]
        list_file = open('%s_%s.txt'%(datasetname, image_set), 'w',encoding=encoding)
        path=os.path.join(datasetpath,'JPEGImages/%s.jpg')
        for image_id in image_ids:
            print(image_id)
            list_file.write(path%(image_id))
            convert_annotation(datasetname, image_id, list_file,'utf-8')
            list_file.write('\n')
        list_file.close()

