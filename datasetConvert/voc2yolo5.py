import xml.etree.ElementTree as ET
from os import getcwd
import os
import shutil
from tqdm import tqdm
#import cv2
from PIL import Image
# sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
# sets=[('seaships_smd', 'label0'), ('seaships_smd', 'test1300'),('seaships_smd', 'label3')]
#sets=[('seaships_smd', 'label2')]
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes = ["passenger ship",#0
    "ore carrier",#1
    "general cargo ship",#2
    "fishing boat",#3
    "Sail boat",#4
    "Kayak",#5
    "flying bird",#6
    "vessel",#7
    "Buoy",#8
    "Ferry",#9
    "container ship",#10
    "Other",#11
    "Boat",#12
    "Speed boat",#13
    "bulk cargo carrier"#14
           ]
classes4 = [
    "boat",
    "Buoy",
    "flying bird",
    "Other",
]
CLASSES=classes

def convert_annotation(datasetpath,image_id,  list_file,encoding='utf-8'):
    in_file = open(os.path.join(datasetpath,'Annotations/%s.xml'%(image_id)),'r',encoding=encoding)
    tree=ET.parse(in_file)
    root = tree.getroot()
    #img=cv2.imread(os.path.join(datasetpath,'JPEGImages/%s.jpg'%(image_id)))
    # img=Image.open('G:/ShipDataSet/BXShipDataset/JPEGImages/622aee26a856644c87eadef9.jpg')
    img = Image.open(os.path.join(datasetpath,'JPEGImages/%s.jpg'%(image_id)))
    width,height=img.size
    # size=root.find('size')
    # width=float(size.find('width').text)
    # height=float(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is None:
            difficult='0'
        else:
            difficult=difficult.text
        cls = obj.find('name').text
        if int(difficult)==1:#or cls not in classes
            continue
        try:
            cls_id = CLASSES.index(cls)
        except:
            print('error class index!###########################################################')
            cls_id=0
        try:
            xmlbox = obj.find('bndbox')
            xmin=int(xmlbox.find('xmin').text)
            ymin=int(xmlbox.find('ymin').text)
            xmax=int(xmlbox.find('xmax').text)
            ymax=int(xmlbox.find('ymax').text)
            b = ((xmin+xmax)/2.0/width, (ymin+ymax)/2.0/height,
                 (xmax-xmin)/width, (ymax-ymin)/height)
        except ZeroDivisionError:
            print(image_id)

        list_file.write(str(cls_id)+' ' + " ".join([str(a) for a in b])+'\n')

def shipClass():
    targetPath='E:/SeaShips_SMD'
    sets=[[('SMD_SS','train'),('SMD_SS','test')],[('BXShip','allShip'),('BXShip','testShip')]]#('SMD_SS','all')
    # sets = [
    #         [('BXShip', 'allShip'), ('BXShip', 'testShip')]]  # ('SMD_SS','all')
    datasetpaths=('E:/SeaShips_SMD','G:\ShipDataSet\BXShipDataset')#'E:/fjj/MarineShips2' #'E:/fjj/SeaShips_SMD'
    # datasetpaths = ( 'G:/ShipDataSet/BXShipDataset',)  # 'E:/fjj/MarineShips2' #'E:/fjj/SeaShips_SMD'
    wd = getcwd()
    # save_path='dataset'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    encoding='utf-8'
    for datasetpath,set in zip(datasetpaths,sets):
        for datasetname, image_set in set:
            SaveDir=os.path.join(targetPath,'dataset/%s_YOLOv5'%(image_set))
            LabelDir=os.path.join(SaveDir,'labels')
            ImgDir=os.path.join(SaveDir,'images')
            if not os.path.exists(LabelDir):
                os.makedirs(LabelDir)
            if not os.path.exists(ImgDir):
                os.makedirs(ImgDir)
            f=open(os.path.join(datasetpath, 'ImageSets/Main/%s.txt' % (image_set)),'r',encoding=encoding)
            image_ids = f.readlines()
            f.close()
            image_ids=[im_id.strip() for im_id in image_ids]
            #path=os.path.join(datasetpath,'JPEGImages/%s.jpg')
            for image_id in tqdm(image_ids):
                #print(image_id)
                # with open('s')list_file.write(path%(image_id))
                #try:
                shutil.copy(os.path.join(datasetpath,'JPEGImages/%s.jpg'%image_id),ImgDir)
                with open(os.path.join(LabelDir,'%s.txt'%(image_id)), 'w', encoding=encoding) as label_file:
                    convert_annotation(datasetpath,image_id, label_file,encoding=encoding)
                # except:
                #     print(image_id)
                #list_file.write('\n')

def ship4class():
    ########################New
    #classes=['boat']
    sets=[('SMD_SS','train'),('SMD_SS','test'),]#('SMD_SS','all')
    datasetpath='E:/fjj/SeaShips_SMD'#'E:/fjj/MarineShips2' #'E:/fjj/SeaShips_SMD'
    #wd = getcwd()
    # save_path='dataset'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    encoding='utf-8'
    for datasetname, image_set in sets:
        SaveDir=os.path.join(datasetpath,'dataset/%s_YOLOv5'%(image_set))
        LabelDir=os.path.join(SaveDir,'labels')
        ImgDir=os.path.join(SaveDir,'images')
        if not os.path.exists(LabelDir):
            os.makedirs(LabelDir)
        if not os.path.exists(ImgDir):
            os.makedirs(ImgDir)
        f=open(os.path.join(datasetpath, 'ImageSets/Main/%s.txt' % (image_set)),'r',encoding=encoding)
        image_ids = f.readlines()
        f.close()
        image_ids=[im_id.strip() for im_id in image_ids]
        #path=os.path.join(datasetpath,'JPEGImages/%s.jpg')
        for image_id in image_ids:
            print(image_id)
            # with open('s')list_file.write(path%(image_id))
            shutil.copy(os.path.join(datasetpath,'JPEGImages/%s.jpg'%image_id),ImgDir)
            with open(os.path.join(LabelDir,'%s.txt'%(image_id)), 'w', encoding=encoding) as label_file:
                convert_annotation(datasetpath,image_id, label_file,encoding=encoding)
                #list_file.write('\n')

if __name__=="__main__":
    # ship4class()
    shipClass()
