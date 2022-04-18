'''
version:0.1

'''
import shutil
import os
from tqdm import tqdm

path='E:/SeaShips_SMD/dataset/test_YOLOv5M/images'
Mainsets=['E:/SeaShips_SMD/ImageSets/Main/test.txt']

with open(Mainsets[0],'r') as f:
    Jspath='E:/SeaShips_SMD/JPEGImages'
    Aspath = 'E:/SeaShips_SMD/Annotations'
    Jtpath='G:/ShipProjectCode/PackImg/JPEGImages'
    Atpath='G:/ShipProjectCode/PackImg/Annotations'
    try:
        os.makedirs(Jtpath)
    except:
        pass
    try:
        os.makedirs(Atpath)
    except:
        pass

    lines=f.readlines()
    lines=[line.strip() for line in lines]

for file_id in tqdm(lines):
    shutil.copy(os.path.join(Jspath, '%s.jpg' % file_id), Jtpath)
    shutil.copy(os.path.join(Aspath, '%s.xml' % file_id), Atpath)