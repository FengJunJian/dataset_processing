'''
寻找没有标注边框的样本
'''

import os
import xml.etree.ElementTree as ET

path='E:/fjj/SeaShips_SMD'
mainset_path='ImageSets/Main/label1.txt'
saveset_path='ImageSets/Main/label1r.txt'
filenames=None
with open(os.path.join(path,mainset_path),'r') as f:
    lines=f.readlines()
    filenames=[line.strip() for line in lines]
#xmls=os.listdir(path)

strings=''
for xml in filenames:
    filename=os.path.join(path,'Annotations',xml+'.xml')
    tree = ET.parse(filename)
    size=tree.find('size')
    width=float(size.find('width').text)
    height=float(size.find('height').text)
    objs=tree.findall('object')
    flag=False
    for obj in objs:
        flag = True
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        # if float(bbox.find('xmin').text)<0:
        #     print(bbox.find('xmin').text)
        #     bbox.find('xmin').text = '0'
        #
        # if float(bbox.find('xmax').text) > width:
        #     print(bbox.find('xmax').text)
        #     bbox.find('xmax').text = str(int(width))
        #
        # if float(bbox.find('ymin').text) < 0:
        #     print(bbox.find('ymin').text)
        #     bbox.find('ymin').text = '0'
        #
        # if float(bbox.find('ymax').text)>height:
        #     # bbox.find('xmin').text='0'
        #     print(bbox.find('ymax').text)
        #     bbox.find('ymax').text = str(int(height))
    if not flag:
        print(filename)
    else:
        strings+=xml+'\n'
with open(os.path.join(path,saveset_path),'w') as f:
    f.write(strings)

# if flag:
#     print(filename)
#     tree.write(filename)