'''
调整数据集中的边框，防止出边界
'''

import os
import xml.etree.ElementTree as ET


path='Annotations'
xmls=os.listdir(path)

for xml in xmls:
    filename=os.path.join(path,xml)
    tree = ET.parse(filename)
    size=tree.find('size')
    width=float(size.find('width').text)
    height=float(size.find('height').text)
    objs=tree.findall('object')
    flag=False
    for obj in objs:
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        if float(bbox.find('xmin').text)<0:
            print(bbox.find('xmin').text)
            bbox.find('xmin').text = '0'
            flag=True
        if float(bbox.find('xmax').text) > width:
            print(bbox.find('xmax').text)
            bbox.find('xmax').text = str(int(width))
            flag = True
        if float(bbox.find('ymin').text) < 0:
            print(bbox.find('ymin').text)
            bbox.find('ymin').text = '0'
            flag = True
        if float(bbox.find('ymax').text)>height:
            # bbox.find('xmin').text='0'
            print(bbox.find('ymax').text)
            bbox.find('ymax').text = str(int(height))
            flag = True
    if flag:
        print(filename)
        tree.write(filename)