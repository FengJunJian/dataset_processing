import numpy as np
import os
from shutil import copy
import scipy.io as scio
import cv2

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