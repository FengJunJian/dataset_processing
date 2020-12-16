'''
make the mainset according the name of images
'''
import os
import glob
datasetPath='E:/fjj/Dataset100'
JPEGImagesPah=os.path.join(datasetPath,'JPEGImages')
MainPath=os.path.join(datasetPath,'ImageSets/Main')
files=os.listdir(JPEGImagesPah)
basenames=[os.path.splitext(file)[0]+'\n' for file in files]
with open(os.path.join(MainPath,'test.txt'),'w') as f:
    f.writelines(basenames)

targetAnnoPath=os.path.join(datasetPath,'Annotations')
sourceAnnoPath=os.path.join()
#files=glob.glob(os.path.join(JPEGImagesPah,'*.jpg'))


