'''
make the mainset according the name of images
'''
import os
import glob
import shutil
import random
datasetPath='E:/fjj/Dataset1000_1'
JPEGImagesPah=os.path.join(datasetPath,'JPEGImages')
MainSetPath=os.path.join(datasetPath,'ImageSets/Main')
files=os.listdir(JPEGImagesPah)
basenames=sorted([os.path.splitext(file)[0]+'\n' for file in files])
with open(os.path.join(MainSetPath,'all.txt'),'w') as f:
    f.writelines(basenames)

targetAnnoPath=os.path.join(datasetPath,'Annotations')
sourceAnnoPath=os.path.join('E:/fjj/SeaShips_SMD/Annotations')
for basename in basenames:
    basename=basename.strip()
    shutil.copy(os.path.join(sourceAnnoPath,basename+'.xml'),targetAnnoPath)
#files=glob.glob(os.path.join(JPEGImagesPah,'*.jpg'))

random.shuffle(basenames)
testnum = round(len(basenames) * 0.1)  #
trainset = sorted(basenames[testnum:])
testset = sorted(basenames[:testnum])
with open(os.path.join(MainSetPath, 'train.txt'), 'w') as f:
    f.writelines(trainset)
with open(os.path.join(MainSetPath, 'test.txt'), 'w') as f:
    f.writelines(testset)