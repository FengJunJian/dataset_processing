'''
make the mainset according the name of annotations
'''
import os
import glob
import shutil
datasetPath='E:/fjj/MarineShips2'
# JPEGImagesPah=os.path.join(datasetPath,'JPEGImages')
# MainPath=os.path.join(datasetPath,'ImageSets/Main')
# files=os.listdir(JPEGImagesPah)
# basenames=[os.path.splitext(file)[0]+'\n' for file in files]
# with open(os.path.join(MainPath,'test.txt'),'w') as f:
#     f.writelines(basenames)

AnnoPath=os.path.join(datasetPath,'Annotations')
xmlfiles=os.listdir(AnnoPath)
basenames=[os.path.splitext(file)[0]+'\n' for file in xmlfiles]
# sourceAnnoPath=os.path.join('E:/fjj/SeaShips_SMD/Annotations')
MainSetPath=os.path.join(datasetPath,'ImageSets/Main')
if not os.path.exists(MainSetPath):
    os.makedirs(MainSetPath)
with open(os.path.join(MainSetPath,'all.txt'),'w',encoding='utf-8') as f:
    for basename in basenames:
        basename=basename.strip()
        f.write(basename+'\n')

    #shutil.copy(os.path.join(sourceAnnoPath,basename+'.xml'),targetAnnoPath)
#files=glob.glob(os.path.join(JPEGImagesPah,'*.jpg'))


