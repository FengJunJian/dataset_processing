import os
import shutil

path='E:/fjj/MarineShips2/JPEGImages'
files=os.listdir(path)
for file in files:
    oldfilename=os.path.join(path,file)
    newfilename=os.path.join(path,file.replace(' ','_'))
#newfile=file.replace(' ','_')
    os.rename(oldfilename,newfilename)