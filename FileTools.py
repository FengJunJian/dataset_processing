import os
import shutil

def renameBatch():
    '''
    rename
    '''
    path='E:/fjj/MarineShips2/JPEGImages'
    files=os.listdir(path)
    for file in files:
        oldfilename=os.path.join(path,file)
        newfilename=os.path.join(path,file.replace(' ','_'))
    #newfile=file.replace(' ','_')
        os.rename(oldfilename,newfilename)

def copyBatch():
    '''
    moveFiles
    '''
    srcpath='E:/SeaShips_SMD/JPEGImages'
    dstpath='../data_processing_cache/Results'
    if not os.path.exists(dstpath):
        os.mkdir(dstpath)
    files=os.listdir(srcpath)
    for file in files:
        oldfilename = os.path.join(srcpath, file)
        newfilename = os.path.join(dstpath, 'a'+file)
        shutil.copy(oldfilename,newfilename)


def removeBatch():
    '''
    批量删除文件

    '''
    root='E:/FE/'
    files=os.listdir(root)
    for i in range(0,len(files),4):
        os.remove(os.path.join(root,files[i]))

if __name__ == '__main__':
    copyBatch()