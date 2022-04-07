import numpy as np
import random
import glob
import os
import shutil
'''
一个函数生成一种MainSet.txt
'''
abs_path='E:\\fyl\\WIDER_val\\'
dataset_path=os.path.join(abs_path,'JPEGImages\\')
MainSet_path=os.path.join(abs_path,'ImageSets\\Main\\')

'''
flag_num:
0: all.txt
1: all.txt 返回label与unlabel
2: subset of all.txt
'''

def mainSetFromImage(datasetPath,mainsetname=None,outFlag=False):
    '''
    make the mainset according the name of images
    '''
    #datasetPath='E:/fjj/Dataset1000_1'
    JPEGImagesPah=os.path.join(datasetPath,'JPEGImages')
    MainPath=os.path.join(datasetPath,'ImageSets/Main')
    files=os.listdir(JPEGImagesPah)
    basenames=[os.path.splitext(file)[0]+'\n' for file in files]

    if outFlag:
        return basenames
    else:
        if mainsetname is None:
            raise ValueError('mainsetname is None')
        with open(os.path.join(MainPath,'{}.txt'.format(mainsetname)),'w') as f:
            f.writelines(basenames)
        return None

    # targetAnnoPath=os.path.join(datasetPath,'Annotations')
    # sourceAnnoPath=os.path.join('E:/fjj/SeaShips_SMD/Annotations')
    # for basename in basenames:
    #     basename=basename.strip()
    #     shutil.copy(os.path.join(sourceAnnoPath,basename+'.xml'),targetAnnoPath)
    #files=glob.glob(os.path.join(JPEGImagesPah,'*.jpg'))

def mainSetFromAnno(datasetPath,mainsetname=None,outFlag=False):
    '''
    make the mainset according the name of annotations
    '''
    #datasetPath = 'E:/fjj/MarineShips2'
    AnnoPath = os.path.join(datasetPath, 'Annotations')
    xmlfiles = os.listdir(AnnoPath)
    basenames = [os.path.splitext(file)[0] + '\n' for file in xmlfiles]
    if outFlag:
        return basenames
    else:
        if mainsetname is None:
            raise ValueError('mainsetname is None')

        MainSetPath = os.path.join(datasetPath, 'ImageSets/Main')
        if not os.path.exists(MainSetPath):
            os.makedirs(MainSetPath)
        with open(os.path.join(MainSetPath, '{}.txt'.format(mainsetname)), 'w', encoding='utf-8') as f:
            f.writelines(basenames)
        # for basename in basenames:
        #     basename = basename.strip()
        #     f.write(basename + '\n')

        # shutil.copy(os.path.join(sourceAnnoPath,basename+'.xml'),targetAnnoPath)
    # files=glob.glob(os.path.join(JPEGImagesPah,'*.jpg'))

#if flag_num==0:#生成 all.txt
def all():
    if not os.path.exists(MainSet_path):
        os.makedirs(MainSet_path)
    #Imgs=glob.glob(dataset_path)
    Imgs=os.listdir(dataset_path)
    np.random.shuffle(Imgs)

    with open(os.path.join(MainSet_path,'all.txt'),'w') as f:
        for img in Imgs:
            image_index = os.path.splitext(img)[0]
            f.write(image_index+'\n')
def label_unlabel():
#elif flag_num==1:#生成label与unlabel训练集
    all=[]
    with open(os.path.join(MainSet_path,'all.txt'),'r') as f:
        all=[x.strip() for x in f.readlines()]
    total_num=len(all)
    train_num=5600#np.ceil(total_num*0.8)
    test_num=total_num-train_num
    trainset=all[:int(train_num)]
    testset=all[int(train_num):]
    label_num = 1600#np.ceil(train_num * 3 / 7.0)#np.ceil(train_num * 1.0 / 3.0)
    unlabel_num = train_num - label_num
    labelset=trainset[:int(label_num)]
    unlabelset=trainset[int(label_num):]
    lset=[]
    unset=[]
    with open(os.path.join(MainSet_path,'train1.txt'),'w') as f:
        for data in trainset:
            f.write(data+'\n')
    with open(os.path.join(MainSet_path, 'test1.txt'), 'w') as f:
        for data in testset:
            f.write(data + '\n')

    for i in range(4):
        with open(os.path.join(MainSet_path,'seaship_label'+str(i)+'.txt'),'w') as f:
            for ind in range(int(len(labelset)*np.math.pow(2,-i))):
                f.write(labelset[ind] + '\n')
        with open(os.path.join(MainSet_path,'seaship_unlabel'+str(i)+'.txt'),'w') as f:
            for ind in range(int(len(unlabelset)*np.math.pow(2,-i))):
                f.write(unlabelset[ind] + '\n')
#elif flag_num==2:
def test():
    all = []
    with open(os.path.join(MainSet_path, 'all.txt'), 'r') as f:
        all = [x.strip() for x in f.readlines()]
    test_num = 1300
    testset = all[:1300]
    with open(os.path.join(MainSet_path,'test%d.txt'%test_num),'w') as f:
        for data in testset:
            f.write(data+'\n')

def train_test_mfolder():
    folders=['E:/SeaShips_SMD/','G:/ShipDataSet/BXShipDataset']#16980，3435
    all=[]
    for folder in folders:
        basenames=mainSetFromAnno(folder,outFlag=True)
        all.extend(basenames)
    random.shuffle(all)
    trainset=all
    testset=all[:round(len(all)*0.1)]

    with open('trainShip.txt','w') as f:
        f.writelines(trainset)
    with open('testShip.txt','w') as f:
        f.writelines(testset)
    #all=np.random.permutation(all)

    # with open(os.path.join(MainSet_path,'test%d.txt'%test_num),'w') as f:
    #     for data in testset:
    #         f.write(data+'\n')

if __name__=='__main__':
    train_test_mfolder()

