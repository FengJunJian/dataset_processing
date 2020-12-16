import numpy as np
import os
import glob
abs_path='E:\\fjj\\SeaShips\\'
dataset_path=os.path.join(abs_path,'JPEGImages\\')
MainSet_path=os.path.join(abs_path,'ImageSets\\Main\\')
flag_num=1
'''
0: all.txt
1: all.txt 返回label与unlabel
'''
if flag_num==0:#生成 all.txt
    if not os.path.exists(MainSet_path):
        os.makedirs(MainSet_path)
    #Imgs=glob.glob(dataset_path)
    Imgs=os.listdir(dataset_path)
    np.random.shuffle(Imgs)

    with open(os.path.join(MainSet_path,'all.txt'),'w') as f:
        for img in Imgs:
            image_index = os.path.splitext(img)[0]
            f.write(image_index+'\n')
elif flag_num==1:#生成label与unlabel训练集
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
elif flag_num==2:
    pass

