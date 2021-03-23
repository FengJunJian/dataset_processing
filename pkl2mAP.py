import pickle
import os
import pandas as pd
# path='E:/fjj/Faster-RCNN/semi_ship/semi-test600/model1.0\ship_whole4000_full/10000/'
path='H:/code/Faster-RCNN/semi_ship/semi-test600/'
files=[file for file in os.listdir(path) if os.path.splitext(file)[1]=='.pkl' and file != 'detections.pkl']#获取其他类别目标PR


def detection_walk(dir):#返回含有检测结果的文件路径（含有pr）
    paths = []
    # ckpts=[]
    for root,dirs,files in os.walk(dir):
        # print(root)
        # print(dirs)
        for name in files:
            #basename,ext=os.path.splitext(name)
            if name=='detections.pkl':
                paths.append(root)
    return paths

AP='mAP.txt'
PRcurve='PR.txt'
fAP=open(AP,'w')
fPR=open(PRcurve,'w')

paths=detection_walk(path)

for p in paths:#文件所在路径
    # fAP.write(p+'\n')
    # fPR.write(p+'\n')
    files = [file for file in os.listdir(p) if os.path.splitext(file)[1] == '.pkl' and file != 'detections.pkl']
    mAP=0
    for file in files:
        fAP.write(p+'/'+file + '\n')
        fPR.write(p+'/'+file + '\n')
        with open(os.path.join(p,file),'rb') as f:
            try:
                PR=pickle.load(f)
                assert len(PR['rec'])==len(PR['prec'])#判断
            except:
                print('Error:'+file)
                continue
                #exit(1)
        try:
            fAP.write(str(PR['ap'])+'\t')
            mAP+=PR['ap']
            fAP.write(str(PR['precision'])+'\t'+str(PR['recall'])+'\n')
        except:
            print('PR Error:'+p+file)
            continue
    mAP/=6
    fAP.write(str(mAP)+'\n\n')
        # with open(save_name, 'w') as f:
        #     for i in range(len(PR['rec'])):
        #         f.write(str(PR['prec'][i])+'\t'+str(PR['rec'][i])+'\n')

fAP.close()
fPR.close()
