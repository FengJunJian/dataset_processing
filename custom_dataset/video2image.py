import cv2
import os
import math
from glob import glob
from tqdm import tqdm
folder='G:/unlabel_image'
subpath='G:/fjj/毕业大论文/视频'#'H:\\dataset\\三亚-上海2019-5-12\\video'
videofile_format='*.MOV'#'DSC_2483.MOV'#'2019-05-18.10.04.42.mp4'#视频文件
videos=glob(os.path.join(subpath,videofile_format))
print('Total videos:',len(videos))
if not os.path.exists(folder):
   os.makedirs(folder)

for videofile in videos:
    videobase=os.path.splitext(os.path.basename(videofile))[0]

    cap=cv2.VideoCapture(os.path.join(subpath,videofile))#输入视频位置
    if cap.isOpened():
        flag, frame = cap.read()
        numFrames=1
        # while flag:
        #     flag, frame = cap.read()  # 读取第几帧
        #     numFrames+=1
        #numFrames=math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT))# 帧的总数

        #interval=162
        #numFrame=0
        numFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('width:',cap.get(cv2.CAP_PROP_FRAME_WIDTH),'height:',cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #continue
        interval = math.floor(numFrames / 25)  # 每个视频采样15帧，相隔interval帧采样,向下取整
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        count=1
        for i in tqdm(range(numFrames)):
            flag, frame = cap.read()  # 读取第几帧
            if flag:
                #print(i%interval)
                if i%interval==0:
                    cv2.imshow('a',frame)#显示帧
                    cv2.waitKey(15)
                    imagefile=os.path.join(folder,'{}_{}.jpg'.format(videobase,count))
                    if not os.path.exists(imagefile):
                        cv2.imwrite(imagefile,frame)# 保存帧
                    count+=1
            else:
                break
            #numFrame+=1

print('End of Video!')