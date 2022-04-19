from torchvision.models import resnet34
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torchvision
import os
from PIL import Image
from numpy import linalg as LA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from collections import OrderedDict
import cv2
from tqdm import tqdm
import shutil
from collections import Counter
stdrgb=[0.229, 0.224, 0.225]
meanrgb=[0.485, 0.456, 0.406]

base_model = resnet34(pretrained=True,)  # 这里也可以使用自己的数据集进行训练
Transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=meanrgb,std=stdrgb)])
extractor=torch.nn.Sequential(OrderedDict(list(base_model.named_children())[:-1]))
def get_image_feature(data):
    model = extractor
    features = model(data)
    return features

class DataKmeans(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        """
                Args:
                    index (int): Index

                Returns:
                    tuple: (sample, target) where target is class_index of the target class.
                """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path,sample, target

def showData(dataset):
    data=iter(dataset)
    im,label=next(data)
    imo=(im*torch.tensor(stdrgb).view(3,1,1)+torch.tensor(meanrgb).view(3,1,1))*255
    #torch.tran
    imo=imo.numpy().astype(np.uint8)
    imshow=np.transpose(imo,(1,2,0))

    cv2.imshow('a',imshow[:,:,::-1])
    cv2.waitKey(1)
if __name__=="__main__":

    pathnpz = "../../data_processing_cache/dataKmeans/data13.npz"
    data_dir = 'E:/FE/UNO/datasets/Classification_advanced'
    if not os.path.exists(pathnpz):

        dataset = DataKmeans(data_dir, Transform)  # Transform

        dataloader = DataLoader(
            dataset,
            batch_size=256,
            # sampler=sampler,
            # shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2)
        base_model = resnet34(pretrained=True, )  # 这里也可以使用自己的数据集进行训练
        Transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=meanrgb, std=stdrgb)])
        extractor = torch.nn.Sequential(OrderedDict(list(base_model.named_children())[:-1]))

        feature_ls = []
        names = []
        gts=[]
        for path,im,target in tqdm(dataloader):
            #features = get_image_feature(im)
            features = extractor(im)
            features=features.detach().numpy()
            features=features.reshape(-1,512)
            # 特征标准化
            vec = features / LA.norm(features,axis=1,keepdims=True)

            # print(name,"==", vec.tolist())
            names.extend(path)
            gts.extend(target)
            feature_ls.extend(vec.tolist())


        np.savez(pathnpz,names=names,gts=gts,features=feature_ls)
    flag_Kmeans=True
    data=np.load(pathnpz)
    feature_ls=data['features']
    names=data['names']
    gts=data['gts']
    names_df = pd.DataFrame(names)
    if 'plabels' not in data.keys() or flag_Kmeans:
        df = pd.DataFrame(feature_ls)

        samples = df.values
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(samples)  # 训练模型
        plabels = kmeans.predict(samples)  # 预测
        np.savez(pathnpz,names=names,gts=gts,features=feature_ls,plabels=plabels)
    plabels = data['plabels']
    folders=os.listdir(data_dir)
    savedir="../../data_processing_cache/dataKmeans/"
    savefolders=[]
    for i in range(len(folders)):
        try:
            savefolders.append(os.path.join(savedir,"%02d"%(i)))
            os.makedirs(os.path.join(savedir,"%02d"%(i)))
        except:
            continue

    for i in set(plabels):
        #print(labels, names_df[labels == 0])
        files=names_df[plabels==i]
        count=Counter(gts[files.index])
        print(i,count.most_common())
        for ind,file in zip(files.index,files.values):
            #print(ind,file)
            name=file.item()
            #folder=os.path.basename(os.path.dirname(name))
            gt=gts[ind]
            basename=os.path.basename(name)
            basename,ext=os.path.splitext(basename)
            shutil.copy(name,os.path.join(savefolders[i],"{basename}_{comment}{ext}".format(basename=basename,comment=str(gt)+'To'+str(i),ext=ext)))
