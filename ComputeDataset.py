from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import copy
def ComputeDataset(ImagePath):
    '''
    Compute the mean and std
    ImagePath: the Path to Images
    '''
    means = [0, 0, 0]
    stdevs = [0, 0, 0]
    dataset=ImageFolder(ImagePath,transforms.Compose([transforms.ToTensor()]))

    num_imgs=len(dataset)
    for data in tqdm(dataset):
        img=data[0]
        for i in range(3):
            means[i]+=img[i,:,:].mean()
            stdevs[i]+=img[i,:,:].std()

    means=np.asarray(means)/num_imgs
    stdevs=np.asarray(stdevs)/num_imgs
    print(means,stdevs)

def BalanceDatasetClassification(ImagePath):
    '''
    Compute the balance weight for different classes
    ImagePath: Path for images
    '''
    datatransforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64,128))#(h,w)
    ])
    dataset = ImageFolder(ImagePath,transforms.ToTensor())
    N=len(dataset)
    train_size=int(N*0.8)
    val_size=int(N*0.1)
    test_size=N-train_size-val_size

    trainDataset,valDataset,testDataset=torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

    trainDataset.dataset=copy(trainDataset.dataset)
    valDataset.dataset = copy(valDataset.dataset)

    weights = make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights),replacement=True)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                               sampler=sampler, num_workers=16, pin_memory=True,prefetch_factor=4)
    return train_loader


def make_weights_for_balanced_classes(images, nclasses):

    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight



if __name__ == '__main__':
    path='../data_processing_cache/Classification'
    ComputeDataset(path)