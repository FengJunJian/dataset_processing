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

def compareDataDistributionDemo(a,b):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(1)
    a = np.array([384, 1758, 1213, 1729, 803, 1725, 258, 45233, 1291, 3642, 709, 4666, 507, 3374, 1570])  # 68862
    a = a / a.sum()
    b = np.array([35, 158, 142, 188, 86, 148, 22, 4344, 126, 358, 66, 447, 46, 304, 166])  # 6636
    b = b / b.sum()
    bar_width = 0.3
    x = np.arange(len(a))
    # plt.figure(1)
    plt.bar(x, a, width=bar_width)
    # plt.figure(2)
    plt.bar(x + bar_width, b, width=bar_width, align="center")

def drawClassDistribution(file="dataClassDistribution.npz"):
    import matplotlib.pyplot as plt
    import numpy as np

    data=np.load(file)
    a = data['labeled']
    b = data['unlabeled']
    c = data['training']
    d = data['test']
    DataSum = [a.sum(), b.sum(), c.sum(), d.sum()]
    # a=np.array([384, 1758, 1213, 1729, 803, 1725, 258, 45233, 1291, 3642, 709, 4666, 507, 3374, 1570])  # 68862
    # b=np.array([35,158,142,188,86,148,22,4344,126,358,66,447,46,304,166])#6636
    # b =b/ b.sum()
    plt.figure(1)
    a = a / a.sum()
    b = b / b.sum()
    c = c / c.sum()
    d = d / d.sum()
    bar_width = 0.2
    x = np.arange(len(a))

    plt.bar(x, a, width=bar_width)
    # plt.figure(2)
    plt.bar(x + bar_width, b, width=bar_width)
    plt.bar(x + 2 * bar_width, c, width=bar_width)
    plt.bar(x + 3 * bar_width, d, width=bar_width)
    plt.legend(
        ['labeled data:' + str(DataSum[0]), 'unlabeled data:' + str(DataSum[1]), 'training data:' + str(DataSum[2]),
         'test data:' + str(DataSum[3])])
    plt.savefig("dataClassDistribution.jpg", bbox_inches = 'tight')  #
    # plt.hist(a, bins=len(a))
    # plt.hist(b)
if __name__ == '__main__':
    #path='../data_processing_cache/Classification'
    #ComputeDataset(path)

    drawClassDistribution("E:/SSL/unbiased-teacher/dataClassDistribution.npz")