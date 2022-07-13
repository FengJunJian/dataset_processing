import torch
def scatter2one_hot(labels,num_classes):
    ''' index = torch.tensor([1, 2, 1, 2, 0])
    labels:[1, 2, 1, 2, 0]'''
    #print(labels.shape)
    index = labels.unsqueeze(1)
    print(index.shape)
    # print(index)
    a = torch.zeros(index.shape[0], num_classes).scatter_(1, index, 1)
    print(a)
    return a