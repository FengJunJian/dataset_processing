import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch_receptive_field import receptive_field, receptive_field_for_unit
from torchvision.models import resnet34,resnet50,vgg16
from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.avgpool(y)
        return y


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v1.7.0
device = "cpu"#torch.device("cpu") # PyTorch v1.7.0
#model = Net().to(device)
net_name="resnet50"#"resnet50" ,"vgg16"
model=getattr(models,net_name)(False)
# model=vgg16(False)
net_kv=OrderedDict(list(model.named_children()))
print(net_kv.keys())
if net_name=="resnet50":
    model=nn.Sequential(OrderedDict(list(model.named_children())[:-3])).to(device)
elif net_name=="vgg16":
    model=nn.Sequential(OrderedDict(list(model.named_children())[:1]))

receptive_field_dict = receptive_field(model, (3, 256, 256),device=device)
receptive_field_for_unit(receptive_field_dict, "2", (1,1))