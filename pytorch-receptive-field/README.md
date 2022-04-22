# pytorch-receptive-field

[![Build Status](https://travis-ci.com/Fangyh09/pytorch-receptive-field.svg?branch=master)](https://travis-ci.com/Fangyh09/pytorch-receptive-field)

Compute CNN receptive field size in pytorch


## Usage
`git clone https://github.com/Fangyh09/pytorch-receptive-field.git`

```python
from torch_receptive_field import receptive_field
receptive_field(model, input_size=(channels, H, W))
```

Or
```python
from torch_receptive_field import receptive_field
dict = receptive_field(model, input_size=(channels, H, W))
receptive_field_for_unit(receptive_field_dict, "2", (2,2))
```

## Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_receptive_field import receptive_field

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.maxpool(y)
        return y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)

receptive_field_dict = receptive_field(model, (3, 256, 256))
receptive_field_for_unit(receptive_field_dict, "2", (2,2))
```
```
------------------------------------------------------------------------------
        Layer (type)    map size      start       jump receptive_field
==============================================================================
        0             [256, 256]        0.5        1.0             1.0
        1             [128, 128]        0.5        2.0             7.0
        2             [128, 128]        0.5        2.0             7.0
        3             [128, 128]        0.5        2.0             7.0
        4               [64, 64]        0.5        4.0            11.0
==============================================================================
Receptive field size for layer 2, unit_position (1, 1),  is
 [(0, 6.0), (0, 6.0)]
```

## More
`start` is the center of first item in the map grid .

`jump` is the distance of the adjacent item in the map grid.

`receptive_field` is the field size of the item in the map grid.


## Todo
- [x] Add Travis CI 
  

## Related
Thanks @pytorch-summary

https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807



# Other

url: https://github.com/google-research/receptive_field

| convnet model       | receptive field | effective stride | effective padding | FLOPs (Billion) |
| ------------------- | --------------- | ---------------- | ----------------- | --------------- |
| alexnet_v2          | 195             | 32               | 64                | 1.38            |
| vgg_16              | 212             | 32               | 90                | 30.71           |
| inception_v2        | 699             | 32               | 318               | 3.88            |
| inception_v3        | 1311            | 32               | 618               | 5.69            |
| inception_v4        | 2071            | 32               | 998               | 12.27           |
| inception_resnet_v2 | 3039            | 32               | 1482              | 12.96           |
| mobilenet_v1        | 315             | 32               | 126               | 1.14            |
| mobilenet_v1_075    | 315             | 32               | 126               | 0.65            |
| resnet_v1_50        | 483             | 32               | 239               | 6.96            |
| resnet_v1_101       | 1027            | 32               | 511               | 14.39           |
| resnet_v1_152       | 1507            | 32               | 751               | 21.81           |
| resnet_v1_200       | 1763            | 32               | 879               | 28.80           |

