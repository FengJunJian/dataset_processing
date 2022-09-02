from thop import profile,clever_format
import torchvision.models as models
from torchvision.models.resnet import Bottleneck,_resnet,ResNet,BasicBlock,conv1x1,conv3x3
import torch
from torch import nn

class Bottleneck_(Bottleneck):
    expansion: int = 1
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample = None,
            groups = 1,
            base_width = 64,
            dilation = 1,
            norm_layer = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = inplanes#int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

class ResNet_(ResNet):
    def __init__(
        self,
        block,
        layers,
        k=4,
        num_classes = 1000,
        zero_init_residual = False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.width=k
        self.inplanes = 16
        inputDim=self.inplanes
        self.dilation = 1
        #block.expansion=1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inputDim*self.width, layers[0]) #64
        self.layer2 = self._make_layer(block, inputDim*2*self.width, layers[1], stride=2,#128
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, inputDim*4*self.width, layers[2], stride=2,#256
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, inputDim*8*self.width, layers[3], stride=2,#512
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inputDim*8*self.width * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x ):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def WRN(*t,**kwargs):
    kwargs['width_per_group'] = 32 * 2
    #kwargs['num_classes']=2
    kwargs['k']=2
    model = ResNet_(Bottleneck_, [4,4,4,4], **kwargs)#[4,4,4,0] [3, 4, 6, 3]
    return model

# model=ResNet(Bottleneck)
x=torch.randn(1,3,224,224)
wrn=WRN(num_classes=6)
# models.resnext50_32x4d()
#models.densenet121()
#wrn=_resnet('wide_resnet50_2', Bottleneck, [4, 4, 3, 1],#[3, 4, 6, 3]
# False, True, width_per_group=64*2)
flops, params = profile(wrn, inputs=(x, ))
print("%s | Params: %.2fM | FLOPs: %.2fG" % (('WRN',params / (1000 ** 2),flops / (1000 ** 3))))
# flops,params=clever_format([flops,params],"%.3f")
# print("%s | Params: %s | FLOPs: %s" % (('WRN',params,flops)))
#def WRN():
#a=models.wide_resnet50_2()

#model = models.densenet121()
#input = torch.randn(1, 3, 224, 224)
#flops, params = profile(model, inputs=(input, ))
names=['alexnet','vgg16','vgg19','wide_resnet50_2','googlenet','resnet50','resnet101','densenet121','squeezenet1_0','resnext50_32x4d','mobilenet_v2','mnasnet0_5']

outputs=[]
for name in names:
    m=models.__dict__[name](False,num_classes=6)
    flops, params = profile(m, inputs=(x, ))
    # flops, params = clever_format([flops, params], "%.2f")
    #print("%s | Params: %s | FLOPs: %s" % (('WRN', params, flops)))
    outputs.append((name,params / (1000 ** 2),flops / (1000 ** 3)))
    #outputs.append((name,params,flops))
#models.wide_resnet50_2()
# models.resnet50()
print("-----------------------------------------------------------------")
for output in outputs:
    # print("%s | Params: %s | FLOPs: %s" % ((output[0], output[1], output[2])))
    print("%s | Params: %.2fM | FLOPs: %.2fG" % (output[0], output[1], output[2]))
print("-----------------------------------------------------------------")

wrn=WRN()

flops, params = profile(wrn, inputs=(x, ))
print("%s | Params: %.2fM | FLOPs: %.2fG" % (('WRN',params / (1000 ** 2),flops / (1000 ** 3))))
#flops,params=clever_format([flops,params],"%.2f")
# print("%s | Params: %s | FLOPs: %s" % (('WRN',params,flops)))