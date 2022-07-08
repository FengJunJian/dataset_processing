import torchvision
import torch
import thop


x = torch.randn(1, 3, 224, 224)
modelt = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()
flops1,params1=thop.profile(modelt, (x,))
print("%s ------- params: %.2fMB ------- flops: %.2fG" % (
    modelt._get_name(), params1 / (1000 ** 2), flops1 / (1000 ** 3)))  #


