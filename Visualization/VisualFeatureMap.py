import os
import numpy as np
import logging
from PIL import Image
from math import ceil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
class BilinearInterpolation(object):
    def __init__(self, w_rate: float, h_rate: float, *, align='center'):
        if align not in ['center', 'left']:
            logging.exception(f'{align} is not a valid align parameter')
            align = 'center'
        self.align = align
        self.w_rate = w_rate
        self.h_rate = h_rate

    def set_rate(self,w_rate: float, h_rate: float):
        self.w_rate = w_rate    # w 的缩放率
        self.h_rate = h_rate    # h 的缩放率

    # 由变换后的像素坐标得到原图像的坐标    针对高
    def get_src_h(self, dst_i,source_h,goal_h) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_i = float(dst_i * (source_h/goal_h))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_i = float((dst_i + 0.5) * (source_h/goal_h) - 0.5)
        src_i += 0.001
        src_i = max(0.0, src_i)
        src_i = min(float(source_h - 1), src_i)
        return src_i
    # 由变换后的像素坐标得到原图像的坐标    针对宽
    def get_src_w(self, dst_j,source_w,goal_w) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_j = float(dst_j * (source_w/goal_w))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_j = float((dst_j + 0.5) * (source_w/goal_w) - 0.5)
        src_j += 0.001
        src_j = max(0.0, src_j)
        src_j = min((source_w - 1), src_j)
        return src_j

    def transform(self, img):
        source_h, source_w, source_c = img.shape  # (235, 234, 3)
        goal_h, goal_w = round(
            source_h * self.h_rate), round(source_w * self.w_rate)
        new_img = np.zeros((goal_h, goal_w, source_c), dtype=np.uint8)

        for i in range(new_img.shape[0]):       # h
            src_i = self.get_src_h(i,source_h,goal_h)
            for j in range(new_img.shape[1]):
                src_j = self.get_src_w(j,source_w,goal_w)
                i2 = ceil(src_i)
                i1 = int(src_i)
                j2 = ceil(src_j)
                j1 = int(src_j)
                x2_x = j2 - src_j
                x_x1 = src_j - j1
                y2_y = i2 - src_i
                y_y1 = src_i - i1
                new_img[i, j] = img[i1, j1]*x2_x*y2_y + img[i1, j2] * \
                    x_x1*y2_y + img[i2, j1]*x2_x*y_y1 + img[i2, j2]*x_x1*y_y1
        return new_img

def visualize_feature_map(img_batch,out_path,type,BI):
    feature_map = torch.squeeze(img_batch)
    feature_map = feature_map.detach().cpu().numpy()
    feature_map_sum=feature_map.sum(0)
    feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
    feature_map_sum = BI.transform(feature_map_sum)
    plt.imshow(feature_map_sum)
    plt.savefig(out_path + "sum_{}.jpg".format(type))
    print("save sum_{}.jpg".format(type))

    feature_map_sum = feature_map[0, :, :]
    feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
    for i in range(0, 2048):
        feature_map_split = feature_map[i,:, :]
        feature_map_split = np.expand_dims(feature_map_split,axis=2)
        if i > 0:
            feature_map_sum +=feature_map_split
        feature_map_split = BI.transform(feature_map_split)

        plt.imshow(feature_map_split)
        plt.savefig(out_path + str(i) + "_{}.jpg".format(type) )
        plt.xticks()
        plt.yticks()
        plt.axis('off')




def image_proprecess(img,size=(224,224)):

    data_transforms = transforms.Compose([
        transforms.Resize(size, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    data = data_transforms(img)
    data = torch.unsqueeze(data,0)
    return data
def Init_Setting():
    #dirname = '/mnt/share/VideoReID/share/models/Methods5_trial1'
    # model = siamese_resnet50(701, stride=1, pool='avg')
    model = resnet50(True)
    backbone=nn.Sequential(*list(model.children())[:-2])
    #backbone[0]
    #trained_path = os.path.join(dirname, 'net_%03d.pth' % epoch)
    #print("load %03d.pth" % epoch)
    #model.load_state_dict(torch.load(trained_path))
    backbone = backbone.cuda().eval()
    return backbone
#使用方法


imgs_path = "../"
save_path = "../../data_processing_cache/visual/"

model = Init_Setting()
img_path=os.path.join(imgs_path , "MVI_1644_VIS_00203.jpg")
img = Image.open(img_path)
original_size=img.size
size=(224,224)
#model.avgpool=nn.Identity()
BI = BilinearInterpolation(original_size[0]/7.0, original_size[1]/7.0)

data = image_proprecess(img)
data = data.cuda()
output = model(data)
if not os.path.exists(save_path):
    os.makedirs(save_path)
visualize_feature_map(output, save_path, "drone", BI)

# BI = BilinearInterpolation(8, 8)
# feature_map = BI.transform(feature_map)