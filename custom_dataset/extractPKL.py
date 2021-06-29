import os
import numpy as np
import pickle as pkl
from annotation_function import combined_roidb
# from tool_function import get_output_dir
import cv2


path='E:/fjj/semi_ship_test/vgg16_semi/test1300/label0_unlabel0_192002top250/semi_model'
det_file=os.path.join(path,'detections.pkl')

dataset_path='E:/fjj/SeaShips_SMD/'
split_name='test1300'#occlusion,light,scale

# pkl_root = os.path.join('E:/fjj/semi_ship_test',split_name)#'../data_processing_cache/'
# dirs=os.listdir(pkl_root)
#pkl_file = os.path.join(pkl_root ,'model/detections.pkl')
imdb, roidb = combined_roidb("shipdataset", split_name,dataset_path)  # 测试数据集test1300,test650
imdbl, _ = combined_roidb("shipdataset", 'light',dataset_path)
imdbs, _ = combined_roidb("shipdataset", 'scale',dataset_path)
imdbo, _ = combined_roidb("shipdataset", 'occlusion',dataset_path)
imdbn,_=combined_roidb("shipdataset", 'normal',dataset_path)

# residualSet=[]
# residualSet.extend(imdbl._image_index)
# residualSet.extend(imdbs._image_index)
# residualSet.extend(imdbo._image_index)
# residualSet=set(residualSet)
# all=set(imdb._image_index)
# rSet=all-residualSet
# with open('normal.txt','w') as f:
#     for s in rSet:
#         f.write(s+'\n')


bboxes_dict={}

all_boxes_l = [[[] for _ in range(imdbl.num_images)] for _ in range(imdbl.num_classes)]
all_boxes_s = [[[] for _ in range(imdbs.num_images)] for _ in range(imdbs.num_classes)]
all_boxes_o = [[[] for _ in range(imdbo.num_images)] for _ in range(imdbo.num_classes)]
all_boxes_n = [[[] for _ in range(imdbo.num_images)] for _ in range(imdbo.num_classes)]

with open(det_file,'rb') as f:
    bboxes=pkl.load(f)
bboxes1=np.array(bboxes)

for i in range(imdb.num_images):
    #for c in range(imdb.num_classes):
    bboxes_dict[imdb._image_index[i]]=bboxes1[:,i]

for c in range(imdb.num_classes):
    for i in range(imdbn.num_images):
        assert imdbn._image_index[i] in bboxes_dict, 'key not exists {}'.format(imdbn._image_index[i])
            #raise ValueError('key not exists')
        all_boxes_n[c][i]=bboxes_dict[imdbn._image_index[i]][c]

for c in range(imdb.num_classes):
    for i in range(imdbl.num_images):
        assert imdbl._image_index[i] in bboxes_dict,'key not exists {}'.format(imdbl._image_index[i])
            #raise ValueError('key not exists')
        all_boxes_l[c][i]=bboxes_dict[imdbl._image_index[i]][c]

for c in range(imdb.num_classes):
    for i in range(imdbs.num_images):
        assert imdbs._image_index[i] in bboxes_dict,'key not exists {}'.format(imdbs._image_index[i])
        all_boxes_s[c][i]=bboxes_dict[imdbs._image_index[i]][c]


for c in range(imdb.num_classes):
    for i in range(imdbo.num_images):
        assert imdbo._image_index[i] in bboxes_dict,'key not exists {}'.format(imdbo._image_index[i])
        all_boxes_o[c][i]=bboxes_dict[imdbo._image_index[i]][c]


with open('../../data_processing_cache/detections_normal.pkl', 'wb') as f:
    pkl.dump(all_boxes_n, f, pkl.HIGHEST_PROTOCOL)
with open('../../data_processing_cache/detections_light.pkl', 'wb') as f:
    pkl.dump(all_boxes_l, f, pkl.HIGHEST_PROTOCOL)
with open('../../data_processing_cache/detections_scale.pkl', 'wb') as f:
    pkl.dump(all_boxes_s, f, pkl.HIGHEST_PROTOCOL)
with open('../../data_processing_cache/detections_occlusion.pkl', 'wb') as f:
    pkl.dump(all_boxes_o, f, pkl.HIGHEST_PROTOCOL)