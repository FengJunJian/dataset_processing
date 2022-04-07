import os
import numpy as np
import pickle as pkl
import colorsys
from annotation_function import combined_roidb
from annotation_function import get_output_dir
from custom_dataset.imgbb_function import write_detection_batch
import cv2
CLASSES=('__background__',#0
                    'passenger ship',#1
                    'ore carrier',#2
                    'general cargo ship',#3
                    'fishing boat',#4
                    'Sail boat',#5
                    'Kayak',#6
                    'flying bird',#flying bird/plane #7
                    'vessel',#vessel/ship #8
                    'Buoy',#9
                    'Ferry',#10
                    'container ship',#11
                    'Other',#12
                    'Boat',#13
                    'Speed boat',#14
                    'bulk cargo carrier',#15
)
hsv_tuples = [(x / len(CLASSES), 1., 1.)
                      for x in range(len(CLASSES))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
colors=[c[::-1] for c in colors]


_class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))

dataset_path = 'E:/fjj/SeaShips_SMD/'
root_im=os.path.join(dataset_path,'JPEGImages')
split_name = 'test1300'  # occlusion,light,scale
# log_path=model_name+'_test_0'
pkl_root = os.path.join('E:/fjj/semi_ship_test', split_name)  # '../data_processing_cache/'
im_save_root='E:/fjj/semi_ship_test'
dirs = os.listdir(pkl_root)
# pkl_file = os.path.join(pkl_root ,'model/detections.pkl')
imdb, roidb = combined_roidb("shipdataset", split_name, dataset_path)  # 测试数据集test1300,test650
# 创建文件
#fw = open(os.path.join(pkl_root, 'ap_' + '.txt'), 'w')

all_boxes = [[[] for _ in range(imdb.num_images)]
             for _ in range(imdb.num_classes)]
'''
all_boxes:shape (num_images,num_classes)
each boxes is stored as np.array([xmin,ymin,xmax,ymax,score])
that is the the second image and 1-index class  
all_boxes[1][2]=boxes
boxes:shape (batch,5) :np.array([[xmin_1,ymin_1,xmax_1,ymax_1,score_1],......,[xmin_n,ymin_n,xmax_n,ymax_n,score_n]])
'''


# det_file=os.path.join(pkl_root, dirs[1],'detections.pkl')
# with open(det_file,'rb') as f:
#     bboxes=pkl.load(f)
# bboxes=np.array(bboxes)
# 
# for i in range(imdb.num_images):
#     #for c in range(imdb.num_classes):
#     bboxes_dict[imdb._image_index[i]]=bboxes[:,i]

drawimg_names=['MVI_1478_VIS_00405','002365','MVI_1486_VIS_00013','MVI_1474_VIS_00429','003426']


for d in dirs:
    if len(os.path.splitext(d)[1]) != 0:
        continue
    det_file = os.path.join(pkl_root, d, 'detections.pkl')

    bboxes_dict={}
    output_dir = get_output_dir(im_save_root, 'im5', d, '')
    with open(det_file, 'rb') as f:
        all_boxes = pkl.load(f)
    all_boxes=np.array(all_boxes)
    for i in range(imdb.num_images):
        # for c in range(imdb.num_classes):
        bboxes_dict[imdb._image_index[i]] = all_boxes[:, i]
    for imfile in drawimg_names:
        im=cv2.imread(os.path.join(root_im,imfile+'.jpg'))
        bboxes=bboxes_dict[imfile]
        predict_bboxes=np.empty((0,5))
        class_inds=[]
        for c in range(1,len(CLASSES)):
            if len(bboxes[c])==0:
                continue
            predict_bboxes=np.concatenate([predict_bboxes,bboxes[c]],axis=0)
            class_inds.extend([c]*len(bboxes[c]))
        im = write_detection_batch(im,class_inds, predict_bboxes,CLASSES,colors)
        cv2.imwrite(os.path.join(output_dir, imfile + '.jpg'), im)
    