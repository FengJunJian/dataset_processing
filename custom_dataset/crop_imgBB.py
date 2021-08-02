import os
import numpy as np
import pickle as pkl
import colorsys
from annotation_function import combined_roidb
from annotation_function import get_output_dir
from imgbb_function import write_detection_batch,crop_bb_transform
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CLASSES = ('__background__',  # 0
           'passenger ship',  # 1
           'ore carrier',  # 2
           'general cargo ship',  # 3
           'fishing boat',  # 4
           'Sail boat',  # 5
           'Kayak',  # 6
           'flying bird',  # flying bird/plane #7
           'vessel',  # vessel/ship #8
           'Buoy',  # 9
           'Ferry',  # 10
           'container ship',  # 11
           'Other',  # 12
           'Boat',  # 13
           'Speed boat',  # 14
           'bulk cargo carrier',  # 15
           )
hsv_tuples = [(x / len(CLASSES), 1., 1.)
              for x in range(len(CLASSES))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
colors = [c[::-1] for c in colors]

_class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))

dataset_path = 'E:/fjj/SeaShips_SMD/'
root_im = os.path.join(dataset_path, 'JPEGImages')
split_name = 'test1300'  # occlusion,light,scale
# log_path=model_name+'_test_0'
pkl_root = os.path.join('E:/fjj/semi_ship_test', split_name)  # '../data_processing_cache/'
im_save_root = 'E:/fjj/semi_ship_test'
dirs = os.listdir(pkl_root)
# pkl_file = os.path.join(pkl_root ,'model/detections.pkl')
imdb, roidb = combined_roidb("shipdataset", split_name, dataset_path)  #
# 创建文件
# fw = open(os.path.join(pkl_root, 'ap_' + '.txt'), 'w')

all_boxes = [[[] for _ in range(imdb.num_images)]
             for _ in range(imdb.num_classes)]

# drawimg_names = ['MVI_1478_VIS_00405', '002365', 'MVI_1486_VIS_00013', 'MVI_1474_VIS_00429', '003426']
drawimg_names = ['MVI_1478_VIS_00405', 'MVI_1486_VIS_00013', 'MVI_1474_VIS_00429', '002365', '003426']
rects = [[[1237, 274, 295.833333333333, 166.40625]],

         [[5.8333, 189.1562, 316.0000, 177.7500]],
         # [ 961.5000, 153.1562, 256.8333, 144.4688]],

         [[1518.5, 826.5625, 336.666666666668, 189.375]],
         # [39.5000000000007,792.0625,336.666666666668,189.375]],
         [[380.5, 182.5, 1037.16666666667, 583.40625]],

         [[1328, 246.4375, 580.166666666665, 326.34375]],
         ]

dict_files_rects = {}
for i in range(len(drawimg_names)):
    dict_files_rects[drawimg_names[i]]=np.array(rects[i],np.int)

for d in dirs:
    if len(os.path.splitext(d)[1]) != 0:
        continue
    det_file = os.path.join(pkl_root, d, 'detections.pkl')

    bboxes_dict = {}
    output_dir = get_output_dir(im_save_root, 'im5', d, '')
    with open(det_file, 'rb') as f:
        all_boxes = pkl.load(f)
    all_boxes = np.array(all_boxes)
    for i in range(imdb.num_images):
        # for c in range(imdb.num_classes):
        bboxes_dict[imdb._image_index[i]] = all_boxes[:, i] #shape(None,5)(xmin,ymin,xmax,ymax,scores)
    for imfile in drawimg_names:
        im = cv2.imread(os.path.join(root_im, imfile + '.jpg'))
        imtmp=im.copy()
        bboxes = bboxes_dict[imfile]#predicted bboxes
        predict_bboxes = np.empty((0, 5))#(xmin,ymin,xmax,ymax,class)
        class_inds = []
        for c in range(1, len(CLASSES)):
            if len(bboxes[c]) == 0:
                continue
            bboxes[c][:,4]=c
            predict_bboxes = np.concatenate([predict_bboxes, bboxes[c]], axis=0)
            #class_inds.extend([c] * len(bboxes[c]))

        im = write_detection_batch(im, predict_bboxes[:, 4].astype(np.int), predict_bboxes, CLASSES, colors)
        cv2.imwrite(os.path.join(output_dir, imfile + '.jpg'), im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        fig = plt.figure(1)
        fig.set_size_inches(im.shape[1] / 100.0, im.shape[0] / 100.0)  # 输出width*height像素
        plt.axis('off')
        currentAxis = plt.gca()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(im)
        rect = dict_files_rects[imfile]  # ROI rect
        for j,r in enumerate(rect):
            ROIrectangle = r.copy()
            ROIrectangle[2] += ROIrectangle[0]
            ROIrectangle[3] += ROIrectangle[1]
            ROIrectangle -= 1

            im_roi_ratio = im.shape[1] / r[2]

            boxes_crop=crop_bb_transform(predict_bboxes,ROIrectangle,im_roi_ratio)
            ROI = imtmp[ROIrectangle[1]:ROIrectangle[3], ROIrectangle[0]:ROIrectangle[2]].copy()

            ROI_resize = cv2.resize(ROI, None, fx=im_roi_ratio, fy=im_roi_ratio)
            ROI_resize = write_detection_batch(ROI_resize, boxes_crop[:, 4].astype(np.int), boxes_crop, CLASSES, colors)

            cv2.imencode('.jpg', ROI_resize)[1].tofile(
                os.path.join(output_dir, imfile+'c%d' % j  + '.jpg'))

            ROIrect = patches.Rectangle((r[0], r[1]), r[2], r[3], linewidth=5, edgecolor=(0,0,0), facecolor='none',
                                     linestyle='dotted')
            currentAxis.add_patch(ROIrect)


        #plt.style.use('classic')

        plt.savefig(os.path.join(output_dir, imfile+'_crop' + '.jpg'))
        fig.clf()
        #fig.cl
        #plt.show()

        #cv2.imwrite(os.path.join(output_dir, imfile+'_crop' + '.jpg'), im)
