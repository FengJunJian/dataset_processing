import os
import cv2
import numpy as np
import colorsys
from imgbb_function import write_detection_batch,crop_bb_transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from annotation_function import annotation_onefile

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


def write_bb_black(im,dets):
    for i in range(len(dets)):
        bbox = dets[i, :4].astype(np.int32)
        im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,0), 3)
    return im

def draw_onefile():
    #files=['DSC_5756_12','DSC_6586_24','MVI_1587_VIS_00152','MVI_1619_VIS_00430']
    #files=['002365','MVI_1478_VIS_00405','MVI_1474_VIS_00429','MVI_1486_VIS_00013','003426']

    files = ['MVI_1478_VIS_00405', 'MVI_1486_VIS_00013','MVI_1474_VIS_00429','002365','003426']
    rects=[[[1237,274, 295.833333333333,166.40625]],

           [[5.8333,189.1562, 316.0000, 177.7500]],
           #[ 961.5000, 153.1562, 256.8333, 144.4688]],

           [[1518.5,826.5625,336.666666666668,189.375]],
            #[39.5000000000007,792.0625,336.666666666668,189.375]],
           [[380.5,182.5 ,1037.16666666667 ,583.40625]],

           [[1328, 246.4375, 580.166666666665, 326.34375]],
            ]
    dict_files_rects = {}
    for i in range(len(files)):
        dict_files_rects[files[i]]=np.array(rects[i],np.int)

    #rect_w=rect[3]/rect[2]
    root_xml='E:/fjj/SeaShips_SMD/Annotations'#'
    root_im='E:/fjj/SeaShips_SMD/JPEGImages'#'E:/'
    _class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))
    for i in range(len(files)):

        file=files[i]
        imgpath = os.path.join(root_im, file + '.jpg')
        basename = os.path.splitext(os.path.basename(imgpath))[0]
        xmlpath = os.path.join(root_xml, basename + '.xml')

        gts, cls = annotation_onefile(xmlpath)
        class_inds = [_class_to_ind[c] for c in cls]
        im = cv2.imread(imgpath)
        imtmp=im.copy()
        rect = dict_files_rects[file]
        im = write_detection_batch(im, class_inds, gts, CLASSES, colors)
        cv2.imencode('.jpg', im)[1].tofile(os.path.join('E:/paper/半监督船舶检测en/复杂场景图1/', basename + '.jpg'))
        im1 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        fig = plt.figure(1)
        fig.set_size_inches(im.shape[1] / 100.0, im.shape[0] / 100.0)  # 输出width*height像素
        plt.axis('off')
        currentAxis = plt.gca()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(im1)
        for j,r in enumerate(rect):
            ROIrectangle = r.copy()
            ROIrectangle[2] += ROIrectangle[0]
            ROIrectangle[3] += ROIrectangle[1]
            ROIrectangle -= 1

            im_roi_ratio=im.shape[1]/r[2]
            #imtmp=im.copy()

            #2
            #filter
            gt_labels=np.concatenate([gts,np.reshape(class_inds,(-1,1))],axis=1)
            boxes_crop=crop_bb_transform(gt_labels,ROIrectangle,im_roi_ratio)
            # boxes=gt_labels.copy()
            #
            # boxes[:, 0:4:2] = np.maximum(np.minimum(boxes[:, 0:4:2], ROIrectangle[2] ), ROIrectangle[0])
            # boxes[:, 1:4:2] = np.maximum(np.minimum(boxes[:, 1:4:2], ROIrectangle[3] ), ROIrectangle[1])
            #
            # inds=np.where(np.bitwise_not(np.bitwise_or((boxes[:,2]-boxes[:,0])==0,(boxes[:,3]-boxes[:,1])==0)))[0]
            # boxes_crop = boxes[inds].astype(np.float32)
            # boxes_crop[:,0:4:2]-= ROIrectangle[0]
            # boxes_crop[:, 1:4:2] -= ROIrectangle[1]
            # boxes_crop[:,0:4]*=im_roi_ratio
            ################ROI
            ROI=imtmp[ROIrectangle[1]:ROIrectangle[3],ROIrectangle[0]:ROIrectangle[2]].copy()
            im=cv2.rectangle(im,(ROIrectangle[0],ROIrectangle[1]),(ROIrectangle[2],ROIrectangle[3]),(0,0,255),1)
            ROI_resize=cv2.resize(ROI,None,fx=im_roi_ratio,fy=im_roi_ratio)
            ROI_resize = write_detection_batch(ROI_resize, boxes_crop[:,4].astype(np.int), boxes_crop, CLASSES, colors)

            cv2.imencode('.jpg',ROI_resize)[1].tofile(os.path.join('E:/paper/半监督船舶检测en/复杂场景图1/',basename+'c%d'%j+'.jpg'))
            ROIrect = patches.Rectangle((r[0], r[1]), r[2], r[3], linewidth=5, edgecolor=(0,0,0), facecolor='none',
                                        linestyle='dotted')
            currentAxis.add_patch(ROIrect)

        # cv2.imencode('.jpg', im)[1].tofile(
        #     os.path.join('E:/paper/半监督船舶检测en/复杂场景图1/', basename + '_crop' + '.jpg'))
        plt.savefig(os.path.join('E:/paper/半监督船舶检测en/复杂场景图1/', basename + '_crop' + '.jpg'))
        fig.clf()

if __name__ == '__main__':
    draw_onefile()




