import os
import cv2
import numpy as np
import colorsys
from visual_function import write_detection_batch

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

# def write_detection_batch(im, class_inds, dets):
#     # inds = np.where(dets[:, -1] >= thresh)[0]
#     # if len(inds) == 0:
#     #     return im
#     for i in range(len(dets)):
#         bbox = dets[i, :4].astype(np.int32)
#         # score = dets[i, -1]
#         im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[class_inds[i]], 2)#10
#
#         string = '%s' % (CLASSES[class_inds[i]])
#         fontFace = cv2.FONT_HERSHEY_COMPLEX
#         fontScale=0.6#2
#         thiness=1#2
#
#         text_size,baseline = cv2.getTextSize(string, fontFace, fontScale, thiness)
#         text_origin = (bbox[0],  bbox[1])#- text_size[1]
#
#         im=cv2.rectangle(im,(text_origin[0]-2,text_origin[1]+1),(text_origin[0]+text_size[0]+1,text_origin[1]-text_size[1]-2),colors[class_inds[i]],cv2.FILLED)
#         im = cv2.putText(im, '%s' % (CLASSES[class_inds[i]]), text_origin,
#                          fontFace, fontScale, (0, 0, 0), thiness)
#     return im

# def write_detection_batch(im, class_inds, dets):
#     # inds = np.where(dets[:, -1] >= thresh)[0]
#     # if len(inds) == 0:
#     #     return im
#     for i in range(len(dets)):
#         bbox = dets[i, :4].astype(np.int32)
#         # score = dets[i, -1]
#         rectangle_tmp = im.copy()
#         cv2.rectangle(rectangle_tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[class_inds[i]], 3)#10
#         #str1=CLASSES[class_inds[i]]
#         string = '%s' % (CLASSES[class_inds[i]])
#         fontFace = cv2.FONT_HERSHEY_COMPLEX#cv2.FONT_HERSHEY_COMPLEX
#         fontScale=1#2
#         thiness=1#2
#
#         text_size,baseline = cv2.getTextSize(string, fontFace, fontScale, thiness)
#         text_origin = (int(bbox[0]-1),  bbox[1])#- text_size[1]
#         #cv2.copyTo(im,dst=rectangle_tmp)
#
#         cv2.rectangle(rectangle_tmp,(text_origin[0]-1,text_origin[1]+1),(text_origin[0]+text_size[0],text_origin[1]-text_size[1]-1),colors[class_inds[i]],cv2.FILLED)
#
#         cv2.addWeighted(im,0.7,rectangle_tmp,0.3,0,im)
#         cv2.putText(im, string, text_origin,
#                     fontFace, fontScale, (0, 0, 0), thiness, lineType=-1, )
#     return im

def write_bb_black(im,dets):
    for i in range(len(dets)):
        bbox = dets[i, :4].astype(np.int32)
        im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,0), 3)
    return im

def draw_onefile():
    #files=['DSC_5756_12','DSC_6586_24','MVI_1587_VIS_00152','MVI_1619_VIS_00430']
    files=['002365','MVI_1478_VIS_00405','MVI_1474_VIS_00429','MVI_1486_VIS_00013','003426']
    root_xml='E:/fjj/SeaShips_SMD/Annotations'#'E:\paper\专利半监督船舶半自动标注\终稿\图/
    root_im='E:/fjj/SeaShips_SMD/JPEGImages'#'E:/'
    _class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))
    for file in files:
        imgpath=os.path.join(root_im,file+'.jpg')
        basename = os.path.splitext(os.path.basename(imgpath))[0]
        xmlpath=os.path.join(root_xml,basename+'.xml')

        gts,cls=annotation_onefile(xmlpath)

        class_inds=[_class_to_ind[c] for c in cls]
        im=cv2.imread(imgpath)
        # im=write_bb_black(im,gts)
        im = write_detection_batch(im,class_inds, gts)
        # cv2.imshow('a',im)
        # cv2.waitKey(2)
        cv2.imwrite(os.path.join('E:/','w'+basename+'.jpg'),im)


if __name__ == '__main__':
    path='E:/fjj/SeaShips_SMD/JPEGImages/'
    draw_onefile()

