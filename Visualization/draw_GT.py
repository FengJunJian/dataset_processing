import os
import cv2
import numpy as np
import colorsys
from custom_dataset.imgbb_function import write_detection_batch,write_detection_PIL

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
CLASSES=('__background__',#0
        'fishing ship',#1
        'cargo ship',#2
        'Other',#3
        'Buoy',#4
)
CLASSES_C=('__background__',#0
        'target',#1
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

def draw_files(root_xml,root_im):
    #files=['DSC_5756_12','DSC_6586_24','MVI_1587_VIS_00152','MVI_1619_VIS_00430']
    files=['002365','MVI_1478_VIS_00405','MVI_1474_VIS_00429','MVI_1486_VIS_00013','003426']
    #files = ['MVI_1478_VIS_00405', ]
    # root_xml='E:/fjj/SeaShips_SMD/Annotations'#'
    # root_im='E:/fjj/SeaShips_SMD/JPEGImages'#'E:/'
    _class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))
    for file in files:
        imgpath=os.path.join(root_im,file+'.jpg')
        basename = os.path.splitext(os.path.basename(imgpath))[0]
        xmlpath=os.path.join(root_xml,basename+'.xml')

        gts,cls=annotation_onefile(xmlpath)

        class_inds=[_class_to_ind[c] for c in cls]
        im=cv2.imread(imgpath)
        # im=write_bb_black(im,gts)
        im = write_detection_batch(im,class_inds, gts,CLASSES,colors)
        # cv2.imshow('a',im)
        # cv2.waitKey(2)
        cv2.imencode('.jpg',im)[1].tofile(os.path.join('E:/paper/半监督船舶检测en/复杂场景图1/',basename+'.jpg'))
        #cv2.imwrite(os.path.join('E:/paper/半监督船舶检测en/复杂场景图1/','w'+basename+'.jpg'),im)


def draw_onefile(root_xml,root_im):
    #files=['DSC_5756_12','DSC_6586_24','MVI_1587_VIS_00152','MVI_1619_VIS_00430']
    #files=['002365','MVI_1478_VIS_00405','MVI_1474_VIS_00429','MVI_1486_VIS_00013','003426']
    #files = ['MVI_1478_VIS_00405', ]
    # root_xml='E:/fjj/SeaShips_SMD/Annotations'#'
    # root_im='E:/fjj/SeaShips_SMD/JPEGImages'#'E:/'
    _class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))

    basename = os.path.splitext(os.path.basename(root_im))[0]

    gts,cls=annotation_onefile(root_xml)

    class_inds=[_class_to_ind[c] for c in cls]

    im = cv2.imdecode(np.fromfile(root_im, dtype=np.uint8), -1)


    # im=write_bb_black(im,gts)
    im=write_detection_PIL(im,class_inds,gts,CLASSES_C,colors)
    #im = write_detection_batch(im,class_inds, gts,CLASSES,colors)
    # cv2.imshow('a',im)
    # cv2.waitKey(2)
    return im


if __name__ == '__main__':
    path='E:\\'
    root_xml=['1.xml','2.xml','3.xml']
    root_im=['1.jpg','2.jpg','3.jpg']
    for i in range(len(root_xml)):
        im=draw_onefile(os.path.join(path,root_xml[i]), os.path.join(path,root_im[i]))
        output='E:\\output'
        basename=os.path.basename(root_im[i])
        cv2.imencode('.jpg', im)[1].tofile(os.path.join(output, basename))
        # cv2.imshow('a',im)
        # cv2.waitKey()
    # draw_onefile()

