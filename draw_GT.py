import os
import cv2
import numpy as np
import colorsys
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

def write_detection(im, class_ind, dets):
    # inds = np.where(dets[:, -1] >= thresh)[0]
    # if len(inds) == 0:
    #     return im
    for i in range(len(dets)):
        bbox = dets[i, :4].astype(np.int32)
        # score = dets[i, -1]
        im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[class_ind], 10)

        string = '%s' % (CLASSES[class_ind])
        fontFace = cv2.FONT_HERSHEY_COMPLEX
        fontScale = 2
        thiness = 2

        text_size, baseline = cv2.getTextSize(string, fontFace, fontScale, thiness)
        text_origin = (bbox[0], bbox[1])  # - text_size[1]

        im = cv2.rectangle(im, (text_origin[0] - 2, text_origin[1] + 1),
                           (text_origin[0] + text_size[0] + 1, text_origin[1] - text_size[1] - 2),
                           colors[class_ind], cv2.FILLED)
        im = cv2.putText(im, '%s' % (CLASSES[class_ind]), text_origin,
                         fontFace, fontScale, (0, 0, 0), thiness)
    return im

if __name__ == '__main__':
    path='E:/fjj/SeaShips_SMD/JPEGImages/'

