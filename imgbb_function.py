#import os
import cv2
import numpy as np
#import colorsys

class visual(object):
    def __init__(self,CLASSES,colors):
        self.classes=CLASSES
        self.colors=colors

    def write_detection_batch(self,im, class_inds, dets):
        # inds = np.where(dets[:, -1] >= thresh)[0]
        # if len(inds) == 0:
        #     return im
        for i in range(len(dets)):
            bbox = dets[i, :4].astype(np.int32)
            # score = dets[i, -1]
            rectangle_tmp = im.copy()
            cv2.rectangle(rectangle_tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colors[class_inds[i]], 3)  # 10
            string = '%s' % (self.classes[class_inds[i]])
            fontFace = cv2.FONT_HERSHEY_COMPLEX  # cv2.FONT_HERSHEY_COMPLEX
            fontScale = 1  # 2
            thiness = 1  # 2

            text_size, baseline = cv2.getTextSize(string, fontFace, fontScale, thiness)
            text_origin = (int(bbox[0] - 1), bbox[1])  # - text_size[1]
            # cv2.copyTo(im,dst=rectangle_tmp)

            cv2.rectangle(rectangle_tmp, (text_origin[0] - 1, text_origin[1] + 1),
                          (text_origin[0] + text_size[0], text_origin[1] - text_size[1] - 1), self.colors[class_inds[i]],
                          cv2.FILLED)

            cv2.addWeighted(im, 0.7, rectangle_tmp, 0.3, 0, im)
            cv2.putText(im, string, text_origin,
                        fontFace, fontScale, (0, 0, 0), thiness, lineType=-1, )
        return im

def write_detection_batch1(im, class_inds, dets,CLASSES,colors):
    # inds = np.where(dets[:, -1] >= thresh)[0]
    # if len(inds) == 0:
    #     return im
    for i in range(len(dets)):
        bbox = dets[i, :4].astype(np.int32)
        # score = dets[i, -1]
        #rectangle_tmp = im.copy()
        #str1=CLASSES[class_inds[i]]
        string = '%s' % (CLASSES[class_inds[i]])
        fontFace = cv2.FONT_HERSHEY_COMPLEX#cv2.FONT_HERSHEY_COMPLEX
        fontScale=2#2
        thiness=2#2

        text_size,baseline = cv2.getTextSize(string, fontFace, fontScale, thiness)
        # if class_inds[i]==14:
        #     text_origin = (int(bbox[0]-1),  bbox[3]+text_size[1])#- text_size[1]
        # else:
        text_origin = (int(bbox[0] - 1), bbox[1])  # - text_size[1]
        #cv2.copyTo(im,dst=rectangle_tmp)

        im=cv2.rectangle(im, (text_origin[0] - 1, text_origin[1] + 1),
                      (text_origin[0] + text_size[0], text_origin[1] - text_size[1] - 1), colors[class_inds[i]],
                      cv2.FILLED)

        #cv2.addWeighted(im,0.6,rectangle_tmp,0.4,0,im)
        im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[class_inds[i]], 10)  # 10
        cv2.putText(im, string, text_origin,
                    fontFace, fontScale, (0, 0, 0), thiness, lineType=-1, )
    return im

def write_detection_batch(im, class_inds, dets,CLASSES,colors):
    # inds = np.where(dets[:, -1] >= thresh)[0]
    # if len(inds) == 0:
    #     return im
    for i in range(len(dets)):
        bbox = dets[i, :4].astype(np.int32)
        # score = dets[i, -1]
        rectangle_tmp = im.copy()
        #str1=CLASSES[class_inds[i]]
        string = '%s' % (CLASSES[class_inds[i]].lower())
        fontFace = cv2.FONT_HERSHEY_COMPLEX#cv2.FONT_HERSHEY_COMPLEX
        fontScale=2.#2
        thiness=2#2

        text_size,baseline = cv2.getTextSize(string, fontFace, fontScale, thiness)
        text_origin = (int(bbox[0]-1),  bbox[1])#- text_size[1]
        #cv2.copyTo(im,dst=rectangle_tmp)

        cv2.rectangle(rectangle_tmp, (text_origin[0] - 1, text_origin[1] + 1),
                      (text_origin[0] + text_size[0], text_origin[1] - text_size[1] - 1), colors[class_inds[i]],
                      cv2.FILLED)

        cv2.addWeighted(im,0.6,rectangle_tmp,0.4,0,im)
        im=cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[class_inds[i]], 3)  # 10
        cv2.putText(im, string, text_origin,
                    fontFace, fontScale, (0, 0, 0), thiness, lineType=-1, )
    return im