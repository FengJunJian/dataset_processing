import os
import cv2
import numpy as np
import colorsys
from PIL import Image,ImageFont,ImageDraw

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
        fontScale=1.#2
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

def write_detection_PIL(im, class_inds,dets,CLASSES,colors,thiness=2,GT_color=None):
    font = ImageFont.truetype('C:/Windows/Fonts/simhei.ttf', 30)#
    for i in range(len(dets)):
        rectangle_tmp = im.copy()
        bbox = dets[i, :4].astype(np.int32)
        class_ind =class_inds[i] #int(dets[i, 4])
        if class_ind==7:
            continue

        if GT_color:
            color=GT_color
        else:
            color=colors[class_ind]

        string = CLASSES[class_ind]

        _,_,text_width,text_height=font.getbbox(string)
        #text_size, baseline = cv2.getTextSize(string, fontFace, fontScale, thiness)
        text_origin = (bbox[0]-1, bbox[1]-text_height)  #text_height - text_size[1]
    ###########################################putText
        cv2.rectangle(rectangle_tmp, (text_origin[0] - 1, text_origin[1] - 1),
                           (text_origin[0] + text_width + 1, text_origin[1] + text_height + 1),
                           color, cv2.FILLED)
        cv2.addWeighted(im, 0.7, rectangle_tmp, 0.3, 0, im)
        im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thiness)
        img = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.text(text_origin, string, font=font, fill=(0, 0, 0))
        im=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        # img.show()
        # im = cv2.putText(im, string, text_origin,
        #                  fontFace, fontScale, (0, 0, 0), thiness,lineType=-1)
    return im

def save_roi_batch(imname, bboxes, cls,save_root):
    basename=os.path.splitext(os.path.basename(imname))[0]
    im=cv2.imread(imname)
    H,W,C=im.shape
    for i in range(len(bboxes)):
        xmin = max(bboxes[i][0],0)
        ymin = max(bboxes[i][1],0)
        xmax = min(bboxes[i][2]+1,W)
        ymax = min(bboxes[i][3]+1,H)
        c=cls[i]
        roi = im[ymin:ymax, xmin:xmax]
        save_name = os.path.join(save_root, c.replace(' ','_'), '%s_%d.jpg' % (basename, i))
        try:
            cv2.imwrite(save_name, roi)
            print('%s:%d bboxes'%(basename,len(bboxes)))
        except:
            print('##########\nError:%s:%d bboxes\n##########'%(basename,len(bboxes)))
            continue
    return

def crop_bb_transform(bb_labels,ROIrectangle,im_roi_ratio=1.0):
    '''
    裁剪缩放图片，以及其GT的变换
    bb_labels:(xmin,ymin,xmax,ymax,label)
    ROIrectangle: 裁剪框
    im_roi_ratio: 缩放比例
    '''
    boxes = bb_labels.copy()

    boxes[:, 0:4:2] = np.maximum(np.minimum(boxes[:, 0:4:2], ROIrectangle[2]), ROIrectangle[0])
    boxes[:, 1:4:2] = np.maximum(np.minimum(boxes[:, 1:4:2], ROIrectangle[3]), ROIrectangle[1])

    inds = np.where(np.bitwise_not(np.bitwise_or((boxes[:, 2] - boxes[:, 0]) == 0, (boxes[:, 3] - boxes[:, 1]) == 0)))[
        0]
    boxes_crop = boxes[inds].astype(np.float32)
    boxes_crop[:, 0:4:2] -= ROIrectangle[0]
    boxes_crop[:, 1:4:2] -= ROIrectangle[1]
    boxes_crop[:, 0:4] *= im_roi_ratio
    return boxes_crop
