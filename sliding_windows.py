# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
#
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from utils.bbox_transform import bbox_transform
import os
from annotation_function import annotation_onefile
from glob import glob

def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    # assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1#锚点[xmin,ymin,xmax,ymax]
    ratio_anchors = _ratio_enum(base_anchor, ratios)#比例
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)#尺度
                         for i in range(ratio_anchors.shape[0])])
    return anchors

def generate_anchors_custom(anchors_WH,base_size=16):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1#锚点[xmin,ymin,xmax,ymax]
    w, h, x_ctr, y_ctr = _whctrs(base_anchor)
    Ws=anchors_WH[:,0,np.newaxis]
    Hs=anchors_WH[:,1,np.newaxis]

    # ws = ws[:, np.newaxis]  # 增加一个维度
    # hs = hs[:, np.newaxis]
    anchors1 = np.hstack((x_ctr - 0.5 * (Ws - 1),
                         y_ctr - 0.5 * (Hs - 1),
                         x_ctr + 0.5 * (Ws - 1),
                         y_ctr + 0.5 * (Hs - 1)))
    # anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
    #                      y_ctr - 0.5 * (hs - 1),
    #                      x_ctr + 0.5 * (ws - 1),
    #                      y_ctr + 0.5 * (hs - 1)))

    #ratio_anchors = _ratio_enum(base_anchor, ratios)
    # anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
    #                      for i in range(ratio_anchors.shape[0])])
    return anchors1


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    #返回锚框的宽，高，中心点x，y
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
#返回多个锚框
    ws = ws[:, np.newaxis]#增加一个维度
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)#转为中心坐标
    size = w * h#方框形式
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)#返回[xmin,ymin,xmax,ymax]
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2),base_size=16):#ratios:w/h
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'
      generate anchors in a sliding way
    """
    # anchors:(xmin,ymin,xmax,ymax)
    anchors = generate_anchors(base_size=base_size,ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]#9
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]#
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))#broadcast
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])

    return anchors, length

# def generate_anchors_pre_tf(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
#     shift_x = tf.range(width) * feat_stride  # width
#     shift_y = tf.range(height) * feat_stride  # height
#     shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
#     sx = tf.reshape(shift_x, shape=(-1,))
#     sy = tf.reshape(shift_y, shape=(-1,))
#     shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
#     K = tf.multiply(width, height)
#     shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))
#
#     anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
#     A = anchors.shape[0]
#     anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)
#
#     length = K * A
#     anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))
#
#     return tf.cast(anchors_tf, dtype=tf.float32), length

def generate_anchors_pre_custom(height, width, feat_stride, base_size=16,anchors_WH=np.array([[ 27,  31],
       [ 41,  54],
       [ 21,  59],
       [ 73,  96],
       [ 40, 126],
       [142, 145],
       [ 56, 217],
       [121, 448],
       [210, 815]])):#ratios:w/h
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'
      generate anchors in a sliding way
    """
    #anchors_WH:(width,height)
    #anchors:(xmin,ymin,xmax,ymax)
    anchors = generate_anchors_custom(anchors_WH,base_size=base_size)
    # anchors = generate_anchors(base_size=base_size, ratios=np.array(anchor_ratios),
    #                                   scales=np.array(anchor_scales))
    A = anchors.shape[0]
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])

    return anchors, length

def bbox_overlaps(boxes,query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    # cdef DTYPE_t iw, ih, box_area
    # cdef DTYPE_t ua
    # cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

if __name__ == '__main__':
    import time
    import cv2
    import matplotlib.pyplot as plt
    # img=cv2.imread('../../0004.jpg')
    # height,width,feat=img.shape[0],img.shape[1],16
    #height, width, feat = 36, 63, 16
    height, width, feat =int(576/16), int(704/16), 16#int(576/4), int(704/4), 4 #int(448/4), int(448/4), 4

    MarineShip_anchors = [[32, 24],
                          [69, 39],
                          [141, 68],
                          [297, 139],
                          [689, 302], ]

    impath='E:\paper\data/*.jpg'#视频_(172)1.jpg
    xmlpath='E:\paper\data/*.xml'
    ims=glob(impath)
    xmls=glob(xmlpath)
    Ppath = 'E:/shipz/P'
    Npath = 'E:/shipz/N'
    if not os.path.exists(Ppath):
        os.makedirs(Ppath)
    if not os.path.exists(Npath):
        os.makedirs(Npath)
    f = open('E:/shipz/gt.txt', 'w')
    for i,imp in enumerate(ims):
        im = cv2.imdecode(np.fromfile(imp, dtype=np.uint8), -1)
        basename=os.path.splitext(os.path.basename(imp))[0]
        im_info=im.shape
        gt,classes=annotation_onefile(xmls[i])
        #gt=np.array([[133,341,180,375],[279,336,327,366],[318,334,526,457],[541,347,677,421]])
        #classes=['fishing ship','fishing ship','fishing ship','fishing ship']
        t = time.time()
        #a,length = generate_anchors_pre(height,width,feat)
        a1,length1=generate_anchors_pre_custom(height,width,feat,anchors_WH=np.array(MarineShip_anchors))
        _allowed_border=0.0
        inds_inside = np.where(
            (a1[:, 0] >= -_allowed_border) &
            (a1[:, 1] >= -_allowed_border) &
            (a1[:, 2] < im_info[1] + _allowed_border) &  # width
            (a1[:, 3] < im_info[0] + _allowed_border)  # height
        )[0]

        # keep only inside anchors
        anchors = a1[inds_inside, :]
        overlaps=bbox_overlaps(anchors,gt)
        argmax_overlaps = overlaps.argmax(axis=1)  # 每个anchor与哪个GT交叠最大
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]  # 每个anchor与gt的最大交叠比

        gt_argmax_overlaps = overlaps.argmax(axis=0)  # 每个gt与哪个anchor交叠最大
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]  ##每个gt与anchors的最大交叠比
        gt_argmax_overlaps,gt_inds = np.where(overlaps == gt_max_overlaps)  # 具备最大的交叠比的anchors
        regression_targets=_compute_targets(anchors, gt[argmax_overlaps,:])

        for i in range(len(gt_argmax_overlaps)):
            bb=anchors[gt_argmax_overlaps[i]].astype(np.int)
            roi=im[bb[1]:bb[3],bb[0]:bb[2]]
            saveName=basename + '_%d.jpg' % (i)
            #cv2.imwrite(os.path.join(Ppath,saveName),roi)
            cv2.imencode('.jpg',roi)[1].tofile(os.path.join(Ppath,saveName))
            roi_c=classes[gt_inds[i]]
            roi_r=regression_targets[gt_argmax_overlaps[i]]
            f.write('%s,%s,%f,%f,%f,%f\n'%(saveName,roi_c,roi_r[0],roi_r[1],roi_r[2],roi_r[3]))
        inds=np.arange(anchors.shape[0])
        #Nset=np.unique(np.where(np.bitwise_and(overlaps>0.3 , overlaps != gt_max_overlaps))[0])
        inds=np.setdiff1d(inds,np.unique(np.where(overlaps>0.2)[0]))
        for i, Ni in enumerate(np.setdiff1d(inds,gt_argmax_overlaps)):
            bb = anchors[Ni].astype(np.int)
            roi = im[bb[1]:bb[3], bb[0]:bb[2]]
            saveName = basename + '_%d.jpg' % (i)
            # cv2.imwrite(os.path.join(Ppath,saveName),roi)
            cv2.imencode('.jpg', roi)[1].tofile(os.path.join(Npath, saveName))

        print(time.time() - t)
    f.close()
    #print(a)
    # for rect in a:
    #     img=cv2.rectangle(img,a,)


    #from IPython import embed
    #embed()