'''
提供函数接口
write_detection：画检测结果
combined_roidb：解析VOC返回标注数据
combined_roidb_un：解析VOC返回无标注数据
'''
import numpy as np
import cv2
import colorsys
from config import CLASSES
from config import abs_path
from imdb import imdb as imdb2
from semi_factory import get_imdb
import roidb as rdl_roidb
import os
###########################################################################################
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
#########################################################################################

def get_output_dir(imdb_name, folder,net_name=None):
    """Return the directory where experimental artifacts are placed.
    If the directory des not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    if folder is None:
        folder = 'default'

    outdir = os.path.abspath(os.path.join(abs_path, folder, net_name, imdb_name))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def get_testing_roidb(imdb,unlabel=False):
    """Returns a roidb (Region of Interest database) for use in testing."""
    print('Preparing testing data...')
    rdl_roidb.prepare_roidb(imdb,unlabel=unlabel)#為每張圖片準備數據結構進行統計
    print('done')
    return imdb.roidb

def combined_roidb(imdb_names,image_set='train',dataset_path=None):
    """
    Combine multiple roidbs
    """
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name,image_set,dataset_path)#返回voc数据集
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method("gt")#设置获取注释文件方式
        print('Set proposal method: {:s}'.format("gt"))
        roidb = get_testing_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]#search database 可多个数据集以‘+’区分
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1],image_set,dataset_path)
        imdb = imdb2(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names,image_set,dataset_path)
    return imdb, roidb

def combined_roidb_un(imdb_names,image_set='train',dataset_path=None):
    """
        Combine multiple roidbs
        """
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name, image_set,dataset_path)  # 返回voc数据集
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method("unlabel")  # 设置获取注释文件方式
        print('Set proposal method: {:s}'.format("unlabel"))
        roidb = get_testing_roidb(imdb,True)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]  # search database 可多个数据集以‘+’区分
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1], image_set,dataset_path)
        imdb = imdb2(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names, image_set,dataset_path)
    return imdb, roidb
#################################################################################################################