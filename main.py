import tensorflow as tf
import cv2
import numpy as np
from imdb import imdb as imdb2
from semi_factory import get_imdb
import roidb as rdl_roidb
import os

abs_path='.'

def get_output_dir(imdb_name, weights_filename,net_name=None):
    """Return the directory where experimental artifacts are placed.
    If the directory des not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    if weights_filename is None:
        weights_filename = 'default'

    outdir = os.path.abspath(os.path.join(abs_path, weights_filename, net_name, imdb_name))
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

def main():
    #model_name = cfg.FLAGS.network_semi
    #self.num = cfg.FLAGS.test_iter  # 10000
    dataset_path='E:/fjj/SeaShips_SMD/'
    model_name='semi_faster'
    log_path='test'
    model_path = 'E:/fjj/semi_ship/vgg16_semi/label0_unlabel0/vgg16_semi_faster_rcnn_iter_192000.ckpt'

    imdb, roidb = combined_roidb("shipdataset", 'test650',dataset_path)  # 测试数据集test1300,test650
    # self.imdb, self.roidb=combined_roidb_un("unlabel_dataset",'unlabel_train',cfg.FLAGS2['semi_unlabeled_dir'])
    output_dir = get_output_dir(imdb.name, log_path, model_name)  # 创建文件'wideresnet'
    #trained_model = cfg.FLAGS.label_sets + '_' + cfg.FLAGS.unlabel_sets
    tfmodel='E:/fjj/semi_ship/vgg16_semi/label0_unlabel0/vgg16_semi_faster_rcnn_iter_192000.ckpt'
    tfmeta='E:/fjj/semi_ship/vgg16_semi/label0_unlabel0/vgg16_semi_faster_rcnn_iter_192000.ckpt.meta'
    sess=tf.Session()
    saver = tf.train.import_meta_graph(tfmeta)
    saver.restore(sess, tfmodel)



if __name__ == '__main__':
    main()