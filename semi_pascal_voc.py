# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
'''
解析voc
自定义函数：
annotation_classes_file：某类别对应的所有样本
annotation_classes_Mainset：寻找数据集下某集合的各类别与对应数目情况
annotation_classes_name：寻找数据集注释下所有的类别名
annotation_meanpixel：寻找数据集下某集合的的RGB均值
annotation_maxGT：寻找数据集下，单张图片最多包含多少个GT

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import subprocess
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse

# from lib.config import config as cfg
from imdb import imdb
from eval.voc_eval import voc_eval
import matplotlib.pyplot as plt
from config import CLASSES

class_name=CLASSES

class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path):
        imdb.__init__(self, image_set)  # imdb_name
        self._year = year
        self._image_set = image_set
        self._devkit_path = devkit_path#self._get_default_path() if devkit_path is None else devkit_path
        self._data_path = self._devkit_path#cfg.FLAGS2["custom_data_dir"]
        # os.path.join(self._devkit_path, 'VOC' + self._year)

        self._classes = class_name  # class_name#
        # ('__background__',  # always index 0
        #              'boat')

        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()  # 加载文本索引
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': True,  # 不判断diff
                       'matlab_eval': False,
                       'rpn_file': None}

        assert os.path.exists(self._devkit_path), \
            'devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, image_name):
        """
        Construct an image path from the image's "index" identifier.
        """
        # image_path = os.path.join(self._data_path, 'JPEGImages', index + self._image_ext)
        image_path = os.path.join(self._data_path,'JPEGImages',image_name + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')

        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        # path=os.path.join(cfg.FLAGS2["data_dir"], 'VOCdevkit' + self._year)
        path = None#cfg.FLAGS2["semi_labeled_dir"]
        if path is None:
            raise ValueError('error:path is None')
        return path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def unlabel_roidb(self):
        """
        Return the database of unlabel.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        gt_roidb=[]
        for i in range(len(self.image_index)):
            gt_dict = {'boxes': np.array([[-1, -1, -1, -1]], dtype=np.int16),
                       'gt_classes': np.array([-1]),
                       'gt_overlaps': -1,
                       'flipped': False,
                       'seg_areas': -1}
            gt_roidb = gt_roidb.append(gt_dict)
        # cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         try:
        #             roidb = pickle.load(fid)
        #         except:
        #             roidb = pickle.load(fid, encoding='bytes')
        #     print('{} gt roidb loaded from {}'.format(self.name, cache_file))
        #     return roidb
        #
        # gt_roidb = [self._load_pascal_annotation(index)
        #             for index in self.image_index]
        # with open(cache_file, 'wb') as fid:
        #     pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        # print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        # print(filename)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [obj for obj in objs if int(obj.find('Difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.int16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self, pathdir):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        # filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filename = '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(pathdir)
        if os.path.exists(path) is False:
            os.mkdir(path)
        path = os.path.join(path, filename)
        return path

    def _write_voc_results_file(self, all_boxes, pathdir):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template(pathdir).format(cls.replace(' ', '_'))
            if os.path.exists(filename):
                continue
            with open(filename, 'w') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        # annopath=cfg.FLAGS2["custom_annotation_dir"]
        annopath = os.path.join(self._devkit_path, 'Annotations/' + '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'ImageSets',
            'Main',
            self._image_set + '.txt')

        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False  # True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template(output_dir).format(cls.replace(' ', '_'))

            # rec, prec, ap = voc_eval.voc_eval_save(
            #     filename, annopath, imagesetfile, cls, cachedir, output_dir,ovthresh=0.5,
            #     use_07_metric=use_07_metric)#
            cachedir = os.path.join(self._devkit_path, 'annotations_cache')
            rec, prec, ap, precision, recall = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)  # 评估算法 PR曲线
            aps += [ap]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            with open(os.path.join(output_dir, cls.replace(' ', '_') + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap, 'precision': precision, 'recall': recall}, f)
            with open(os.path.join(output_dir, cls.replace(' ','_') + '_pr.txt'), 'w') as f:
                for i in range(len(rec)):
                    f.write(str(prec[i]) + '\t' + str(rec[i]) + '\n')
            plt.plot(rec, prec)
            plt.xlim(0, 1)
            # 设置y轴的取值范围为：-1到3
            plt.ylim(0, 1.05)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title('PR Curve')
            plt.savefig(os.path.join(output_dir, cls.replace(' ','_') + '_pr.png'))
            # plt.show()
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        #print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')
        return aps

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join('lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format('matlab')
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):  # 评估性能
        self._write_voc_results_file(all_boxes, output_dir)  # 为每类写box文件
        aps=self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template(output_dir).format(cls.replace(' ', '_'))
                #filename = self._get_voc_results_file_template(output_dir).format(cls)
                os.remove(filename)
        return aps
    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True



if __name__ == '__main__':
    # from datasets.pascal_voc import pascal_voc
    # annopath = 'E:\\fjj\\SeaShips_SMD\\Annotations'#
    # print(annotation_classes_Mainset(annopath, 'test650'))

    datasetpath='E:\\fjj\\SeaShips_SMD'#
    #print(annotation_maxGT(datasetpath))
    #annotation_meanpixel(datasetpath,['all'])
    #a=annotation_classes_name(datasetpath)
    #annotation_meanpixel(datasetpath,['all'])
    #print(a)
    '''
    class_names = annotation_classes_name(path)
    print(class_names)
    with open('H:\\fjj\\SeaShips_SMD\\class.txt', 'w') as f:
        for name in class_names.keys():
            f.writelines(name + '\n')
      '''

