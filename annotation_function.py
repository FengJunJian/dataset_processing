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
import pickle
import xml.etree.ElementTree as ET
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
#数据集读取
def get_output_dir(rootdir,imdb_name,folder,net_name=''):
    """Return the directory where experimental artifacts are placed.
    If the directory des not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    if folder is None:
        folder = 'default'

    outdir = os.path.abspath(os.path.join(rootdir, imdb_name,folder,net_name))
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
#检测结果保存
def detection2pkl(num_images,num_classes,bboxes,scores):
    pass
    # all_boxes = [[[] for _ in range(num_images)]
    #              for _ in range(num_classes)]
    #
    # det_boxes = np.concatenate([bboxes, scores], axis=-1)
    #
    # det_file = 'detections.pkl'
    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

def annotation_classes_file(path,class_name):
    from collections import Counter
    # 寻找数据集下各类别对应的文件(.xml)
    # path：注释文件所在目录
    # class_name: 类别名
    class_files=[]
    total_objs=0
    files = os.listdir(path)
    for filename in files:
        tree = ET.parse(os.path.join(path, filename))
        objs = tree.findall('object')
        total_objs += len(objs)
        # boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        # gt_classes = np.zeros((num_objs), dtype=np.int32)
        # overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        # seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            if(class_name==obj.find('name').text.strip()):
            #name = obj.find('name').text.strip()
                class_files.append(filename)
    print('total objects:',total_objs)
    return set(class_files)

def annotation_classes_Mainset(annotation_path,mainset):
    from collections import Counter
    # 寻找数据集下某集合的各类别与对应数目情况
    # annotation_path：注释文件所在目录
    # mainset: 数据集名
    mainset_path=os.path.join(annotation_path,'../ImageSets/Main',mainset+'.txt')
    basenames=[]
    with open(mainset_path,'r') as f:
        basenames=f.readlines()
    basenames=[basename.strip() for basename in basenames]
    class_files={}
    #class_files.setdefault('a',[]).append(1)
    class_names=[]
    total_objs=0
    #files = os.listdir(path)
    for filename in basenames:
        tree = ET.parse(os.path.join(annotation_path, filename+'.xml'))
        objs = tree.findall('object')
        total_objs += len(objs)
        for ix, obj in enumerate(objs):
            name = obj.find('name').text.strip()
            class_names.append(name)
            class_files.setdefault(name, []).append(filename)

    print('total objects:',total_objs)
    print('class_files:',class_files)
    print('len class:',len(class_files.keys()))
    return dict(Counter(class_names))

def annotation_classes_name(dataset_path):
    from collections import Counter
    # 寻找数据集注释下所有的类别名
    # path：注释文件所在目录
    class_names = []
    total_objs=0
    Annotation_path=os.path.join(dataset_path,'Annotations')
    files = os.listdir(Annotation_path)
    for filename in files:
        tree = ET.parse(os.path.join(Annotation_path, filename))
        objs = tree.findall('object')
        total_objs += len(objs)

        # boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        # gt_classes = np.zeros((num_objs), dtype=np.int32)
        # overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        # seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            name = obj.find('name').text.strip()
            class_names.append(name)
    return dict(Counter(class_names))


def annotation_meanpixel(dataset_path,mainsets):
    #from collections import Counter
    # 寻找数据集下某集合的的RGB均值
    # annotation_path：注释文件所在目录
    # mainset: 数据集名
    Basenames = []
    for mainset in mainsets:
        mainset_path = os.path.join(dataset_path, 'ImageSets/Main', mainset + '.txt')
        with open(mainset_path, 'r') as f:
            basenames = f.readlines()
        Basenames = Basenames+[basename.strip() for basename in basenames]

    JPEGpath=os.path.join(dataset_path,'JPEGImages')
    fullnames=[os.path.join(JPEGpath,basename+'.jpg') for basename in Basenames]
    total_num=len(fullnames)
    total=np.zeros(3,np.int64)
    for i,filename in enumerate(fullnames):
        img=cv2.imread(filename)
        b,g,r=cv2.split(img)
        total[0] += np.mean(b)
        total[1] += np.mean(g)
        total[2] += np.mean(r)
        if i%300==0:
            print(i)
    print(total)
    mean=np.divide(total,total_num)
    print('b,g,r',mean)
    return mean

def annotation_maxGT(dataset_path,mainsets=None):
    # 寻找数据集下某集合中，单张图片最多包含多少个GT
    # annotation_path：注释文件所在目录
    # mainset: 数据集名

    maxGT = 0
    Annotation_path = os.path.join(dataset_path, 'Annotations')
    files = os.listdir(Annotation_path)
    for filename in files:
        tree = ET.parse(os.path.join(Annotation_path, filename))
        objs = tree.findall('object')
        if len(objs)>maxGT:
            maxGT=len(objs)

    return maxGT

def annotation_onefile(xmlpath):
    """
    加载一张图片的GT
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    return (xmin,ymin,xmax,ymax)
    """
    filename = xmlpath#os.path.join(self._data_path, 'Annotations', index + '.xml')
    # print(filename)
    tree = ET.parse(filename)
    objs = tree.findall('object')

    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.int16)
    gt_classes = []

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        cls = obj.find('name').text.strip()
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes.append(cls)
        #overlaps[ix, cls] = 1.0
        #seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    #overlaps = scipy.sparse.csr_matrix(overlaps)

    return boxes,gt_classes

if __name__ == '__main__':
    # from datasets.pascal_voc import pascal_voc
    # annopath = 'E:\\fjj\\SeaShips_SMD\\Annotations'#
    # print(annotation_classes_Mainset(annopath, 'test650'))
    #E:\fjj\MarineShips2\ImageSets\Main
    datasetpath='E:\\SeaShips_SMD'#
    print(annotation_classes_name(datasetpath))
    print(annotation_maxGT(dataset_path=datasetpath))
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
