'''
根据bbox的pkl来评估ap
'''
import numpy as np

import os
import pickle

from tool_function import combined_roidb
from tool_function import get_output_dir

def main():

    dataset_path='E:/fjj/SeaShips_SMD/'
    test_num='1300'
    split_name='test1300'
    model_name='yolo'
    log_path=model_name+'_test_0'
    file_root = '../data_processing_cache/'
    imdb, roidb = combined_roidb("shipdataset", split_name,dataset_path)  # 测试数据集test1300,test650
    # unimdb, unroidb=combined_roidb_un("unlabel_dataset",'unlabel_train','semi_unlabeled_dir')
    output_dir = get_output_dir(file_root, log_path, imdb.name,model_name)  # 创建文件

    all_boxes = [[[] for _ in range(imdb.num_images)]
                 for _ in range(imdb.num_classes)]
    '''
    all_boxes:shape (num_images,num_classes)
    each boxes is stored as np.array([xmin,ymin,xmax,ymax,score])
    that is the the second image and 1-index class  
    all_boxes[1][2]=boxes
    boxes:shape (batch,5) :np.array([[xmin_1,ymin_1,xmax_1,ymax_1,score_1],......,[xmin_n,ymin_n,xmax_n,ymax_n,score_n]])
    '''

    pkl_file=file_root+model_name+'_detections{}{}.pkl'.format(test_num,'')
    pkl_file=file_root+'yolo_detections_1283_0.pkl'
    with open(pkl_file, 'rb') as f:
        all_boxes = pickle.load(f)
    print('Evaluating detections')
    aps = imdb.evaluate_detections(all_boxes, output_dir)
    aps1 = [v for i, v in enumerate(aps) if i != 6]
    with open(os.path.join(output_dir, 'ap_' + '.txt'), 'w') as f:
        f.write(str(aps) + '\n')
        f.write('mean AP:' + str(np.mean(aps)) + '\n')
        f.write('mean AP without flying bird:' + str(np.mean(aps1)) + '\n')
    print(np.mean(aps))
    print(np.mean(aps1))

if __name__ == '__main__':
    main()