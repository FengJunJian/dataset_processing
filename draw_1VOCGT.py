import os
import cv2
import numpy as np

from annotation_function import combined_roidb
from annotation_function import write_detection

def main():
    dataset_path = 'E:/fjj/SeaShips_SMD/'
    split_name = 'draw1'
    save_path='GT'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    imdb, roidb = combined_roidb("shipdataset", split_name,dataset_path)
    for data in roidb:
        boxes=data['boxes']
        img_path=data['image']
        basename=os.path.basename(img_path)
        classes=data['gt_classes']
        img=cv2.imread(img_path)
        for j in set(classes):
            if j==7:
                continue
            indexs = np.where(classes == j)
            img=write_detection(img,j,boxes[indexs])
        cv2.imwrite(os.path.join(save_path,basename),img)


if __name__ == '__main__':
    main()