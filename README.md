# dataset processing
##Metrics of dataset with **VOC** format

1.Go to  ./coco/PythonAPI

Run `python setup.py build_ext --inplace`

Run `python setup.py build_ext install`

2.Go to ./utils and run `python setup.py build_ext --inplace`

3.Run evalution_detection_pkl.py

-----------------------------------------------------------------------------------------

Please modify the following variables:

In  evalution_detection_pkl.py

dataset_path: VOC dataset path.

split_name: Which mainset to evaluate.

In  semi_pascal_voc.py

class_name: modify as the class_names in your dataset.

--------------------------------------
In rename_classname.py ...

