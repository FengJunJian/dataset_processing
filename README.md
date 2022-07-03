# dataset processing

[![OSCS Status](https://www.oscs1024.com/platform/badge/FengJunJian/dataset_processing.svg?size=small)](https://www.oscs1024.com/project/FengJunJian/dataset_processing?ref=badge_small)
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

two tool files:visual_function.py,annotation_function.py
annotation_function.py: mainly for annotation processing
visual_function.py:mainly for visualization such as writing boundingboxes.

