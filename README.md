# dataset processing
Metrics of dataset with **VOC** format

1.Go to  ./coco/PythonAPI

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run `python setup.py build_ext --inplace`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run `python setup.py build_ext install`

2.Go to ./utils and run `python setup.py build_ext --inplace`

3.Run main.py

-----------------------------------------------------------------------------------------

​	Please modify the following variables:

​	**In  main.py**

​	*dataset_path*: VOC dataset path.

​	*split_name*: Which mainset to evaluate.

​	**In  semi_pascal_voc.py**

​	class_name: modify as the class_names in your dataset.

-----------------------------

​	