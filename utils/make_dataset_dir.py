import os

root_dir='Dataset'
Annotation_path=os.path.join(root_dir,'Annotations')
ImageSets_path=os.path.join(root_dir,'ImageSets')
Main_path=os.path.join(ImageSets_path,'Main')
JPEGImages_path=os.path.join(root_dir,'JPEGImages')

# if not os.path.exists(root_dir):
#     os.mkdir(root_dir)
if not os.path.exists(Main_path):
    os.makedirs(Main_path)

if not os.path.exists(Annotation_path):
    os.makedirs(Annotation_path)

if not os.path.exists(JPEGImages_path):
    os.makedirs(JPEGImages_path)
