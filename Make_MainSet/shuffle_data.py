import numpy as np
import os

dataset_path='ImageSets\\Main'
with open(os.path.join(dataset_path,'unlabel.txt'),'r') as f:
    image_index = [x for x in f.readlines()]
np.random.seed(1)
np.random.shuffle(image_index)
image10000=[]
image5000=[]
image2500=[]
image1=[]
image1.append(image_index[0])
for id,i in enumerate(image_index):
    if len(image5000)<5000:
        image5000.append(i)
    if len(image2500)<2500:
        image2500.append(i)
    if len(image10000) < 10000:
        image10000.append(i)
    else:
        break
#image_index_temp=image_index[1:-1:2]
with open(os.path.join(dataset_path,'semi-unlabel1.txt'),'w') as f:
    for img in image1:
        f.write(img)
with open(os.path.join(dataset_path,'semi-unlabel2500.txt'),'w') as f:
    for img in image2500:
        f.write(img)
with open(os.path.join(dataset_path,'semi-unlabel5000.txt'),'w') as f:
    for img in image5000:
        f.write(img)
with open(os.path.join(dataset_path,'semi-unlabel10000.txt'),'w') as f:
    for img in image10000:
        f.write(img)