#检查对应注释是否有目标
import os
import xml.etree.ElementTree as ET
abs_path='E:\\fjj\\\SeaShips_SMD\\'
dataset_path=os.path.join(abs_path,'JPEGImages\\')
MainSet_path=os.path.join(abs_path,'ImageSets\\Main\\')
Anno_path=os.path.join(abs_path,'Annotations\\')
labels=['label0','label1','label2','label3','all','test1283']
for i in range(len(labels)):
    with open(os.path.join(MainSet_path,labels[i]+'.txt')) as f:
        lines=f.readlines()
        filenames=[line.strip() for line in lines]

    #xmls=os.listdir(path)

    strings=''
    count=0
    for xml in filenames:
        filename=os.path.join(Anno_path,xml+'.xml')
        tree = ET.parse(filename)
        size=tree.find('size')
        width=float(size.find('width').text)
        height=float(size.find('height').text)
        objs=tree.findall('object')
        flag = False
        if len(objs)>0:
            flag=True
        # for obj in objs:
        #     flag = True
        #     bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            # if float(bbox.find('xmin').text)<0:
            #     print(bbox.find('xmin').text)
            #     bbox.find('xmin').text = '0'
            #
            # if float(bbox.find('xmax').text) > width:
            #     print(bbox.find('xmax').text)
            #     bbox.find('xmax').text = str(int(width))
            #
            # if float(bbox.find('ymin').text) < 0:
            #     print(bbox.find('ymin').text)
            #     bbox.find('ymin').text = '0'
            #
            # if float(bbox.find('ymax').text)>height:
            #     # bbox.find('xmin').text='0'
            #     print(bbox.find('ymax').text)
            #     bbox.find('ymax').text = str(int(height))

        if not flag:
            print(filename)
            count+=1
        else:
            strings+=xml+'\n'
    print(count)
    with open(os.path.join(MainSet_path,labels[i]+'t.txt'),'w') as f:
        f.write(strings)

