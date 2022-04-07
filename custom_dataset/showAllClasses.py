'''
替换数据集中的类别名
'''
import os
import xml.etree.ElementTree as ET
from collections import Counter
# replace_dict={
#     'flying bird/plane':'flying bird',
#     'vessel/ship':'vessel'
# }
#ks=replace_dict.keys()
CLASSES = ('__background__',  # 0
           'passenger ship',  # 1
           'ore carrier',  # 2
           'general cargo ship',  # 3
           'fishing boat',  # 4
           'Sail boat',  # 5
           'Kayak',  # 6
           'flying bird',  # flying bird/plane #7
           'vessel',  # vessel/ship #8
           'Buoy',  # 9
           'Ferry',  # 10
           'container ship',  # 11
           'Other',  # 12
           'Boat',  # 13
           'Speed boat',  # 14
           'bulk cargo carrier',  # 15
           )


# c=['bird', 'ship', 'ore carrier', 'buoy', 'Other', 'Fishing boat', 'fishing boat', 'Steamship', 'other', 'container ship', 'Speedboat', 'general cargo ship', 'bulk cargo carrier', 'Ordinary cargo ship', 'Boat', 'Ferry Boat', 'Sailboat', 'Kayak', 'Bulk cargo ship', 'Ferry', 'bulk cargo carrier001', 'Buoy', 'passenger ship', 'Ore carrier', 'Passenger', 'vessel', 'Sail boat', 'Speed boat', 'Container ship', 'flying bird', 'steamship', 'boat']


#path='G:/ShipDataSet/BXShipDataset/Annotations'#'Annotations'
path='E:/SeaShips_SMD/Annotations'#'Annotations'
xmls=os.listdir(path)
classnames=[]
nclassnames=[]
for xml in xmls:
    filename=os.path.join(path,xml)
    tree = ET.parse(filename)
    objs=tree.findall('object')

    replace_list=[]
    for obj in objs:
        name=obj.find('name').text#.lower().strip()
        classnames.append(name)
        if name not in CLASSES:
            nclassnames.append(name)
            print('{}:{}'.format(name,filename))
        # for k in ks:
        #     if name==k:
        #         obj.find('name').text=replace_dict[k]
        #         flag=True
        #         replace_list.append(name)
    # if flag:
    #     print(filename,set(replace_list))
    #tree.write(filename)
print('###########################################')
print(len(nclassnames),Counter(nclassnames))
print(set(nclassnames))
print('###########################################')
print(Counter(classnames))
print(set(classnames))
# f=open(filename,'r')
# lines=f.readlines()
# f.close()