'''
替换数据集中的类别名
'''
import os
import xml.etree.ElementTree as ET

replace_dict={
    'flying bird/plane':'flying bird',
    'vessel/ship':'vessel'
}
ks=replace_dict.keys()
path='Annotations'
xmls=os.listdir(path)

for xml in xmls:
    filename=os.path.join(path,xml)
    tree = ET.parse(filename)
    objs=tree.findall('object')
    flag=False
    replace_list=[]
    for obj in objs:
        name=obj.find('name').text.lower().strip()
        for k in ks:
            if name==k:
                obj.find('name').text=replace_dict[k]
                flag=True
                replace_list.append(name)
    if flag:
        print(filename,set(replace_list))
    tree.write(filename)
#
# f=open(filename,'r')
# lines=f.readlines()
# f.close()