'''
替换数据集中的类别名
'''
import os
import xml.etree.ElementTree as ET
# CLASSES = ('__background__',  # 0
#            'passenger ship',  # 1
#            'ore carrier',  # 2
#            'general cargo ship',  # 3
#            'fishing boat',  # 4
#            'Sail boat',  # 5
#            'Kayak',  # 6
#            'flying bird',  # flying bird/plane #7
#            'vessel',  # vessel/ship #8
#            'Buoy',  # 9
#            'Ferry',  # 10
#            'container ship',  # 11
#            'Other',  # 12
#            'Boat',  # 13
#            'Speed boat',  # 14
#            'bulk cargo carrier',  # 15
#            )
replace_dict={
    'Fishing boat':'fishing boat',
    'Passenger':'passenger ship',
    'Ferry Boat':'Ferry',
    'buoy':'Buoy',
    'Sailboat':'Sail boat',
    'Bulk cargo ship':'bulk cargo carrier',
    'bird':'flying bird',
    'Ore carrier':'ore carrier',
    'other':'Other',
    'bulk cargo carrier001':'bulk cargo carrier',
    'Container ship':'container ship',
    'bulk cargo ship':'bulk cargo carrier',
    'Speedboat':'Speed boat',
    'boat':'Boat',
    'Ordinary cargo ship':'general cargo ship',
    'ship':'Boat',
    'Steamship':'vessel',
    'steamship':'vessel'

    ## 'vessel/ship':'vessel'
}
ks=replace_dict.keys()
path='G:/ShipDataSet/BXShipDataset/Annotations'#'Annotations'
#path='Annotations'
xmls=os.listdir(path)

for xml in xmls:
    filename=os.path.join(path,xml)
    tree = ET.parse(filename)
    objs=tree.findall('object')
    flag=False
    replace_list=[]
    for obj in objs:
        name=obj.find('name').text.strip()#.lower()
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