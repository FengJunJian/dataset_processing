'''
批量删除文件

'''
import os

root='E:/FE/'
files=os.listdir(root)
for i in range(0,len(files),4):
    os.remove(os.path.join(root,files[i]))