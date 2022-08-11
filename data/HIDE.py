import os
from shutil import copyfile

with open('/home/liqiang/桌面/mdata/HIDE_dataset/test.txt') as f:
    testdata=[i.strip() for i in f.readlines()]
print(testdata)

with open('/home/liqiang/桌面/mdata/HIDE_dataset/train.txt') as f:
    traindata=[i.strip() for i in f.readlines()]
print(traindata)

GT=os.listdir('/home/liqiang/桌面/mdata/HIDE_dataset/GT/')
print(GT)

count=0
for i in GT:
    if i in traindata:
        count+=1
        copyfile('/home/liqiang/桌面/mdata/HIDE_dataset/GT/'+i,'/home/liqiang/桌面/mdata/实验/dataset/HIDE/train/sharp/'+i)
    elif i in testdata:
        count += 1
        copyfile('/home/liqiang/桌面/mdata/HIDE_dataset/GT/' + i,
                 '/home/liqiang/桌面/mdata/实验/dataset/HIDE/test/sharp/' + i)
    else:
        print(i)

print(count)