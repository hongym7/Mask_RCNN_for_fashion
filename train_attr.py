import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

train_df = pd.read_csv('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/train.csv')

max_clz = train_df.ClassId.max()
print('last class id : ', max_clz)

max_attr = 0
for i in train_df.AttributesIds:
    for a in str(i).split(','):
        if a != 'nan':
            a = int(a)
            if a > max_attr:
                max_attr = a

print('last attr id : ', max_attr)

clz_attr = np.zeros((max_clz+1, max_attr+1))
clz_attrid2idx = [[] for _ in range(max_clz+1)]
#print('clz_attr shape : ', clz_attr.shape)

for c, i in zip(train_df.ClassId, train_df.AttributesIds):
    for a in str(i).split(','):
        if a != 'nan':
            a = int(a)
            clz_attr[c, a] = 1
            if not a in clz_attrid2idx[c]:
                clz_attrid2idx[c].append(a)

clz_attr_num = clz_attr.sum(axis=1).astype(np.int64)
#print(clz_attr_num)
#print(clz_attrid2idx)

clz_list = list([])

for idx in range(46):
    clz_list.append(list([]))

idx = 0

for attr in train_df.AttributesIds:

    if str(train_df.AttributesIds[idx]) != 'nan':
        temp_attr_list = clz_list[train_df.ClassId[idx]]
        temp_attr_list.append(train_df.AttributesIds[idx])
        clz_list[train_df.ClassId[idx]] = list(set(temp_attr_list))

    temp_attr_list = list([])
    idx = idx + 1
