import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import json
from pathlib import Path


IMAGE_SIZE = 512
DATA_DIR = Path('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7')

with open('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/label_descriptions.json', 'r') as file:
    label_desc = json.load(file)
train_df = pd.read_csv('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/train.csv')

print(f'Shape of training dataset: {train_df.shape}')

categories_df = pd.DataFrame(label_desc['categories'])
attributes_df = pd.DataFrame(label_desc['attributes'])

category_map, attribute_map = {}, {}
for cat in label_desc.get('categories'):
    category_map[cat.get('id')] = cat.get('name')
for attr in label_desc.get('attributes'):
    attribute_map[attr.get('id')] = attr.get('name')

train_df = train_df.drop('AttributesIds', axis=1)

image_df = train_df.groupby('ImageId')['EncodedPixels', 'ClassId'].agg(lambda x: list(x))
size_df = train_df.groupby('ImageId')['Height', 'Width'].mean()
input_df = image_df.join(size_df, on='ImageId')


def crop_mask(row_data):

    image_id = row_data.name,
    path = str(DATA_DIR / 'train' / row.name) + '.jpg',
    labels = row_data['ClassId'],
    annotations = row_data['EncodedPixels'],
    height = row_data['Height'], width = row_data['Width']

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(annotations)), dtype=np.uint8)





for i, row in input_df.iterrows():
    crop_mask(row)

