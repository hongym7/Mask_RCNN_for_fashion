# Import required libraries
import os
import gc
import sys
import json
import random
from pathlib import Path

import cv2  # CV2 for image manipulation
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from imgaug import augmenters as iaa

import seaborn as sns
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold

import random

import tensorflow
import keras

with open('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/label_descriptions.json', 'r') as file:
    label_desc = json.load(file)
sample_sub_df = pd.read_csv('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/sample_submission.csv')
test_df1 = pd.read_csv('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/submission_fashion20200520T1756_683.csv')
test_df2 = pd.read_csv('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/submission_score_1213.csv')

categories_df = pd.DataFrame(label_desc.get('categories'))
attributes_df = pd.DataFrame(label_desc.get('attributes'))

cat_id_list = categories_df['id']
cat_name_list = categories_df['name']


def get_label(class_id):

    for idx in range(len(cat_id_list)):
        if cat_id_list[idx] == class_id:
            return cat_name_list[idx]

    return 'error'


def create_mask(size):
    image_ids = test_df1['ImageId'].unique()
    random.shuffle(image_ids)
    image_ids = image_ids[:size]

    images_meta = []

    for image_id in image_ids:
        img = mpimg.imread(f'/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/test3/{image_id}.jpg')
        images_meta.append({
            'image': img,
            'shape': img.shape,
            'encoded_pixels': test_df1[test_df1['ImageId'] == image_id]['EncodedPixels'],
            'encoded_pixels2': test_df2[test_df2['ImageId'] == image_id]['EncodedPixels'],
            'class_ids': test_df1[test_df1['ImageId'] == image_id]['ClassId'],
            'class_ids2': test_df2[test_df2['ImageId'] == image_id]['ClassId'],
            'attributes_ids': test_df1[test_df1['ImageId'] == image_id]['AttributesIds'],
            'attributes_ids2': test_df2[test_df2['ImageId'] == image_id]['AttributesIds']
        })

        print('image id ' + image_id)

    masks = []
    masks2 = []
    class_id_list = []
    class_id_list2 = []
    attr_id_list = []
    attr_id_list2 = []

    for image in images_meta:
        shape = image.get('shape')
        encoded_pixels = list(image.get('encoded_pixels'))
        class_ids = list(image.get('class_ids'))
        attributes_ids = list(image.get('attributes_ids'))

        encoded_pixels2 = list(image.get('encoded_pixels2'))
        class_ids2 = list(image.get('class_ids2'))
        attributes_ids2 = list(image.get('attributes_ids2'))

        # Initialize numpy array with shape same as image size
        height, width = shape[:2]

        # Iterate over encoded pixels and create mask
        for segment, (pixel_str, class_id, attributes_id) in enumerate(zip(encoded_pixels, class_ids, attributes_ids)):
            splitted_pixels = list(map(int, pixel_str.split()))
            pixel_starts = splitted_pixels[::2]
            run_lengths = splitted_pixels[1::2]
            mask = np.zeros((height, width)).reshape(-1)
            assert max(pixel_starts) < mask.shape[0]
            for pixel_start, run_length in zip(pixel_starts, run_lengths):
                pixel_start = int(pixel_start) - 1
                run_length = int(run_length)
                mask[pixel_start:pixel_start + run_length] = 255 - class_id * 4
            class_id_list.append(class_id)
            attr_id_list.append(attributes_id)
            masks.append(mask.reshape((height, width), order='F'))


        # Iterate over encoded pixels and create mask
        for segment, (pixel_str, class_id, attributes_id) in enumerate(zip(encoded_pixels2, class_ids2, attributes_ids2)):
            splitted_pixels = list(map(int, pixel_str.split()))
            pixel_starts = splitted_pixels[::2]
            run_lengths = splitted_pixels[1::2]
            mask = np.zeros((height, width)).reshape(-1)
            assert max(pixel_starts) < mask.shape[0]
            for pixel_start, run_length in zip(pixel_starts, run_lengths):
                pixel_start = int(pixel_start) - 1
                run_length = int(run_length)
                mask[pixel_start:pixel_start + run_length] = 255 - class_id * 4
            class_id_list2.append(class_id)
            attr_id_list2.append(attributes_id)
            masks2.append(mask.reshape((height, width), order='F'))

    return masks, images_meta, class_id_list, attr_id_list, masks2, class_id_list2, attr_id_list2


def plot_segmented_images(size=1, figsize=(10, 10)):
    # First create masks from given segments

    masks, images_meta, class_id_list, attr_id_list, masks2, class_id_list2, attr_id_list2 = create_mask(size)

    fig, axes = plt.subplots(nrows=2, ncols=max([len(masks)+1, len(masks2)+1]), figsize=figsize)

    axes[0][0].imshow(images_meta[0]['image'])
    axes[1][0].imshow(images_meta[0]['image'])

    for idx in range(len(masks)):
        axes[0][idx+1].imshow(images_meta[0]['image'])
        axes[0][idx+1].imshow(masks[idx], alpha=0.75)
        axes[0][idx+1].set_title(str(class_id_list[idx]) + ' ' + get_label(class_id_list[idx]) + ' \n ' + str(attr_id_list[idx]))
        axes[0][idx+1].axis('off')

    for idx in range(len(masks2)):
        axes[1][idx+1].imshow(images_meta[0]['image'])
        axes[1][idx+1].imshow(masks2[idx], alpha=0.75)
        axes[1][idx+1].set_title(str(class_id_list2[idx]) + ' ' + get_label(class_id_list2[idx]) + ' \n ' + str(attr_id_list2[idx]))
        axes[1][idx+1].axis('off')

    plt.show()


plot_segmented_images()