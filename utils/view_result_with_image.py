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

import tensorflow
import keras

with open('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/label_descriptions.json', 'r') as file:
    label_desc = json.load(file)
#sample_sub_df = pd.read_csv('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/sample_submission.csv')
sample_sub_df = pd.read_csv('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/submission_score_1213.csv')
train_df = pd.read_csv('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/train.csv')


def create_mask(size):
    image_ids = train_df['ImageId'].unique()[:size]
    images_meta = []

    for image_id in image_ids:
        img = mpimg.imread(f'/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/train/{image_id}.jpg')
        images_meta.append({
            'image': img,
            'shape': img.shape,
            'encoded_pixels': train_df[train_df['ImageId'] == image_id]['EncodedPixels'],
            'class_ids': train_df[train_df['ImageId'] == image_id]['ClassId']
        })

    masks = []
    for image in images_meta:
        shape = image.get('shape')
        encoded_pixels = list(image.get('encoded_pixels'))
        class_ids = list(image.get('class_ids'))

        # Initialize numpy array with shape same as image size
        height, width = shape[:2]
        mask = np.zeros((height, width)).reshape(-1)

        # Iterate over encoded pixels and create mask
        for segment, (pixel_str, class_id) in enumerate(zip(encoded_pixels, class_ids)):
            splitted_pixels = list(map(int, pixel_str.split()))
            pixel_starts = splitted_pixels[::2]
            run_lengths = splitted_pixels[1::2]
            assert max(pixel_starts) < mask.shape[0]
            for pixel_start, run_length in zip(pixel_starts, run_lengths):
                pixel_start = int(pixel_start) - 1
                run_length = int(run_length)
                mask[pixel_start:pixel_start + run_length] = 255 - class_id * 4
        masks.append(mask.reshape((height, width),
                                  order='F'))  # https://stackoverflow.com/questions/45973722/how-does-numpy-reshape-with-order-f-work
    return masks, images_meta


def plot_segmented_images(size=12, figsize=(14, 14)):
    # First create masks from given segments
    masks, images_meta = create_mask(size)

    # Plot images in groups of 4 images
    n_groups = 4

    count = 0
    for index in range(size // 4):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        for row in ax:
            for col in row:
                col.imshow(images_meta[count]['image'])
                col.imshow(masks[count], alpha=0.75)
                col.axis('off')
                count += 1
        plt.show()
    gc.collect()


plot_segmented_images()