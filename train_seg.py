# Import required libraries
import os
import gc
import sys
import json
import random
from pathlib import Path

import matplotlib.pyplot as pp


import cv2 # CV2 for image manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from utils.warmup_lr import WarmUpCosineDecayScheduler

from tqdm import tqdm

from imgaug import augmenters as iaa

import seaborn as sns
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

with open('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/label_descriptions.json', 'r') as file:
    label_desc = json.load(file)
sample_sub_df = pd.read_csv('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/sample_submission.csv')
train_df = pd.read_csv('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7/train.csv')

print(f'Shape of training dataset: {train_df.shape}')

print(f'# of images in training set: {train_df["ImageId"].nunique()}')
print(f'# of images in test set: {sample_sub_df["ImageId"].nunique()}')

categories_df = pd.DataFrame(label_desc['categories'])
attributes_df = pd.DataFrame(label_desc['attributes'])

category_map, attribute_map = {}, {}
for cat in label_desc.get('categories'):
    category_map[cat.get('id')] = cat.get('name')
for attr in label_desc.get('attributes'):
    attribute_map[attr.get('id')] = attr.get('name')

#train_df['ClassId'] = train_df['ClassId'].map(category_map)
#train_df['ClassId'] = train_df['ClassId'].astype('category')

#train_df['ClassId'] = train_df['ClassId'].cat.codes
train_df = train_df.drop('AttributesIds', axis=1)

image_df = train_df.groupby('ImageId')['EncodedPixels', 'ClassId'].agg(lambda x: list(x))
size_df = train_df.groupby('ImageId')['Height', 'Width'].mean()
input_df = image_df.join(size_df, on='ImageId')

import os
from pathlib import Path

#COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
COCO_WEIGHTS_PATH = 'train/fashion20200520T1756/mask_rcnn_fashion_0738.h5'

DATA_DIR = Path('/mnt/disk2/dl_data/imaterialist-fashion-2020-fgvc7')
TRAIN_DIR = Path('train')

import sys

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


class FashionConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "fashion"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(categories_df)  # background + 46 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side
    # in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10


config = FashionConfig()
config.display()


class FashionDataset(utils.Dataset):
    def __init__(self, df):
        super().__init__(self)

        self.IMAGE_SIZE = 512

        # Add classes
        for cat in label_desc['categories']:
            self.add_class('fashion', cat.get('id') + 1, cat.get('name'))

        # Add images
        for i, row in df.iterrows():
            self.add_image("fashion",
                           image_id=row.name,
                           path=str(DATA_DIR / 'train' / row.name) + '.jpg',
                           labels=row['ClassId'],
                           annotations=row['EncodedPixels'],
                           height=row['Height'], width=row['Width']
                           )

    def _resize_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        return img

    def load_image(self, image_id):
        return self._resize_image(self.image_info[image_id]['path'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [x for x in info['labels']]

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        mask = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []

        count = 0

        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height'] * info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]

            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel + annotation[2 * i + 1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')

            sub_mask = cv2.resize(sub_mask, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

            mask[:, :, m] = sub_mask
            labels.append(int(label) + 1)

            count = count + 1

        return mask, np.array(labels)


# This code partially supports k-fold training,
# you can specify the fold to train and the total number of folds here
FOLD = 0
N_FOLDS = 2

kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
splits = kf.split(input_df)  # ideally, this should be multilabel stratification


def get_fold():
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return input_df.iloc[train_index], input_df.iloc[valid_index]


train_df, valid_df = get_fold()

train_dataset = FashionDataset(train_df)
train_dataset.prepare()

valid_dataset = FashionDataset(valid_df)
valid_dataset.prepare()

# Note that any hyperparameters here, such as LR, may still not be optimal
LR = 1e-3

EPOCHS = [50, 150, 480, 1000]

learning_rate_base = 1e-4
total_steps = 200
warmup_steps = int(10 * 12 / 16)

import warnings
warnings.filterwarnings("ignore")

model = modellib.MaskRCNN(mode='training', config=config, model_dir=TRAIN_DIR)

# Load weights trained on MS COCO, but skip layers that
# are different due to the different number of classes
# See README for instructions to download the COCO weights
model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])

augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0.08, 0.15))
])


# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)


model.train(train_dataset, valid_dataset,
            learning_rate=LR*2, # train heads with higher lr to speedup learning
            epochs=EPOCHS[0],
            layers='heads',
            augmentation=augmentation
            )

history = model.keras_model.history.history


model.train(train_dataset, valid_dataset,
            learning_rate=LR,
            epochs=EPOCHS[1],
            layers='all',
            augmentation=augmentation,
            #custom_callbacks=[warm_up_lr]
            )

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

model.train(train_dataset, valid_dataset,
            learning_rate=LR/5,
            epochs=EPOCHS[2],
            layers='all',
            augmentation=augmentation,
            #custom_callbacks=[warm_up_lr]
            )

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

model.train(train_dataset, valid_dataset,
            learning_rate=LR/10,
            epochs=EPOCHS[3],
            layers='all',
            augmentation=augmentation,
            #custom_callbacks=[warm_up_lr]
            )

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

epochs = range(EPOCHS[-1])

best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch-1])

