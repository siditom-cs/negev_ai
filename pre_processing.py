import glob
import os
import shutil

import mkdir as mkdir
from PIL import Image
import cv2
from itertools import product
import keras
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
#import seaborn as sns
#sns.set_style("white")

from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)


all_train_fn = glob.glob('./train/*')
total_samples = len(all_train_fn)
idx = np.arange(total_samples)
# train_fn, val_fn = train_test_split(all_train_fn, stratify=mask_df.labels, test_size=0.1, random_state=10)

# train_dir = './train'
# for full_fn in train_fn:
#     fn = full_fn.split('/')[-1]
#     shutil.move(full_fn, os.path.join(train_dir, fn))
#
# train_dir = './train'
# for full_fn in masks_train_fn:
#     fn = full_fn.split('/')[-1]
#     shutil.move(full_fn, os.path.join(train_dir, fn))
#
# train_dir = './validation'
# for full_fn in val_fn:
#     fn = full_fn.split('/')[-1]
#     shutil.move(full_fn, os.path.join(train_dir, fn))
#
# train_dir = './validation'
# for full_fn in masks_val_fn:
#     fn = full_fn.split('/')[-1]
#     shutil.move(full_fn, os.path.join(train_dir, fn))

# DATA GENERATOR

train_im_path, train_mask_path = './train', './train_mask'
h, w, batch_size = 256, 256, 16

val_im_path, val_mask_path = './validation', './validation_mask'


class DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'

    def __init__(self, train_im_path=train_im_path, train_mask_path=train_mask_path,
                 augmentations=None, batch_size=batch_size, img_size=256, n_channels=3, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.train_im_paths = glob.glob(train_im_path + '/*')

        self.train_im_path = train_im_path
        self.train_mask_path = train_mask_path

        self.img_size = img_size

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.train_im_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:min((index + 1) * self.batch_size, len(self.train_im_paths))]

        # Find list of IDs
        list_IDs_im = [self.train_im_paths[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_IDs_im)

        if self.augment is None:
            return X, np.array(y) / 255
        else:
            im, mask = [], []
            for x, y in zip(X, y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im), np.array(mask) / 255

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.train_im_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(list_IDs_im), self.img_size, self.img_size, self.n_channels))
        y = np.empty((len(list_IDs_im), self.img_size, self.img_size, 1))

        # Generate data
        for i, im_path in enumerate(list_IDs_im):
            im = np.array(Image.open(im_path))
            mask_path = im_path.replace(self.train_im_path, self.train_mask_path)

            mask = np.array(Image.open(mask_path))

            if len(im.shape) == 2:
                im = np.repeat(im[..., None], 3, 2)

            # Resize sample
            X[i,] = cv2.resize(im, (self.img_size, self.img_size))

            # Store class
        #             y[i,] = cv2.resize(mask,(self.img_size,self.img_size))[..., np.newaxis]
        #             y[y>0] = 255

        return np.uint8(X), np.uint8(y)

# AUGMENTATIONS


AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    RandomSizedCrop(min_max_height=(128, 256), height=h, width=w,p=0.5),
    ToFloat(max_value=1)
],p=1)


AUGMENTATIONS_TEST = Compose([
    ToFloat(max_value=1)
],p=1)

# split image to small images

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)

# example
# im = Image.open(r"train\Orthophoto\ortho_100.png")
# tile("ortho_100.png",r"train\Orthophoto", r"train\Orthophoto\splitted", 50)

# main

a = DataGenerator(batch_size=64,shuffle=False)
images,masks = a.__getitem__(0)
max_images = 64
grid_width = 16
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))

for i,(im, mask) in enumerate(zip(images,masks)):
    ax = axs[int(i / grid_width), i % grid_width]
    ax.imshow(im.squeeze(), cmap="bone")
    ax.imshow(mask.squeeze(), alpha=0.5, cmap="Reds")
    ax.axis('off')
plt.suptitle("Chest X-rays, Red: Pneumothorax.")