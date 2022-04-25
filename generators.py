import keras
import keras.backend as K
import glob
import os
import shutil

from PIL import Image, ImageOps
import cv2
from itertools import product
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 

from config import *
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)

train_im_path, train_mask_path = os.path.join(args.train_path,'Orthophoto'), os.path.join(args.train_path,'Ground_truth')
h, w, batch_size = args.image_hight, args.image_width, args.batch_size
val_im_path, val_mask_path = os.path.join(args.train_path,'Orthophoto'), os.path.join(args.train_path,'Ground_truth')

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
            return X/255, np.array(y) 
        else:
            im, mask = [], []
            for x, y in zip(X, y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im)/255, np.array(mask) 

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
            mask_path = im_path.replace(self.train_im_path, self.train_mask_path).replace('ortho','mask')

            mask = np.array(Image.open(mask_path))

            if len(im.shape) == 2:
                im = np.repeat(im[..., None], 3, 2)

            # Resize sample
            X[i,] = cv2.resize(im, (self.img_size, self.img_size))

            #Store class
            mask = np.uint8((mask==0)*0 + (mask==128)*1 + (mask==255)*2)
            
            y[i,] = cv2.resize(mask,(self.img_size,self.img_size))[..., np.newaxis]
            
            #y[y>0] = 255

        return np.uint8(X), np.uint8(y)

class CombinedDataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'

    def __init__(self, train_im_path=train_im_path, train_mask_path=train_mask_path,
                 augmentations=None, batch_size=batch_size, img_size=256, n_channels=3, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.train_im_paths = glob.glob(train_im_path + '/*')

        self.train_im_path = train_im_path
        self.train_mask_path = train_mask_path
        self.train_slope_path = train_im_path.replace('Orthophoto', 'Slope')
        self.train_tcurv_path = train_im_path.replace('Orthophoto', 'Tang_curv')

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
            return X/255, np.array(y) 
        else:
            im, mask = [], []
            for x, y in zip(X, y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im)/255, np.array(mask) 

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
            mask_path = im_path.replace(self.train_im_path, self.train_mask_path).replace('ortho','mask')
            slope_path = im_path.replace(self.train_im_path, self.train_slope_path).replace('ortho','slope')
            tcurv_path = im_path.replace(self.train_im_path, self.train_tcurv_path).replace('ortho','tcurv')

            im = np.array(ImageOps.grayscale(Image.open(im_path)))
            slope = np.array(Image.open(slope_path))
            tcurv = np.array(Image.open(tcurv_path))
            mask = np.array(Image.open(mask_path))

            """
            print("shape(im): ", im.shape)
            print("np.max(im): ", np.max(im))
            print("np.min(im): ", np.min(im))
            print("shape(slope): ", slope.shape)
            print("np.max(slope): ", np.max(slope))
            print("np.min(slope): ", np.min(slope))
            print("shape(tcurv): ", tcurv.shape)
            print("np.max(tcurv): ", np.max(tcurv))
            print("np.min(tcurv): ", np.min(tcurv))
            """
            cmb_im = np.empty((im.shape[0], im.shape[1], 3))
            cmb_im[:,:,0] = im
            cmb_im[:,:,1] = slope
            cmb_im[:,:,2] = tcurv
            im = cmb_im
            if len(im.shape) == 2:
                im = np.repeat(im[..., None], 3, 2)

            # Resize sample
            X[i,] = cv2.resize(im, (self.img_size, self.img_size))

            #Store class
            mask = np.uint8((mask==0)*0 + (mask==128)*1 + (mask==255)*2)
            
            y[i,] = cv2.resize(mask,(self.img_size,self.img_size))[..., np.newaxis]
            
            #y[y>0] = 255

        return np.uint8(X), np.uint8(y)

# AUGMENTATIONS


"""
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
        OpticalDistortion(distort_limit=1, shift_limit=0.5),
        ], p=0.2),
    RandomSizedCrop(min_max_height=(128, 256), height=h, width=w,p=0.7),
    ToFloat(max_value=1)
],p=1)
"""
AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
         ], p=0.3),
    OneOf([
        #ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=1, shift_limit=0.5),
        ], p=0.1),
    RandomSizedCrop(min_max_height=(128, 256), height=h, width=w,p=0.7),
    ToFloat(max_value=1)
],p=1)

def unittest_generator():

    #train_im_path, train_mask_path = os.path.join(args.train_path,'Orthophoto_'+str(args.tile_dim)), os.path.join(args.train_path,'Ground_truth_'+str(args.tile_dim))
    
    train_im_path, train_mask_path = os.path.join(args.train_path,'Orthophoto'), os.path.join(args.train_path,'Ground_truth')
    #a = CombinedDataGenerator(batch_size=32,shuffle=False, augmentations=AUGMENTATIONS_TRAIN, train_im_path=train_im_path,train_mask_path=train_mask_path)
    a = CombinedDataGenerator(batch_size=2,shuffle=False, augmentations=None, train_im_path=train_im_path,train_mask_path=train_mask_path)
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
    plt.savefig("unittest_generator.png")

unittest_generator()
