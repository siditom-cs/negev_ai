import keras
import keras.backend as K
import glob
import os
import shutil

from PIL import Image
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
from keras.losses import SparseCategoricalCrossentropy
import keras.callbacks as callbacks
from keras.callbacks import  ModelCheckpoint
from keras.callbacks import Callback
import tensorflow as tf

train_im_path, train_mask_path = os.path.join(args.train_path,'Orthophoto'), os.path.join(args.train_path,'Ground_truth')
h, w, batch_size = args.image_hight, args.image_width, args.batch_size
val_im_path, val_mask_path = os.path.join(args.train_path,'Orthophoto'), os.path.join(args.train_path,'Ground_truth')


scce = SparseCategoricalCrossentropy()



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
    a = DataGenerator(batch_size=32,shuffle=False, augmentations=AUGMENTATIONS_TRAIN, train_im_path=train_im_path,train_mask_path=train_mask_path)
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

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)

def separateIm2patches(path=args.train_path, size=256):
    outdir = os.path.join(path+'_'+str(size))
    if not os.path.exists(outdir):
        #print("Warning::separateIm2patches:: split to pathes folder already exists")
        os.mkdir(outdir)
    filepaths = glob.glob(path+'/*')
    #filepaths = glob.glob(os.path.join(path,'Orthophoto')+'/*')
    for im_path in filepaths:
        #mask_path = im_path.replace('Orthophoto', 'Ground_truth').replace('ortho','mask')
        print(im_path) 
        tile(os.path.basename(im_path), path, path+'_'+str(size), size)

#separateIm2patches(train_im_path, size = args.tile_dim)
#separateIm2patches(train_mask_path, size = args.tile_dim)
#unittest_generator()


# https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow
def get_iou_vector(A, B):
    # Numpy version    
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        t = K.reshape(t,K.shape(p))
        non_bg_p = K.greater(p,0)
        non_bg_t = K.greater(t,0)
        eq_p_t = K.equal(p,t)
        intersect = K.all(K.stack([non_bg_p, non_bg_t, eq_p_t],axis=0),axis=0)
        intersect = K.get_value(K.sum(K.cast(intersect, dtype='float32')))
        union = K.any(K.stack([non_bg_p,non_bg_t],axis=0),axis=0)
        union = K.get_value(K.sum(K.cast(union, dtype='float32')))
        #metric += K.sum(K.cast(K.all(K.stack([non_bg_p, non_bg_t, eq_p_t],axis=0),axis=0))#/K.sum(K.greater((non_bg_p+non_bg_t),0))
        metric += intersect/union
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.numpy_function(get_iou_vector, [label, K.argmax(pred, axis=-1)], tf.float64)

def get_bg_perc_vector(A, B):
    # Numpy version    
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        t = K.reshape(t,K.shape(p))
        bg_p = K.equal(p,0)
        metric += K.get_value(K.sum(K.cast(bg_p, dtype='float32')))/(A.shape[1]*A.shape[2])
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_bg_metric(label, pred):
    # Tensorflow version
    return tf.numpy_function(get_bg_perc_vector, [label, K.argmax(pred, axis=-1)], tf.float64)


def dice_coef_3cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 3 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_3cat_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_3cat(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return scce(y_true, y_pred) + dice_coef_3cat_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return scce(y_true, y_pred) - K.log(1. - dice_coef_3cat_loss(y_true, y_pred))


class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            callbacks.ModelCheckpoint(os.path.join(args.save_path,"model_weights.h5"),monitor='val_loss', 
                                   mode = 'min', save_best_only=True, verbose=1),
            #swa,
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)


