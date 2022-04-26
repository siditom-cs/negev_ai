import os 
import keras 
from tensorflow.keras.optimizers import Adam
from config import *
from generators import CombinedDataGenerator, DataGenerator, AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST, BinaryDataGenerator
from utils import SnapshotCallbackBuilder, bce_dice_loss, dice_coef_3cat_loss, my_iou_metric, my_bg_metric, my_acc_metric, dice_loss, dice_coef, soft_dice_loss_road, dice_coef_road, iou_coef_road
from road_binary_model import *
from focal_loss import SparseCategoricalFocalLoss
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)


if __name__ == '__main__':

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


    #train_im_path, train_mask_path = os.path.join(args.train_path,'combined'), os.path.join(args.train_path,'Ground_truth')
    #train_im_path, train_mask_path = os.path.join(args.train_path,'Orthophoto_128'), os.path.join(args.train_path,'Ground_truth_128')
    train_im_path, train_mask_path = os.path.join(args.train_path,'Orthophoto'), os.path.join(args.train_path,'Ground_truth')
    val_im_path, val_mask_path = os.path.join(args.validation_path,'Orthophoto'), os.path.join(args.validation_path,'Ground_truth')
    h, w, batch_size = args.image_hight, args.image_width, args.batch_size
     
    
    training_generator = BinaryDataGenerator(batch_size=16, augmentations=AUGMENTATIONS_TRAIN,img_size=256, train_im_path=train_im_path,train_mask_path=train_mask_path)
    #training_generator = DataGenerator(batch_size=2, augmentations=None,img_size=128, train_im_path=train_im_path,train_mask_path=train_mask_path)
    validation_generator = BinaryDataGenerator(batch_size=16, augmentations=AUGMENTATIONS_TEST,img_size=256, train_im_path=val_im_path,train_mask_path=val_mask_path) #TODO change the paths at runtime.
    
    snapshot = SnapshotCallbackBuilder(nb_epochs=args.epochs,nb_snapshots=1,init_lr=1e-3)
    
    
    model = build_unet(input_shape=(256, 256, 3))
    opt = Adam(learning_rate=0.0001)
    loss = SparseCategoricalFocalLoss(gamma=2, class_weight=[0.1,10,10])
    if args.load > 0:
        model.load_weights(os.path.join(args.load_path,'model_weights.h5'))
        opt = Adam(learning_rate=0.0001)
    model.summary()
    #model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[my_iou_metric,my_bg_metric ])
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[my_iou_metric, my_bg_metric])
    #model.compile(loss=dice_coef_3cat_loss, optimizer='adam', metrics=[my_iou_metric, my_bg_metric, my_acc_metric])
    model.compile(loss=soft_dice_loss_road, optimizer='adam', metrics=[iou_coef_road, my_bg_metric, my_acc_metric], run_eagerly=True)
    #model.compile(loss=dice_loss, optimizer='adam', metrics=[dice_coef, my_bg_metric, my_acc_metric])
    history = model.fit(training_generator,
                    validation_data=training_generator,                            
                    use_multiprocessing=False,
                    epochs=args.epochs,verbose=1,
                    callbacks=snapshot.get_callbacks())


