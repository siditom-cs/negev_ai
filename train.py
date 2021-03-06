import os 
import keras 
from config import *
from generators import CombinedDataGenerator, DataGenerator, AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST
from utils import SnapshotCallbackBuilder, bce_dice_loss, dice_coef_3cat_loss, my_iou_metric, my_bg_metric
from base_model import *
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
    h, w, batch_size = args.image_hight, args.image_width, args.batch_size
     
    AUGMENTATIONS_TRAIN_ = Compose([
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
        RandomSizedCrop(min_max_height=( 64, 128), height=128, width=128,p=0.7),
        ToFloat(max_value=1)
    ],p=1)
    
    training_generator = DataGenerator(batch_size=2, augmentations=AUGMENTATIONS_TRAIN,img_size=512, train_im_path=train_im_path,train_mask_path=train_mask_path)
    #training_generator = DataGenerator(batch_size=2, augmentations=None,img_size=128, train_im_path=train_im_path,train_mask_path=train_mask_path)
    validation_generator = DataGenerator(batch_size=2, augmentations=AUGMENTATIONS_TEST,img_size=512, train_im_path=train_im_path,train_mask_path=train_mask_path) #TODO change the paths at runtime.
    
    snapshot = SnapshotCallbackBuilder(nb_epochs=args.epochs,nb_snapshots=1,init_lr=1e-3)
    
    
    model = build_effienet_unet(input_shape=(128, 128, 3))
    model.summary()
    
    #model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[my_iou_metric,my_bg_metric ])
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[my_iou_metric, my_bg_metric])
    model.compile(loss=dice_coef_3cat_loss, optimizer='adam', metrics=[my_iou_metric, my_bg_metric])
    history = model.fit(training_generator,
                    validation_data=training_generator,                            
                    use_multiprocessing=False,
                    epochs=args.epochs,verbose=1,
                    callbacks=snapshot.get_callbacks())


