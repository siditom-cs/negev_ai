import os 
import keras 
from config import *
from utils import DataGenerator, AUGMENTATIONS_TRAIN, SnapshotCallbackBuilder, bce_dice_loss, my_iou_metric
from base_model import *


if __name__ == '__main__':

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


    train_im_path, train_mask_path = os.path.join(args.train_path,'Orthophoto'), os.path.join(args.train_path,'Ground_truth')
    
    #training_generator = DataGenerator(batch_size=2, augmentations=AUGMENTATIONS_TRAIN,img_size=256, train_im_path=train_im_path,train_mask_path=train_mask_path)
    training_generator = DataGenerator(batch_size=2, augmentations=None,img_size=256, train_im_path=train_im_path,train_mask_path=train_mask_path)
    validation_generator = DataGenerator(batch_size=2, augmentations=None,img_size=256, train_im_path=train_im_path,train_mask_path=train_mask_path) #TODO change the paths at runtime.
    
    snapshot = SnapshotCallbackBuilder(nb_epochs=args.epochs,nb_snapshots=1,init_lr=1e-3)
    
    
    model = build_effienet_unet(input_shape=(256, 256, 3))
    model.summary()
    
    #model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[my_iou_metric])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[my_iou_metric])
    history = model.fit(training_generator,
                    validation_data=validation_generator,                            
                    use_multiprocessing=False,
                    epochs=args.epochs,verbose=2,
                    callbacks=snapshot.get_callbacks())


