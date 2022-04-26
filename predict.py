import cv2
from PIL import Image
from itertools import product
import os
import numpy as np

def predict_image(model, image):
    reshaped_image = image.reshape(1,128,128,3)
    pred_image = model.predict(reshaped_image)
    return pred_image

def predict_full_image(model, img_dir):
    tiles = {}
    d = 128
    image = cv2.imread(img_dir)
    w, h = image.shape[0], image.shape[1]
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        cropped_image = image[i : i+d, j : j+d]
        pred = predict_image(model, cropped_image)
        #pred_converted = np.argmax(pred, axis=-1).reshape(128,128)
        pred_converted = (pred[:,:,:,0]>0.7).astype('int32')
        pred_converted = pred_converted.reshape(128,128)
        pred_converted[pred_converted == 1] = 255 
        #pred_converted[pred_converted == 2] = 255
        final_tile = Image.fromarray(pred_converted, 'L')
        tiles[(i,j)] = final_tile
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    full_image = Image.new('L', (512, 512))
    for i,j in grid:
        full_image.paste(tiles[(i, j)], (i,j))
        
    return full_image

