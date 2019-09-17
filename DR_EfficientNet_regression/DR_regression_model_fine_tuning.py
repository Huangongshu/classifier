# -*- coding: utf-8 -*-
#@author:huan

import sys
sys.path.append('/home/longpeiji/efficientnet-master/efficientnet/')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2 
import numpy as np
import pandas as pd

from keras.backend import k
from initializers import EfficientConv2DKernelInitializer
from layers import Swish
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D,Dropout,Dense
from keras.models import Sequential
from keras.activations import elu
from keras.optimizers import Adam
from GroupNormalization import GroupNormalization

def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and 
    returns the a preprocessed image with 
    3 channels
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image, sigmaX=10):
    """
    The whole preprocessing pipeline:
    1. Read in image/home/longpeiji/keras_efficientnet
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
    return image

def append_ext(fn):
    return fn+".jpg"


IMG_WIDTH = 456
IMG_HEIGHT = 456
CHANNELS = 3
BATCH_SIZE = 16

#batch_stop_num=4
start_epoch=18
epoch=50
train_df=pd.read_csv('/home/ouzhuang/kaggle_data/trainLabels.csv')
train_df = train_df.astype({'image': 'str', 'level': 'int8'})
train_df["image"]=train_df["image"].apply(append_ext)
TRAIN_IMG_PATH='/home/ouzhuang/kaggle_data'

# Add Image augmentation to our generator
train_datagen = ImageDataGenerator(rotation_range=180,horizontal_flip=True,
                                   zoom_range=0.2,channel_shift_range=10.,
                                   shear_range=0.2,width_shift_range=0.2,
                                   height_shift_range=0.2,rescale=1 / 255.,
                                   preprocessing_function=preprocess_image)


# Use the dataframe to define train and validation generators
train_generator = train_datagen.flow_from_dataframe(train_df,x_col='image',y_col='level',
                                                    directory = TRAIN_IMG_PATH,target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,class_mode='other')

model=load_model('/home/longpeiji/keras_efficientnet/model/mymodel_17.h5')
init_lr=k.get_value(model.optimizer.lr)   
     
model.compile(optimizer=Adam(lr=init_lr),loss='mse',metrics=['accuracy'])
model.trainable =True  

train_batch_stop_num=train_generator.samples//BATCH_SIZE

for i in range(start_epoch,epoch):

    history=model.fit_generator(train_generator,steps_per_epoch=train_batch_stop_num,\
                                      epochs=1,max_queue_size=10, \
                                      workers=8,use_multiprocessing=True)
    with open('/home/longpeiji/keras_efficientnet/train/loss_and_acc' + '_' + str(i) + '.txt','w') as f:
        s = str(history.history['loss'][-1])
        f.write(s)

    model.save('/home/longpeiji/keras_efficientnet/model/mymodel'+'_'+str(i)+'.h5')
    