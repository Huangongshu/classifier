# -*- coding: utf-8 -*-
#@author:huan

import sys
sys.path.append('/home/longpeiji/efficientnet-master')

import cv2 
import numpy as np

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
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
    return image


from efficientnet import EfficientNetB5
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.layers import GlobalAveragePooling2D,Dropout,Dense
from keras.models import Sequential
from keras.activations import elu
from keras.optimizers import Adam
from GroupNormalization import GroupNormalization

IMG_WIDTH = 456
IMG_HEIGHT = 456
CHANNELS = 3

BATCH_SIZE = 200

train_df=pd.read_csv('/home/ouzhuang/kaggle_data/trainLabels.csv',dtype=str)
import pandas as pd
def append_ext(fn):
    return fn+".jpg"
train_df["image"]=train_df["image"].apply(append_ext)
TRAIN_IMG_PATH='/home/ouzhuang/kaggle_data'

# Add Image augmentation to our generator
train_datagen = ImageDataGenerator(rotation_range=360,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=preprocess_image, 
                                   rescale=1 / 128.,validation_split=0.25)

# Use the dataframe to define train and validation generators
train_generator = train_datagen.flow_from_dataframe(train_df, 
                                                    x_col='image', 
                                                    y_col='level',
                                                    directory = TRAIN_IMG_PATH,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='other'
                                                    subset='training')

test_generator = train_datagen.flow_from_dataframe(train_df, 
                                                  x_col='image', 
                                                  y_col='level',
                                                  directory = TRAIN_IMG_PATH,
                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='other'
                                                  subset='validation')

effnet = EfficientNetB5(weights=None,
                        include_top=False,
                        input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))
effnet.load_weights('/home/longpeiji/model_weights/efficientnet-b5_imagenet_1000_notop.h5')

for i, layer in enumerate(effnet.layers):
    if "batch_normalization" in layer.name:
        effnet.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)

def build_model():
    """
    A custom implementation of EfficientNetB5
    for the APTOS 2019 competition
    (Regression)
    """
    model = Sequential()
    model.add(effnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(5, activation=elu))
    model.add(Dense(1, activation="linear"))
    model.compile(loss='mse',
                  optimizer=Adam(lr=0.00005), 
                  metrics=['mse', 'acc'])
    print(model.summary())
    return model

# Initialize model
model = build_model()





epoch=100
for i in range(epoch):
#    history=model.fit_generator(data_gene.train_generator(100,110),\
#                                steps_per_epoch=110,epochs=1)
    
    train_history=model.fit_generator(train_generator,steps_per_epoch=train_generator.samples//BATCH_SIZE,epochs=1,workers=16,use_multiprocessing=True,max_queue_size=200)        
    test_history=model.fit_generator(train_generator,steps_per_epoch=train_generator.samples//BATCH_SIZE,epochs=1,workers=16,use_multiprocessing=True,max_queue_size=200)        
        
    with open('/home/longpeiji/keras_efficientnet/train/loss_and_acc'+'_'+str(i)+'.txt') as f:
        s=str(train_history.history['loss'][-1])+'---'+str(train_history.history['binary_accuracy'][-1])
        f.write(s)
    with open('/home/longpeiji/keras_efficientnet/test/loss_and_acc'+'_'+str(i)+'.txt') as f:
        s=str(test_history.history['loss'][-1])+'---'+str(test_history.history['binary_accuracy'][-1])
        f.write(s)               
    model.save('/home/longpeiji/keras_efficientnet/model/mymodel'+'_'+str(i)+'.h5')
    
