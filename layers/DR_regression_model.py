# -*- coding: utf-8 -*-
#@author:huan

import sys
sys.path.append('/home/longpeiji/efficientnet-master/')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2 
import numpy as np
import pandas as pd
from efficientnet import EfficientNetB5
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D,Dropout,Dense
from keras.models import Sequential
from keras.activations import elu
from keras.optimizers import Adam

def crop_image_from_gray(img, tol=7):
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
    return image

def append_ext(fn):
    return fn+".jpg"

def build_model():
    model = Sequential()
    model.add(effnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(5, activation=elu))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="linear"))
    model.compile(loss='mse',
                  optimizer=Adam(lr=0.00005),
                  metrics=['accuracy'])
    return model

IMG_WIDTH = 456
IMG_HEIGHT = 456
CHANNELS = 3
BATCH_SIZE = 16

train_df=pd.read_csv('/home/ouzhuang/kaggle_data/trainLabels.csv',dtype=str)
train_df["image"]=train_df["image"].apply(append_ext)
TRAIN_IMG_PATH='/home/ouzhuang/kaggle_data'

# Add Image augmentation to our generator
train_datagen = ImageDataGenerator(rotation_range=180,
                                   horizontal_flip=True,
                                   zoom_range=0.2,
                                   channel_shift_range=10.,
                                   shear_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rescale=1 / 255.,
                                   preprocessing_function=preprocess_image,
                                   validation_split=0.3)


# Use the dataframe to define train and validation generators
train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    x_col='image',
                                                    y_col='level',
                                                    directory = TRAIN_IMG_PATH,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='other',
                                                    subset='training')

test_generator = train_datagen.flow_from_dataframe(train_df,
                                                  x_col='image',
                                                  y_col='level',
                                                  directory = TRAIN_IMG_PATH,
                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='other',
                                                  subset='validation')
# pip install -U git+https://github.com/qubvel/efficientnet 下载安装模块，并在模块中使用模型，from efficientnet import EfficientNetB5
#导入模型，这样用
#effnet = efn.EfficientNetB5(weights='/home/ouzhuang/model_weights/efficientnet-b5_imagenet_1000_notop.h5',
#                           include_top=False,
#                           input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))
#import efficientnet.tfkeras
#from tensorflow.keras.models import load_model
#model = load_model('path/to/model.h5')
#不然用load_model会找不到部分函数

effnet = EfficientNetB5(weights=None,
                        include_top=False,
                        input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))
effnet.load_weights('/home/longpeiji/model_weights/efficientnet-b5_imagenet_1000_notop.h5')

# Initialize model
model = build_model()
for i, layer in enumerate(effnet.layers):
    if "batch_normalization" in layer.name:
        effnet.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)

train_batch_stop_num=train_generator.samples//BATCH_SIZE
test_batch_stop_num=test_generator.samples//BATCH_SIZE
#batch_stop_num=4
epoch=50
for i in range(epoch):

    history=model.fit_generator(train_generator,steps_per_epoch=train_batch_stop_num,\
                                      epochs=1,validation_data=test_generator,\
                                      validation_steps=test_batch_stop_num,max_queue_size=10, \
                                      workers=8,use_multiprocessing=True)
    print(vars(history))
    with open('/home/longpeiji/keras_efficientnet/train/loss_and_acc' + '_' + str(i) + '.txt','w') as f:
        s = str(history.history['loss'][-1])
        f.write(s)
    with open('/home/longpeiji/keras_efficientnet/test/loss_and_acc'+'_'+str(i)+'.txt','w') as f:
        s=str(history.history['val_loss'][-1])
        f.write(s)
    model.save('/home/longpeiji/keras_efficientnet/model/mymodel'+'_'+str(i)+'.h5')
    
