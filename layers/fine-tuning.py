# -*- coding: utf-8 -*-
#@author:huan 

from keras.models import load_model
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
from user_datagenerator import User_Generator

model_path='/home/huan/桌面/mymodel.h5'
#model_path='D:/learning/cross_education_model/mymodel_3.h5'

model=load_model(model_path) #下载模型
model.trainable =True     
model.compile(optimizer=SGD(lr=model.optimizer.lr,momentum=0.9),loss='binary_crossentropy',metrics=[binary_accuracy])

train_type1_data_dir='/home/longpeiji/image_assessment/train/blur'
train_type2_data_dir='/home/longpeiji/image_assessment/train/no_blur'
#train_type1_data_dir='D:/huan/image_assessment/train/blur'
#train_type2_data_dir='D:/huan/image_assessment/train/no_blur'

data_gene=User_Generator(train_type1_data_dir,train_type2_data_dir)

epoch=200
start=100
for i in range(start,epoch,1):
#    history=model.fit_generator(data_gene.train_generator(100,110),\
#                                steps_per_epoch=110,epochs=1)

    history=model.fit_generator(data_gene.train_generator(100,110),\
                                steps_per_epoch=110,epochs=1,workers=16,use_multiprocessing=True,max_queue_size=110)        
      
    with open('/home/longpeiji/keras_dr/train/loss_and_acc'+'_'+str(i)+'.txt','w') as f:
        s=str(history.history['loss'][-1])+'---'+str(history.history['binary_accuracy'][-1])
        f.write(s)
    model.save('/home/longpeiji/keras_dr/save_model/mymodel'+'_'+str(i)+'.h5')



    
    
