# -*- coding: utf-8 -*-
#@authour:huan
 
from keras.models import Model
from keras.metrics import binary_accuracy
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense,GlobalAveragePooling2D,BatchNormalization,Dropout
from keras.optimizers import Adam

import sys
sys.path.append('/home/longpeiji/utils')
from user_datagenerator import User_Generator

w1='/home/longpeiji/model_weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
w2='D:/learning/model_weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
base_model=ResNet50(weights=w1)
x=base_model.get_layer('avg_pool').output
#x=GlobalAveragePooling2D()(x)
x=BatchNormalization()(x)
x=Dropout(0.5)(x)
x=Dense(100,activation='relu')(x)
pred=Dense(1,activation='sigmoid')(x) 

model=Model(inputs=base_model.input,outputs=pred)
model.trainable =True
   
model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy',\
              metrics=[binary_accuracy])

train_type1_data_dir='/home/longpeiji/image_assessment/train/blur'
train_type2_data_dir='/home/longpeiji/image_assessment/train/no_blur'
#train_type1_data_dir='D:/huan/image_assessment/train/blur'
#train_type2_data_dir='D:/huan/image_assessment/train/no_blur'

data_gene=User_Generator(train_type1_data_dir,train_type2_data_dir)

epoch=100
for i in range(epoch):
#    history=model.fit_generator(data_gene.train_generator(100,110),\
#                                steps_per_epoch=110,epochs=1)

    history=model.fit_generator(data_gene.train_generator(100,110),\
                                steps_per_epoch=110,epochs=1,workers=16,use_multiprocessing=True,max_queue_size=110)        
#    test_pred_save_path='/home/longpeiji/keras_user_resnet/save_model/user_test_y_pred'+'_'+str(i)+'.xlsx'
#    test_evaluate_save_path='/home/longpeiji/save_model_imformation/user_test_evaluate'+'_'+str(i)+'.txt'
#    test_pred_save_path='D:/learning/test_mobilenet_model/user_test_y_pred'+'_'+str(i)+'.xlsx'
#    test_evaluate_save_path='D:/learning/test_mobilenet_model/user_test_evaluate'+'_'+str(i)+'.txt'
    test_data_dir='/home/longpeiji/image_assessment/test'
    
    
    with open('/home/longpeiji/keras_user_resnet/save_model/loss_and_acc'+'_'+str(i)+'.txt','w') as f:
        s=str(history.history['loss'][-1])+'---'+str(history.history['binary_accuracy'][-1])
        f.write(s)
    model.save('/home/longpeiji/keras_user_resnet/save_model/mymodel'+'_'+str(i)+'.h5')
                            
