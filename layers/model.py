# -*- coding: utf-8 -*-
#@authour:huan
 
from keras.models import Model
from keras.metrics import binary_accuracy
from keras.applications.mobilenet import MobileNet,preprocess_input
from keras.layers import Dense,GlobalAveragePooling2D
#from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from sklearn.metrics import auc,roc_curve,confusion_matrix,f1_score
from user_datagenerator import User_Generator

def evaluate_model(data_dir,im_num,pred_save_path,evaluate_save_path,train_loss,train_accuracy):
    train_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = train_data_gen.flow_from_directory(data_dir,target_size=(224,224),\
                    color_mode='rgb',batch_size=im_num,class_mode='binary',shuffle=True)

    for x,y in train_generator:
        x_test=x
        y_true=y
        break
    y_pred=model.predict(x_test)
    pd.DataFrame(y_pred).to_excel(pred_save_path)
    fpr,tpr,thresholds=roc_curve(y_true,y_pred)
    auc_value=auc(fpr,tpr) 
    
    score=[np.where(x>=0.5,1,0) for x in y_pred]
    u=confusion_matrix(y_true,score)
    Accuracy=(u[0][0]+u[1][1])/(u[0][0]+u[0][1]+u[1][0]+u[1][1])
    Precision=u[0][0]/(u[0][0]+u[0][1])
    Sensitivity=u[1][1]/(u[1][1]+u[1][0])  #敏感度,tpr
    Specificity=u[0][0]/(u[0][1]+u[0][0])  #特异性，FPR
    f1_v=f1_score(y_true,score)
    
    pd.DataFrame(data={'---loss---------':['**'+str(train_loss)+'**'],\
                       '-----train_accuracy-----':[str(train_accuracy)+'**'],\
                       '-----Accuracy---':[str(Accuracy)+'**'],\
                       '---auc---':[str(auc_value)+'**'],\
                       '---Precision---':[str(Precision)+'**'],\
                       '---Sensitivity---':[str(Sensitivity)+'**'],\
                       '---Specificity--':[str(Specificity)+'**'],\
                       '--f1_v':[str(f1_v)+'**']})\
                       .to_csv(evaluate_save_path)


w1='/home/longpeiji/model_weights/mobilenet_1_0_224_tf.h5'
w2='D:/learning/model_weights/mobilenet_1_0_224_tf.h5'
base_model=MobileNet(input_shape=(224,224,3),weights=w2)
x=base_model.get_layer('conv_pw_13_relu').output
x=GlobalAveragePooling2D()(x)
x=Dense(100,activation='relu')(x)
pred=Dense(1,activation='sigmoid')(x) 

model=Model(inputs=base_model.input,outputs=pred)
 
for layer in model.layers:
    model.trainable =True
    
model.compile(optimizer='rmsprop',loss='binary_crossentropy',\
              metrics=[binary_accuracy])

train_data_dir1='/home/longpeiji/image_assessment/train'
train_data_dir2='D:/huan/image_assessment/train'

data_gene=User_Generator('D:/huan/image_assessment/train/blur',\
                         'D:/huan/image_assessment/train/no_blur')
epoch=2
for i in range(epoch):
    history=model.fit_generator(data_gene.train_generator(10,2),\
                                steps_per_epoch=2,epochs=1)
    test_pred_save_path='/home/longpeiji/save_model_imformation/user_test_y_pred'+'_'+str(i)+'.xlsx'
    test_evaluate_save_path='/home/longpeiji/save_model_imformation/user_test_evaluate'+'_'+str(i)+'.txt'
    test_data_dir='D:/huan/image_assessment/test'

    evaluate_model(test_data_dir,4765,test_pred_save_path,\
                   test_evaluate_save_path,history.history['loss'][-1],\
                   history.history['binary_accuracy'][-1])
                            