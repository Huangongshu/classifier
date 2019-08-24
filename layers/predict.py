# -*- coding: utf-8 -*-
#@author:huan

from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
import pandas as pd
import numpy as np
from sklearn.metrics import auc,roc_curve,confusion_matrix,f1_score
from keras.preprocessing.image import ImageDataGenerator

model_path='D:/learning/mymodel.h5'

test_data_dir1='/home/longpeiji/image_assessment/test'
test_data_dir2='D:/huan/image_assessment/test'
test_pred_save_path='y_pred.xlsx'
test_evaluate_save_path='evaluate.xlsx'

validation_data_dir1='/home/longpeiji/image_assessment/test'
validation_data_dir2='D:/huan/image_assessment/test'
test_pred_save_path='y_pred.xlsx'
test_evaluate_save_path='evaluate.xlsx'

model=load_model(model_path) #下载模型

def evaluate_model(data_dir,im_num,pred_save_path,evaluate_save_path,name):
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
    
    pd.DataFrame(data={name+'-Accuracy':[Accuracy],\
                          name+'-auc':[auc_value],\
                          name+'-Precision':[Precision],\
                          name+'-Sensitivity':[Sensitivity],\
                          name+'-Specificity ':[Specificity],\
                          name+'-f1_v':[f1_v]})\
                            .to_excel(evaluate_save_path)
  
im_path='D:/huan/image_assessment/validation'                      
evaluate_model(im_path,1986,test_pred_save_path,test_evaluate_save_path,'validation')
