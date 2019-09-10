# -*- coding: utf-8 -*-
#@author:huan

from sklearn.metrics import auc,roc_curve,confusion_matrix,f1_score,recall_score

import sys
sys.path.append('/home/longpeiji/utils')

from user_datagenerator import User_Generator
import pandas as pd
from glob import glob
import numpy as np
from keras.models import load_model
import time
import os

def evaluate_model(data_dir1,data_dir2,model,im_num,pred_save_path,evaluate_save_path):
    data_gene=User_Generator(data_dir1,data_dir2)
    y_true,y_pred=[]
    for x_test,label in data_gene.test_or_validation(200):
        y_p=model.predict([x_test])
        for y1,y2 in zip(y_p,label):
            y_pred.append(y1)
            y_true.append(y2)
    pd.DataFrame(y_pred).to_excel(pred_save_path)
    fpr,tpr,thresholds=roc_curve(y_true,y_pred)
    auc_value=auc(fpr,tpr) 
    
    score=[np.where(x>=0.5,1,0) for x in y_pred]
    u=confusion_matrix(y_true,score)
    Accuracy=(u[0][0]+u[1][1])/(u[0][0]+u[0][1]+u[1][0]+u[1][1])
    Sensitivity=u[1][1]/(u[1][1]+u[1][0])  #敏感度,tpr
    Specificity=u[0][0]/(u[0][1]+u[0][0])  #特异性，FPR
    f1_v=f1_score(y_true,score)
    pd.DataFrame(data={'--Accuracy--':[str(Accuracy)+'**'],\
                       '--auc--':[str(auc_value)+'**'],\
                       '--Sensitivity--':[str(Sensitivity)+'**'],\
                       '--Specificity--':[str(Specificity)+'**'],\
                       '--f1_v':[str(f1_v)+'**']})\
                       .to_excel(evaluate_save_path)
                       
def write_result(model_base_path,x,y,i,save_path):
    model=load_model(model_base_path)
    y_pred=model.predict([x])
#    pd.DataFrame(y_pred).to_excel(save_path+'/'+'y_pred'+str(i)+'.xlsx')
    fpr,tpr,thresholds=roc_curve(y,y_pred)
    auc_value=auc(fpr,tpr) 
    
    score=[np.where(x>=0.5,1,0) for x in y_pred]
    u=confusion_matrix(y,score)
    Accuracy=(u[0][0]+u[1][1])/(u[0][0]+u[0][1]+u[1][0]+u[1][1])
    Sensitivity=u[1][1]/(u[1][1]+u[1][0])  #敏感度,tpr
    Specificity=u[0][0]/(u[0][1]+u[0][0])  #特异性，FPR
    f1_v=f1_score(y,score)
    r=recall_score(y,score)
    pd.DataFrame(data={'Accuracy':[Accuracy],\
                       'auc':[auc_value],\
                       'recall_score':[r],\
                       'Sensitivity':[Sensitivity],\
                       'Specificity':[Specificity],\
                       'f1_v':[f1_v]})\
                       .to_excel(save_path+'/'+'validation_y_pred'+str(i)+'.xlsx')

#print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))    
#path='D:/learning/keras_dr_image_assessment/data_augmentation'
#type1_path='D:/huan/image_assessment/validation/blur'
#type2_path='D:/huan/image_assessment/validation/no_blur'

path='/home/longpeiji/keras_user_resnet/save_model'
save_path='/home/longpeiji/keras_user_resnet/validation'
type1_path='/home/longpeiji/image_assessment/validation/blur'
type2_path='/home/longpeiji/image_assessment/validation/no_blur'
im_num=2000

data_gen=User_Generator(type1_path,type2_path)
for x,y in data_gen.test_or_validation(im_num):
    while 1:
        try:   
            l=len(glob(save_path+'/*.xlsx'))
            if l==0:
                for p in glob(path+'/*.h5'): 
                    num=int(os.path.basename(p).split('_')[1].split('.')[0])         
                    write_result(p,x,y,num,save_path)
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))            
            else:
                for p in glob(path+'/*.h5'): 
                    num=int(os.path.basename(p).split('_')[1].split('.')[0])
                    if num>l-1:           
                        write_result(p,x,y,num,save_path)
                        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))  
            
        except OSError :  #处理文件没有传完就去读的异常
            time.sleep(60)  #给model.save 60s时间进行模型保存
