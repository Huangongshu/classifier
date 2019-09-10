# -*- coding: utf-8 -*-
#@author:huan

from sklearn.metrics import auc,roc_curve,confusion_matrix,f1_score,recall_score
import sys
sys.path.append('/home/longpeiji/utils')
from user_datagenerator import User_Generator
import pandas as pd
import numpy as np
from keras.models import load_model
import time
                       
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
    return Accuracy,auc_value,r,Sensitivity,Specificity,f1_v


print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
path='/home/huan/Desktop/mymodel_55.h5'
save_path='/home/huan/Desktop'
type1_path='/home/huan/huan/new_validation/blur'
type2_path='/home/huan/huan/new_validation/no_blur'
im_num=200

data_gen=User_Generator(type1_path,type2_path)
Accuracy=auc_value=r1=Sensitivity=Specificity=r=f1_v=n=0


for x,y in data_gen.test_or_validation(im_num):     
    Ac,au,r,Se,Sp,f=write_result(path,x,y,1,save_path)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))            
    Accuracy+=Ac
    auc_value+=auc_value
    r+=r1
    Sensitivity+=Se
    Specificity+=Sp
    f1_v+=f
    n+=1
  
i=1    
pd.DataFrame(data={'Accuracy':[Accuracy/n],\
                   'auc':[auc_value/n],\
                   'recall_score':[r/n],\
                   'Sensitivity':[Sensitivity/n],\
                   'Specificity':[Specificity/n],\
                   'f1_v':[f1_v/n]})\
                   .to_excel(save_path+'/'+'y_pred'+str(i)+'.xlsx')        

            
  
        

