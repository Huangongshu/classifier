# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:58:48 2019

@author: huan
"""

from sklearn.cluster import *
from keras.models import load_model,Model
from user_datagenerator import *
import numpy as np
from shutil import *

def return_id(label,y,*n):
    index_dict={}
    for i in n:
        index0=np.where(label==i)
        y=np.asarray(y)
        type_path0=y[index0]        
        index_dict[int(i)]=type_path0        
    return index_dict

def move_picture(src,dst):
    for i in src:
        copy(i,dst)
    
base_model=load_model('C:/Users/27011/Desktop/mymodel_27.h5')
model=Model(inputs=base_model.input,outputs=base_model.get_layer('dense_1').output)

#type1_path='/home/longpeiji/image_assessment/train/blur'
#type2_path='/home/longpeiji/image_assessment/test'
type1_path='D:/huan/test'
type2_path='D:/huan/test'

data_gene=User_Generator(type1_path,type2_path)

fe_map=[]
im_path=[]

for x,y in data_gene.data_path(1):
    xs=np.squeeze(np.asarray(x))
    ys=y
    im_path.append(ys[0])
    pred=model.predict([np.expand_dims(xs,axis=0)])
    pred=np.squeeze(np.asarray(pred))
    fe_map.append(pred)


#fe_map=np.squeeze(np.asarray(fe_map))
km=KMeans(n_clusters=3,init=random)
label=km.fit_predict(fe_map)
index_dict=return_id(label,im_path,0,1,2)
#print(index_dict[0])

cluster0_path='/home/longpeiji/cluster/0'
cluster1_path='/home/longpeiji/cluster/1'
cluster2_path='/home/longpeiji/cluster/2'
cluster3_path='/home/longpeiji/cluster/3'

move_picture(index_dict[0],cluster0_path)
move_picture(index_dict[1],cluster1_path)
move_picture(index_dict[2],cluster2_path)
move_picture(index_dict[3],cluster3_path)
