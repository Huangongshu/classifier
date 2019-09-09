# -*- coding: utf-8 -*-
#@author:huan 

import multiprocessing
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from t_sne import T_SNE_data
from keras.models import load_model,Model
from user_datagenerator import *
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/longpeiji/utils')

def select_k_value(data,init='random',start=2,end=20):
    scores=[]
    models=[]
    for i in range(start,end):
        model=KMeans(n_clusters=i,\
                  init=init,n_jobs=multiprocessing.cpu_count(),)
        model.fit(data)
        score=silhouette_score(data,model.labels_,metric='euclidean')
        scores.append(score)
        models.append(model)
    best_model=models[scores.index(max(scores))]
    
    return best_model

im_path=''
model_path='D:/learning/mymodel_55.h5'
base_model=load_model(model_path)
model=Model(inputs=base_model.input,outputs=base_model.get_layer('dense_1').output)

type1_path='D:/huan/image_assessment/train/blur'
type2_path='D:/huan/image_assessment/test/blur'

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


km=select_k_value(fe_map,end=10)
label=km.predict(fe_map)
tsne_result=T_SNE_data(fe_map,init='pca',n_iter=1000)
plt.scatter(tsne_result[0],tsne_result[1],c=label)
plt.show()
