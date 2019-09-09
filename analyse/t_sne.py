# -*- coding: utf-8 -*-
#@author:huan

from sklearn.manifold  import TSNE

def T_SNE_data(data,init='pca',n_iter=1000):
    ts=TSNE(n_components=2,perplexity=30,init=init,n_iter=n_iter).fit_transform(data)
    return ts[:,0],ts[:,1]

