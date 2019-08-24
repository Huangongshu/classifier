# -*- coding: utf-8 -*-
#@author:huan 

from dr_im_preprocessing import im_preprocess
from glob import glob
import numpy as np
import cv2
from random import shuffle


class User_Generator():
    
    def __init__(self,type1_path,type2_path,is_training=True,shuffle=True,img_format='jpg'):
        
        self.shuffle =shuffle
        self.img_format = img_format
        self.type1_path = type1_path
        self.type2_path = type2_path
        self.count=0
#        self.image_preprocessing=im_preprocess_func
        self._make_label_data(self.type1_path,0,self.type2_path,1)
        if is_training == True:
            self._get()
        if shuffle == True:
            self._shuffle()
 
    
    def _make_label_data(self,path0,num0,path1,num1):
        self.l0 = [(x,num0) for x in glob(path0+'/*.jpg')]
        self.l1 = [(x,num1) for x in glob(path1+'/*.jpg')]


    def _get(self):
        s = len(self.l1)//len(self.l0)
        l3 = []
        for i in range(s):
            l3 = l3+self.l0
        self.l0 = l3


    def _shuffle(self):
        shuffle(self.l0)
        shuffle(self.l1)


    def test_or_validation(self):
        '''
        parameter:
            batch_epoch:the parameter must be given,and its product with
                  batch_size should be less than the total numbel of samples.
        '''
        y = []
        x = []
#        s=batch_epoch*batch_size
#        assert s<=len(self.l0)
#        assert s<=len(self.l1)        
        data = self.l0+self.l1
        shuffle(data)
        
        for path,label in data:
            im = cv2.imread(path)
            im = im_preprocess(im)
            im = cv2.resize(im,(224,224))
            x.append(im)
            y.append(int(label))
            
        return np.array(x),np.array(y)
    
    
    def train_generator(self,batch_size,batch_num):
        
        batch_s = batch_size//2
        i = 0
        s = batch_num*batch_s
        assert s <= len(self.l0)
        assert s <= len(self.l1)
        
        while 1:
            if i >= batch_num:
                i = 0
                self._shuffle()
            x = [];y = []
            b = slice(i*batch_s,i*batch_s+batch_s)
            data=self.l0[b]+self.l1[b]
            
            if shuffle == True:
                shuffle(data)            
            y = [];x = []
            for path,label in data:
                im = cv2.imread(path)
                im = im_preprocess(im)
                im = cv2.resize(im,(224,224))
                x.append(im)
                y.append(int(label))
            i += 1
            yield (np.array(x),np.array(y))
 
if __name__=='__main__':
    path1 = 'D:/huan/new_high_res_merge/blur'
    path2 = 'D:/huan/new_high_res_merge/clear'
    data_gene = User_Generator(path1,path2)


