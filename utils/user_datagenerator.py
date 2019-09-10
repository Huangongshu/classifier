# -*- coding: utf-8 -*-
#@author:huan 

from dr_im_preprocessing import im_preprocess
from glob import glob
import numpy as np
import cv2
from random import shuffle


class User_Generator():
    
    def __init__(self,type1_path,type2_path,is_training=True,is_shuffle=True,img_format='jpg'):
        
        self.shuffle =shuffle
        self.img_format = img_format
        self.type1_path = type1_path
        self.type2_path = type2_path
        self.count=0
#        self.image_preprocessing=im_preprocess_func
        self._make_label_data(self.type1_path,0,self.type2_path,1)
#        if is_training == True:
#            self._get()
        if is_shuffle == True:
            self._shuffle()
 
    
    def _make_label_data(self,path0,num0,path1,num1):
        self.l0 = [(x,num0) for x in glob(path0+'/*.jpg')]
        self.l1 = [(x,num1) for x in glob(path1+'/*.jpg')]


#    def _get(self):
#        s = len(self.l1)//len(self.l0)
#        l3 = []
#        for i in range(s):
#            l3 = l3+self.l0
#        self.l0 = l3


    def _shuffle(self):
        shuffle(self.l0)
        shuffle(self.l1)


    def test_or_validation(self,batch_size,type_data=None):

        if type_data==None:
            data = self.l0+self.l1   
            shuttle(data)
        elif int(type_data)==0:
            data=self.l0
        else:
            data=self.l1
    #        assert s<=len(self.l0)
    #        assert s<=len(self.l1) 
          
        for start in range(0,len(data)-batch_size+1,batch_size):
            x=[];y=[]
            a=slice(start,start+batch_size)
            for path,label in data[a]:
                im = cv2.imread(path)
                im = cv2.resize(im,(224,224))
                b,g,r=cv2.split(im)
                im=cv2.merge((r,g,b))     
                im=(im/255.0-0.5)*2 #归一化                
                x.append(im)
                y.append(label)            
            yield x,y

    
    def train_generator(self,batch_size,batch_num):
        '''
        parameter:
            batch_num:the parameter must be given,and its product with
                  batch_size should be less than the total numbel of samples.
        '''        
        batch_s = batch_size//2
        b1=len(self.l0)//batch_s
        b2=len(self.l1)//batch_s
        i = 0      
        while 1:
            if i >= batch_num:
                i = 0
                self._shuffle()
            b = slice((i%b1)*batch_s,(i%b1)*batch_s+batch_s)
            c = slice((i%b2)*batch_s,(i%b2)*batch_s+batch_s)
            data = self.l0[b]+self.l1[c]
            
            shuffle(data)            
            y = [];x = []
            for path,label in data:
                im = cv2.imread(path)
                im = im_preprocess(im)
                im = cv2.resize(im,(224,224))
                b,g,r=cv2.split(im)
                im=cv2.merge((r,g,b))
                im=(im/255.0-0.5)*2 #归一化
                x.append(im)
                y.append(label)
            i += 1
            yield (np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32)) #要是元组，而要变成数组，不能是列表

    def data_path(self,batch_size,type_data=None): 
        
        if type_data==None:
            data = self.l0+self.l1    
        elif int(type_data)==0:
            data=self.l0
        else:
            data=self.l1         
        for start in range(0,len(data)-batch_size+1,batch_size):
            x=[];y=[]
            a=slice(start,start+batch_size)
            for path,label in data[a]:
                im = cv2.imread(path)
                im = cv2.resize(im,(224,224))
                b,g,r=cv2.split(im)
                im=cv2.merge((r,g,b))     
                im=(im/255.0-0.5)*2 #归一化                
                x.append(im)
                y.append(path)            
            yield x,y
    
    
if __name__=='__main__':
    path1 = 'D:/huan/image_assessment/validation/blur'
    path2 = 'D:/huan/image_assessment/validation/no_blur'
#    path1 = '/home/longpeiji/image_assessment/train/blur'
#    path2 = '/home/longpeiji/image_assessment/train/no_blur'
    data_gene = User_Generator(path1,path2)
    xs=[]
    for x,y in data_gene.data_path(100,0):
        xs.append(x)
