# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#@authour:huan

#import matplotlib.pyplot as plt
#def show_acc(history):
#    """ 绘制精度曲线 """
#    plt.clf()
#    history_dict = history.history
#    acc = history_dict['binary_accuracy']
#    val_acc = history_dict['val_binary_accuracy']
#
#    epochs = range(1, len(val_acc) + 1)
#
#    plt.plot(epochs, acc, 'bo', label='Training acc')
#    plt.plot(epochs, val_acc, 'b', label='Validation acc')
#    plt.xlabel('Epochs')
#    plt.ylabel('Acc')
#    plt.legend()
#    plt.show()
    
from keras.models import Model
from keras.metrics import binary_accuracy
#from keras.preprocessing import image
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
    
    pd.DataFrame(data={'---loss---------':['**'+str(np.mean(train_loss))+'**'],\
                       '-----train_accuracy-----':[str(np.mean(train_accuracy))+'**'],\
                       '-----Accuracy---':[str(Accuracy)+'**'],\
                       '---auc---':[str(auc_value)+'**'],\
                       '---Precision---':[str(Precision)+'**'],\
                       '---Sensitivity---':[str(Sensitivity)+'**'],\
                       '---Specificity--':[str(Specificity)+'**'],\
                       '--f1_v':[str(f1_v)+'**']})\
                       .to_csv(evaluate_save_path)


w1='/home/longpeiji/model_weights/mobilenet_1_0_224_tf.h5'
w2='D:/learning/model_weights/mobilenet_1_0_224_tf.h5'
base_model=MobileNet(input_shape=(224,224,3),weights=w1)
#base_model.summary() #打印网络层
x=base_model.get_layer('conv_pw_13_relu').output
x=GlobalAveragePooling2D()(x)
#x=Dropout(0.5)(x)
x=Dense(100,activation='relu')(x)
pred=Dense(1,activation='sigmoid')(x) #二分类一般用sigmoid，softmax一般用于多分类
#pred=Dense(1,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=pred)

#for n,layer in enumerate(model.layers):#打印网络层
#    print(n,layer.name)
   
#for layer in model.layers:
#    model.trainable =True   

#for layer in model.layers[:68]: #68层
#    model.trainable =False
#    
#for layer in model.layers[68:]:
#    model.trainable =True    

#model.compile(optimizer='Adam',loss='categorical_crossentropy')
model.compile(optimizer='rmsprop',loss='binary_crossentropy',\
              metrics=[binary_accuracy])

train_data_dir1='/home/longpeiji/image_assessment/train'
train_data_dir2='D:/huan/image_assessment/train'

#train_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)#库里的生成器
#train_generator = train_data_gen.flow_from_directory(train_data_dir1,target_size=(224,224),color_mode='rgb',batch_size=100,class_mode='binary',shuffle=True)
#history=model.fit_generator(train_generator,steps_per_epoch=111,epochs=15)


#epochs=2#use model.train_on_batch()
#for i in range(epochs):
#    for x,y in data_gene.generate_batch(4,2):
#        model.train_on_batch(x,y)
#        
#data_gene=User_Generator('/home/longpeiji/image_assessment/train/blur','/home/longpeiji/image_assessment/train/no_blur')
data_gene=User_Generator('D:/huan/image_assessment/train/blur',\
                         'D:/huan/image_assessment/train/no_blur') #自己定义的生成器
epoch=100
for i in range(epoch):
    if epoch%2==0:
        for layer in model.layers[:68]: #68层
            model.trainable =False
    
        for layer in model.layers[68:]:
            model.trainable =True 
    else:
        for layer in model.layers[:68]: #68层
            model.trainable =True
    
        for layer in model.layers[68:]:
            model.trainable =False  
    history=model.fit_generator(data_gene.train_generator(200,55),\
                                steps_per_epoch=55,epochs=1)
    test_pred_save_path='/home/longpeiji/corss_education_model/user_test_y_pred'+'_'+str(i)+'.xlsx'
    test_evaluate_save_path='/home/longpeiji/corss_education_model/user_test_evaluate'+'_'+str(i)+'.txt'

    test_data_dir='/home/longpeiji/image_assessment/test'
#    test_data_dir='D:/huan/image_assessment/test'
    
    #Each iteration write down training and test data
    evaluate_model(test_data_dir,4765,test_pred_save_path,\
                   test_evaluate_save_path,history.history['loss'],\
                   history.history['binary_accuracy'])
    model.save('/home/longpeiji/corss_education_model/mymodel'+'_'+str(i)+'.h5')
    

#fine-tuning
#for layer in model.layers[:68]: #68层
#    model.trainable =True
#    
#for layer in model.layers[68:]:
#    model.trainable =False   
#model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),\
#              loss='binary_crossentropy',metrics=['accuracy'])

#train_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
#train_generator = train_data_gen.flow_from_directory(train_data_dir1,target_size=(224,224),color_mode='rgb',batch_size=100,class_mode='binary',shuffle=True)
#history=model.fit_generator(train_generator,steps_per_epoch=111,epochs=15)

#data_gene=User_Generator('/home/longpeiji/image_assessment/train/blur','/home/longpeiji/image_assessment/train/no_blur')
#data_gene=User_Generator('D:/huan/image_assessment/train/blur',\
#                         'D:/huan/image_assessment/train/no_blur')
#history=model.fit_generator(data_gene.train_generator(6,10),\
#                            steps_per_epoch=10,epochs=6)

