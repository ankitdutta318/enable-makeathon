#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:15:44 2018

@author: cruxbreaker
"""

# Import libraries
import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

PATH = os.getcwd()
# Define data path
data_path = PATH + '/TrainData'
data_dir_list = os.listdir(data_path)

img_rows=128
img_cols=128
num_channel=1
num_epoch=20

# Define the number of classes
num_classes = 24

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
		img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)
        
# Assigning Labels

# Define the number of classes
num_classes = 24

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:100]=0
labels[100:500]=1
labels[500:900]=2
labels[900:1300]=3
labels[1300:1400]=4
labels[1400:1500]=5
labels[1500:1600]=6
labels[1600:2000]=7
labels[2000:2400]=8
labels[2400:2700]=9
labels[2700:3100]=10
labels[3100:3500]=11
labels[3500:3900]=12
labels[3900:4300]=13
labels[4300:4400]=14
labels[4400:4800]=15
labels[4800:5200]=16
labels[5200:5600]=17
labels[5600:6100]=18
labels[6100:6500]=19
labels[6500:6900]=20
labels[6900:7000]=21
labels[7000:7100]=22
labels[7100:]=23
	  
names = ['t',
         'o',
         'f',
         'p',
         'x',
         'y',
         'u',
         'b',
         'q',
         'l',
         'c',
         'd',
         'e',
         'k',
         'w',
         'a',
         'm',
         'g',
         'j',
         'n',
         'h',
         's',
         'v',
         'i']
	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.hdf5")
print("Loaded model from disk")

loaded_model=load_model('model.hdf5')



# ----------------------------------------Testing a new image----------------------------------------------
test_image = cv2.imread('Y.jpg')
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   
if num_channel==1:
	if K.image_dim_ordering()=='tf':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
# Predicting the test image
print((loaded_model.predict(test_image)))
prediction = loaded_model.predict_classes(test_image)
print(names[prediction[0]])