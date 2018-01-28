#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:08:20 2017

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
K.set_image_dim_ordering('th')

#%%

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

#%%

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
        
#%%
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
	  
names = ['gesture_t',
         'gesture_o',
         'gesture_f',
         'gesture_p',
         'gesture_x',
         'gesture_y',
         'gesture_u',
         'gesture_b',
         'gesture_q',
         'gesture_l',
         'gesture_c',
         'gesture_d',
         'gesture_e',
         'gesture_k',
         'gesture_w',
         'gesture_a',
         'gesture_m',
         'gesture_g',
         'gesture_j',
         'gesture_n',
         'gesture_h',
         'gesture_s',
         'gesture_v',
         'gesture_i']
	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
	
#%%
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.hdf5")
print("Loaded model from disk")

loaded_model=load_model('model.hdf5')
#%%
# Evaluating the model

score = loaded_model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(loaded_model.predict(test_image))
print(loaded_model.predict_classes(test_image))
print(y_test[0:1])

# Testing a new image
test_image = cv2.imread('a.jpg')
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   
if num_channel==1:
	if K.image_dim_ordering()=='th':
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
print(loaded_model.predict_classes(test_image))

#%%

# Visualizing the intermediate layer

#
def get_featuremaps(loaded_model, layer_idx, X_batch):
	get_activations = K.function([loaded_model.layers[0].input, K.learning_phase()],[loaded_model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

layer_num=3
filter_num=0

activations = get_featuremaps(loaded_model, int(layer_num),test_image)

print (np.shape(activations))
feature_maps = activations[0][0]      
print (np.shape(feature_maps))

if K.image_dim_ordering()=='th':
	feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
print (feature_maps.shape)

fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))	
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
	ax.imshow(feature_maps[:,:,i],cmap='gray')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')

#%%
# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = loaded_model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
#y_pred = loaded_model.predict_classes(X_test)
#print(y_pred)
target_names = ['class 0(gesture_t)', 
                'class 1(gesture_o)', 
                'class 2(gesture_f)',
                'class 3(gesture_p)',
                'class 4(gesture_x)',
                'class 5(gesture_y)',
                'class 6(gesture_u)',
                'class 7(gesture_b)',
                'class 8(gesture_q)',
                'class 9(gesture_l)',
                'class 10(gesture_c)',
                'class 11(gesture_d)',
                'class 12(gesture_e)',
                'class 13(gesture_k)',
                'class 14(gesture_w)',
                'class 15(gesture_a)',
                'class 16(gesture_m)',
                'class 17(gesture_g)',
                'class 18(gesture_j)',
                'class 19(gesture_n)',
                'class 20(gesture_h)',
                'class 21(gesture_s)',
                'class 22(gesture_v)',
                'class 23(gesture_i)']

					
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
plt.show()
