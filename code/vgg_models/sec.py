import os, sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]

#export CUDA_VISIBLE_DEVICES=0
import matplotlib.pyplot as plt

import keras
import tensorflow 
from clr_callback import *
import numpy as np
np.random.seed(2016)
import scipy
import os
import glob
import math
import pickle
import datetime
#import pandas as pd
from keras.layers.noise import GaussianDropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D,AveragePooling2D, Conv2D,MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import RMSprop,SGD
from keras.optimizers import Adam,Adagrad,Adadelta
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import RepeatVector
from keras.layers.normalization import BatchNormalization
#from keras.layers.core import  ActivityRegularization
#from keras.regularizers import WeightRegularizer
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import UpSampling2D
from keras.layers.noise import GaussianNoise
from keras.layers.core import Reshape
from keras.layers.recurrent import LSTM,GRU
#from keras.layers import Merge
#from keras.regularizers import l2
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
#from keras.applications.densenet import DenseNet121
#from keras_applications.imagenet_utils import _obtain_input_shape
from keras.applications import vgg16 as keras_vgg16
from keras.layers import BatchNormalization
from keras.models import Model
# dimensions of our images.
img_width, img_height = 224,224

#from keras.layers import Flatten, Dense, Input
#from keras_vggface.vggface import VGGFace

hidden_dim=512

train_data_dir = '/home/rahulp2/cov/new/train'
validation_data_dir = '/home/rahulp2/cov/new/validation'
test_data_dir = '/home/rahulp2/cov/new/test'
nb_train_samples = 336
nb_validation_samples = 72
nb_test_samples = 251
nb_epoch = 150
img_rows, img_cols, img_channel = 224, 224, 3
input_tensor_shape=(img_rows, img_cols, img_channel)

#input_layer = Input(shape=(224, 224,3),
 #             name='image_input')
base_model = keras_vgg16.VGG16(input_shape=(224,224,3),weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
#x= Conv2D(512, (3, 3), activation='relu', padding='same',)(x)
#x= BatchNormalization()(x)
#x= MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
#x= Conv2D(512,(1,1),activation='relu',padding='same')(x)
#x= BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
#x = ActivityRegularization(l2=0.01)(x)
#x=Dropout(0.2)(x)
#x = Flatten(name='flatten')(x)
x = Dense(64, activation='relu', name='fc6')(x)
x= Dropout(0.5, name='dropout_1')(x)
#x = Dense(512, activation='relu', name='fc7')(x)
#x=Dropout(0.5, name='dropout_2')(x)

predictions = Dense(1, activation='sigmoid', name='predictions')(x)


#input_layer = Input(shape=(224, 224,3),
 #             name='image_input')
#base_model = DenseNet121(input_shape=(224,224,3), weights='imagenet', include_top=False)

#for layer in base_model.layers:
#        layer.trainable=False
    
#x_model = base_model.output
#x_model = Flatten()(x_model)    
#x_model = Dense(128, activation='relu',name='fc1_Dense')(x_model)
#x_model = Dropout(0.5, name='dropout_1')(x_model)
    
#predictions = Dense(1, activation='sigmoid',name='output_layer')(x_model)
    
model = Model(input=base_model.input, output=predictions)

#vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
#last_layer = vgg_model.get_layer('pool5').output
#x = Flatten(name='flatten')(last_layer)
#x = Dense(hidden_dim, activation='relu', name='fc6')(x)
#x= Dropout(0.5, name='dropout_1')(x)
#x = Dense(hidden_dim, activation='relu', name='fc7')(x)
#x=Dropout(0.5, name='dropout_2')(x)
#out = Dense(1, activation='sigmoid', name='fc8')(x)
#model = Model(vgg_model.input, out


print(model.summary())

clr_triangular = CyclicLR(base_lr=0.0001, max_lr=0.001,
                        step_size=15.,mode='triangular')

rms =RMSprop( lr=0.0001)
 
#rms =Adagrad( lr=0.001,epsilon=1e-08)
model.compile(loss='binary_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])


filepath="weights2-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1)
callbacks_list = [checkpoint,clr_triangular]

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling

test_datagen = ImageDataGenerator(rescale=1./255)
vali_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')


validation_generator = vali_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='binary')

history= model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,validation_data=validation_generator,nb_val_samples=nb_validation_samples,
        callbacks=callbacks_list)

print(model.summary())

scores = model.evaluate_generator(test_generator, 251)
print("Accuracy = ", scores[1])

model.save_weights('first_try.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
#plt.show()
plt.savefig('acc.png')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('loss.png')
