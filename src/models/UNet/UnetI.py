from tensorflow.keras.optimizers import Adam
from tensorflow import reduce_mean
from tensorflow import print as tf_print
from tensorflow import square as square
from tensorflow import tensor_scatter_nd_update
from tensorflow.math import count_nonzero
#import keras.backend as K
from keras import backend as K

import numpy as np

HYPER = None

def buildModel( hyperparameters):
    HYPER = hyperparameters
    
    input_layer = Input((HYPER.IMG_HEIGHT,HYPER.IMG_WIDTH,1))
    output_layer = buildUnetModel(input_layer,hyperparameters.START_NEURONS)    

    model_unet = Model(input_layer, output_layer)
    
    if not hasattr(hyperparameters, 'OPTIMIZER'):
        hyperparameters.OPTIMIZER = Adam()
        
    if hyperparameters.LOSS == "mae_nz":
        loss_function = mae_nz
    else:
        loss_function = hyperparameters.LOSS
    
    #model_unet.compile(optimizer=hyperparameters.OPTIMIZER, loss=hyperparameters.LOSS,metrics=[hyperparameters.LOSS])
    model_unet.compile(optimizer=hyperparameters.OPTIMIZER, loss=loss_function )#,metrics=[hyperparameters.LOSS])
    
    return model_unet



def mae_nz( y_true, y_pred):
    abs_difference = abs(y_true - y_pred)
    numnonzeros = count_nonzero(y_true - y_pred)
    
    loss= reduce_mean(abs_difference, axis=-1)     
    return loss * ((64.0*64.0)/K.cast(numnonzeros, K.tf.float32))  


import os, sys
import shutil
import glob
import re
from skimage import io
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation, Add
from tensorflow.keras.layers import ZeroPadding2D, Input, AveragePooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform


def buildUnetModel(input_layer, start_neurons):
    conv1 = Conv2D(start_neurons*1,(3,3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(start_neurons*1,(3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    
    conv2 = Conv2D(start_neurons*2,(3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(start_neurons*2,(3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)
    pool2 = Dropout(0.25)(pool2)

    conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(0.25)(pool3)
    
    conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(0.25)(pool4)

    #Middle
    convm = Conv2D(start_neurons * 16, (3,3), activation='relu', padding='same')(pool4)
    convm = Conv2D(start_neurons * 16, (3,3), activation='relu', padding='same')(convm)
    
    #upconv part
    deconv4 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.25)(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
    
    deconv3 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.25)(uconv3)
    uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(uconv3)
    uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(uconv3)
    
    deconv2 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.25)(uconv2)
    uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(uconv2)
    uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(uconv2)
    
    deconv1 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.25)(uconv1)
    uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same')(uconv1)
    uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same')(uconv1)
    
    output_layer = Conv2D(1, (1,1), padding='same', activation='sigmoid')(uconv1)
    
    return output_layer


