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
    
    #model_unet.compile(optimizer=hyperparameters.OPTIMIZER, loss=hyperparameters.LOSS,metrics=[hyperparameters.LOSS])
    model_unet.compile(optimizer=hyperparameters.OPTIMIZER, loss=mae_nz )#,metrics=[hyperparameters.LOSS])
    
    return model_unet





# from keras import backend as K
# import numpy as np

# def custom_loss(y_true, y_pred):
#     return y_pred / K.cast(K.tf.count_nonzero(y_true), K.tf.float32)

# y_t = K.placeholder((1,2))
# y_p = K.placeholder((1,2))

# loss = custom_loss(y_t, y_p)

# print(K.get_session().run(loss, {y_t: np.array([[1,1]]), y_p: np.array([[2,4]])}))

def depth_loss_func(pred_depth,actual_depth):
    n = K.shape(pred_depth)[0]
 #   n = pred_depth.shape[0]
    di = K.log(pred_depth)-K.log(actual_depth)
    di_sq = K.square(di)
    sum_d = K.sum(di)
    sum_d_sq = K.sum(di_sq)
    loss = ((1/n)*sum_d_sq)-((1/(n*n))*sum_d*sum_d) # getting an error in this step
    return loss

def my_root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def mae_nz( y_true, y_pred):
    squared_difference = abs(y_true - y_pred)
    numnonzeros = count_nonzero(y_true)
    
    loss= reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`    
    return loss * ((64.0*64.0)/K.cast(numnonzeros, K.tf.float32))  

    
    #y_true = y_true * (y_true != 0) 
    #y_pred = y_pred * (y_true != 0)
    
    #z = count_nonzero(loss)

    #return my_root_mean_squared_error(y_true, y_pred)

# def mae_nz( y_true, y_pred):
#     squared_difference = square(y_true - y_pred)
#     loss= reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
#     loss= reduce_mean(loss, axis=-1) 
#     loss= reduce_mean(loss, axis=-1)
    
#     loss = loss * K.cast(count_nonzero(loss), K.tf.float32)
    
#     #y_true = y_true * (y_true != 0) 
#     #y_pred = y_pred * (y_true != 0)
    
#     #z = count_nonzero(loss)

#     #return my_root_mean_squared_error(y_true, y_pred)
    
# #     value = 45.0
# #     indices = [0]

# #     loss = tensor_scatter_nd_update(loss, [indices], [value])
# #     print("Zeros="+str(z))
# #     print(y_true.shape)
# #     print("LOSS:" + str(loss.shape))
#     return loss
# # #    loss = K.square(y_pred - y_true)  # (batch_size, 2)
#     #tf_print(y_true.shape)
#  #   print(y_pred.shape)
#   #  print(loss.shape)
#    # return loss
    


def TODO_mae_nz( y_true, y_pred):
    print("\n ->" + str(y_true.shape)+str('\n'))
    loss = K.square(y_pred - y_true)  # (batch_size, 2)
    
    # summing both loss values along batch dimension 
    loss = K.sum(loss, axis=1)        # (batch_size,)
    
    print(loss)
    
    return loss    
#################

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


