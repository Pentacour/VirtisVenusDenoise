HYPER = None

def buildModel( hyperparameters):
   
    if not hasattr(hyperparameters, 'OPTIMIZER'):
        hyperparameters.OPTIMIZER = RMSprop()

    HYPER = hyperparameters
    
    return build_convsim_model(HYPER.IMG_HEIGHT, HYPER.IMG_WIDTH, HYPER.OPTIMIZER)

#################

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation, Add
from tensorflow.keras.layers import ZeroPadding2D, Input, AveragePooling2D, Flatten, Dense, LeakyReLU, Reshape
import numpy as np
import tensorflow 
from tensorflow.keras.optimizers import Adam, RMSprop
from keras import backend as K
from tensorflow import reduce_mean
from tensorflow.math import count_nonzero

def mae_nz( y_true, y_pred):
    squared_difference = abs(y_true - y_pred)
    numnonzeros = count_nonzero(y_true - y_pred)
    
    loss= reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`    
    return loss * ((64.0*64.0)/K.cast(numnonzeros, K.tf.float32))  



def build_convsim_model( img_height, img_width, optimizer  ):
    input_img = Input(shape=(img_height, img_width, 1))
    
    # Up
    u_y = Conv2D(32, (4, 4), strides =(2,2), padding='same')(input_img)
    u_cube1 = Activation('relu')(u_y)
    u_cube2 = Conv2D(16, (4, 4), strides = (2,2), padding='same')(u_cube1)
    u_cube2 = Activation('relu')(u_cube2)
    u_cube3 = Conv2DTranspose(32, (4, 4), padding='same', strides=(2,2))(u_cube2)
    u_cube3 = Activation('relu')(u_cube3)
    u_add = Add()([u_cube1, u_cube3])
    u_out = Conv2DTranspose(1, (4,4), padding='same', strides=(2,2))(u_add)
    u_out = Activation('tanh')(u_out)
        
    # Down
    d_y = Conv2D(32, (4, 4), strides =(2,2), padding='same')(input_img)
    d_cube1 = Activation('relu')(d_y)
    d_cube2 = Conv2D(16, (4, 4), strides = (2,2), padding='same')(d_cube1)
    d_cube2 = Activation('relu')(d_cube2)
    d_cube3 = Conv2D(8, (4, 4), strides = (2,2), padding='same')(d_cube2)
    d_cube3 = Activation('relu')(d_cube3)
    d_cube4 = Conv2DTranspose(16, (4, 4), padding='same', strides=(2,2))(d_cube3)
    d_cube4 = Activation('relu')(d_cube4)
    d_add_in= Add()([d_cube2, d_cube4])
    d_cube5 = Conv2DTranspose(32, (4, 4), padding='same', strides=(2,2))(d_add_in)
    d_cube5 = Activation('relu')(d_cube5)
    d_add_ex= Add()([d_cube1, d_cube5])
    d_out   = Conv2DTranspose(1, (4,4), padding='same', strides=(2,2))(d_add_ex)
    d_out = Activation('tanh')(d_out)
    
    # SUM
    y_sum = Add()([u_out, d_out])
        
    model = Model(input_img, y_sum)
    model.compile(optimizer=optimizer, loss=mae_nz, metrics=['mean_absolute_error'])
    return model
                   
