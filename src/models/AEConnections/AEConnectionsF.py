HYPER = None

def buildModel( hyperparameters):
    if not hasattr(hyperparameters, 'OPTIMIZER'):
        hyperparameters.OPTIMIZER = Adam(0.001,beta_1=0.9)
        
    if hyperparameters.LOSS == "mae_nz":
        loss_function = mae_nz
    else:
        loss_function = hyperparameters.LOSS

        
    HYPER = hyperparameters
    return build_aeconn_model(HYPER.IMG_HEIGHT, HYPER.IMG_WIDTH, HYPER.OPTIMIZER, loss_function)

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
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from tensorflow import reduce_mean
from tensorflow.math import count_nonzero


def mae_nz( y_true, y_pred):
    squared_difference = abs(y_true - y_pred)
    numnonzeros = count_nonzero(y_true - y_pred)
    
    loss= reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`    
    return loss * ((64.0*64.0)/K.cast(numnonzeros, K.tf.float32))  


def build_aeconn_model( img_height, img_width, optimizer, loss_function ):

    input_img = Input(shape=(img_height, img_width, 1))
    
    #Encoder 
    y = Conv2D(32, (3, 3), padding='same',strides =(1,1))(input_img)
    y = LeakyReLU()(y)
    y = Conv2D(64, (3, 3), padding='same',strides =(2,2))(y)
    y = LeakyReLU()(y)
    y1 = Conv2D(128, (3, 3), padding='same',strides =(2,2))(y) # skip-1
    y = LeakyReLU()(y1)
    y = Conv2D(256, (3, 3), padding='same',strides =(2,2))(y)
    y = LeakyReLU()(y)
    y2 = Conv2D(256, (3, 3), padding='same',strides =(2,2))(y)# skip-2
    y = LeakyReLU()(y2)
    y = Conv2D(512, (3, 3), padding='same',strides =(2,2))(y)
    y = LeakyReLU()(y)
    y = Conv2D(1024, (3, 3), padding='same',strides =(2,2))(y)
    y = LeakyReLU()(y)#Flattening for the bottleneck
    vol = y.shape
    x = Flatten()(y)
    latent = Dense(128, activation='relu')(x) 
    	

    y = Dense(np.prod(vol[1:]), activation='relu')(latent)
    y = Reshape((vol[1], vol[2], vol[3]))(y)
    y = Conv2DTranspose(1024, (3,3), padding='same')(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(512, (3,3), padding='same',strides=(2,2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(256, (3,3), padding='same',strides=(2,2))(y)
    y= Add()([y2, y]) # second skip connection added here
    y = lrelu_bn(y)
    y = Conv2DTranspose(256, (3,3), padding='same',strides=(2,2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(128, (3,3), padding='same',strides=(2,2))(y)
    y= Add()([y1, y]) # first skip connection added here
    y = lrelu_bn(y)
    y = Conv2DTranspose(64, (3,3), padding='same',strides=(2,2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(32, (3,3), padding='same',strides=(2,2))(y)
    y = LeakyReLU()(y)
    y = Conv2DTranspose(1, (3,3), activation='sigmoid', padding='same',strides=(1,1))(y)
	

    model_1 = Model(input_img,y)
    model_1.compile(optimizer=optimizer, loss=loss_function)
    
    return model_1


# Helper function to apply activation and batch normalization to the 
# output added with output of residual connection from the encoderdef lrelu_bn(inputs):
    
def lrelu_bn(inputs):
   lrelu = LeakyReLU()(inputs)
   bn = BatchNormalization()(lrelu)
   return bn




