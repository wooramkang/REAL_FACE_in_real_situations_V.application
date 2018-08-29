from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from utils import *
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Reshape, Conv2DTranspose, merge, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from PIL import Image
import utils
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

'''

    written by wooramkang 2018.08.28
    i coded all the models
     and the bases of concepts are
     
        1. for making embedding
            inception-v4
            triplet-loss
            + etc 
            => my work = Facenet - inception-v2 + inception-v4 -stn + stn else
        
        2. for denosing
            denosing autoencoder
            
        3. for making resolution clear
            super-resolution
            especially, ESRCNN
            
        4. for removing lights
            CLAHE
            
        5. for twisted face
            affine transform
                   
'''


def conv2d_bn(X, nb_filter, num_row, num_col, padding='same', strides=(1, 1), use_bias=False):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    X = Convolution2D(nb_filter, (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(0.00004),
                      kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                      distribution='normal', seed=None))(X)
    X = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(X)
    X = Activation('relu')(X)
    return X

def inception_A(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    X1 = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_first')(X)
    X1 = Conv2D(64, (1, 1), strides=(1, 1))(X1)

    X2 = Conv2D(96, (1, 1), strides=(1,1))(X)

    X3 = Conv2D(64, (1, 1), strides=(1,1))(X)
    X3 = Conv2D(96, (3, 3), strides=(1, 1))(X3)

    X4 = Conv2D(64, (1, 1), strides=(1,1))(X)
    X4 = Conv2D(96, (3, 3), strides=(1, 1))(X4)
    X4 = Conv2D(96, (3, 3), strides=(1, 1))(X4)

    X = concatenate([X1, X2, X3, X4], axis=channel_axis)
    return X


def inception_B(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1


    return X


def inception_C(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    return X


def inception_reduction_A(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    return X

def inception_reduction_B(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    return X

def super_resolution(X):


    return X

def Autoencoder(inputs):
    '''
    written by wooramkang 2018.08.29

    this simple AE came from my AutoEncoder git

    and
    it's on modifying
    '''
    kernel_size = 3
    #latent_dim = 256
    layer_filters = [64, 128, 256]

    x = inputs

    for filters in layer_filters:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=2,
                   activation='relu',
                   padding='same')(x)

    '''
    shape_x = K.int_shape(x)

    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()

    latent_inputs = Input(shape=(latent_dim,), name='input_for_decoder')

    x = Dense(shape_x[1]*shape_x[2]*shape_x[3])(latent_inputs)
    x = Reshape((shape_x[1], shape_x[2], shape_x[3]))(x)
    '''

    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=2,
                            activation='relu',
                            padding='same')(x)

    outputs = Conv2DTranspose(filters=channels,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)
    '''
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()
    '''

    return outputs


def stem_model(X):
    # input 299*299*3
    # output 35*35*384
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    X = Conv2D(32, (3, 3), strides=(2, 2), name='init_conv1', padding='same')(X)
    X = Conv2D(32, (3, 3), strides=(1, 1), name='init_conv2', padding='same')(X)
    X = Conv2D(64, (3, 3), strides=(1, 1), name='init_conv3')(X)

    #first branch
    X1 = MaxPooling2D((3, 3), strides=2, name='branch_max1', padding='same')(X)
    X2 = Conv2D(82, (3, 3), strides=(2, 2), name='branch_conv1', padding='same')(X)
    X = concatenate([X1, X2], axis=channel_axis)

    #second branch
    X1 = Conv2D(64, (1, 1), strides=(1, 1))(X)
    X1 = Conv2D(96, (3, 3), strides=(1, 1), padding='same')(X1)

    X2 = Conv2D(64, (1, 1), strides=(1, 1))(X)
    X2 = Conv2D(64, (7, 1), strides=(1, 1))(X2)
    X2 = Conv2D(64, (1, 7), strides=(1, 1))(X2)
    X2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(X2)
    X = concatenate([X1, X2], axis=channel_axis)

    #third branch
    X1 =  Conv2D(192, (3, 3), strides=(1, 1), padding='same')(X)
    X2 =  MaxPooling2D((3, 3), strides=2, name='branch_max1', padding='same')(X)
    X = concatenate([X1, X2], axis=channel_axis)

    return X


def Model_mixed(input_shape):

    X_input = Input(input_shape, name='model_input')
    X = Autoencoder(X_input)
    X = stem_mode(X)
    X = inception_A(X)
    X = inception_A(X)
    X = inception_A(X)
    X = inception_A(X)
    X = inception_reduction_A(X)
    X = inception_B(X)
    X = inception_B(X)
    X = inception_B(X)
    X = inception_B(X)
    X = inception_B(X)
    X = inception_B(X)
    X = inception_B(X)
    X = inception_reduction_B(X)
    X = inception_C(X)
    X = inception_C(X)
    X = inception_C(X)

    model = Model(inputs = X_input, outputs = X, name='REALFACE_Model')
    return model


def triplet_loss(y_true, y_pred, alpha=0.3):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


model = Model_mixed(input_shape=(3, 96, 96))
model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])