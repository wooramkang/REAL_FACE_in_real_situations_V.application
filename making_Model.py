from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import tensorflow as tf
from utils import *
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Reshape, Conv2DTranspose, Dropout
from keras.layers import Average, Convolution2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import regularizers
from keras import initializers
from keras.layers.core import Lambda, Flatten, Dense
from PIL import Image
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

    branch_0 = conv2d_bn(X, 96, 1, 1)

    branch_1 = conv2d_bn(X, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    branch_2 = conv2d_bn(X, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(X)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)

    return X


def inception_B(X):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(X, 384, 1, 1)

    branch_1 = conv2d_bn(X, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)

    branch_2 = conv2d_bn(X, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(X)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1)

    X = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return X


def inception_C(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    branch_0 = conv2d_bn(X, 256, 1, 1)

    branch_1 = conv2d_bn(X, 384, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = concatenate([branch_10, branch_11], axis=channel_axis)


    branch_2 = conv2d_bn(X, 384, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = concatenate([branch_20, branch_21], axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(X)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1)

    X = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)

    return X

def inception_reduction_A(X):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(X, 384, 3, 3, strides=(2,2), padding='valid')

    branch_1 = conv2d_bn(X, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, strides=(2,2), padding='valid')

    branch_2 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(X)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return X

def inception_reduction_B(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(X, 192, 1, 1)
    branch_0 = conv2d_bn(branch_0, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_1 = conv2d_bn(X, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, strides=(2,2), padding='valid')

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(X)

    X = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
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
    layer_filters = [256, 128, 64]
    channels = 3
    x = inputs

    checker = True

    for filters in layer_filters:
        if checker:
            x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=2,
                   activation='relu',
                   padding='same')(x)
            checker = False
        else:
            x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation='relu',
                   padding='same')(x)
            checker = True

    for filters in layer_filters[::-1]:
        if checker:
            x = Conv2DTranspose(filters=filters,
                   kernel_size=kernel_size,
                   strides=2,
                   activation='relu',
                   padding='same')(x)
            checker = False
        else:
            x = Conv2DTranspose(filters=filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation='relu',
                   padding='same')(x)
            checker = True

    x = Conv2DTranspose(filters=3,
                              kernel_size=kernel_size,
                              strides=2,
                              activation='sigmoid',
                              padding='same',
                              name='finaloutput_AE'
                              )(x)

    return x


def Super_resolution(X):

    layer_filters = [(64,1),(64,3),(64,5),(64,7)]
    x = X
    output_layer = []

    x = Conv2D(filters=128,
              kernel_size=(3,3),

              strides=1,
              activation='relu',
              padding='same')(x)

    for filters, kernel_size in layer_filters:
        output_layer.append(Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   activation='relu',
                                   padding='same')(x))

    avg_output = Average()(output_layer)

    out = Conv2D(3, (2,2), activation='relu', padding='same', name ='finaloutput_SUPresol')(avg_output)

    return out


def Stem_model(X):
    # input 299*299*3
    # output 35*35*384
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    X = conv2d_bn(X, 32, 3, 3, strides=(2,2), padding='valid')
    X = conv2d_bn(X, 32, 3, 3, padding='valid')
    X = conv2d_bn(X, 64, 3, 3)

    branch_0 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(X)
    branch_1 = conv2d_bn(X, 96, 3, 3, strides=(2,2), padding='valid')

    X = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(X, 64, 1, 1)
    branch_0 = conv2d_bn(branch_0, 96, 3, 3, padding='valid')

    branch_1 = conv2d_bn(X, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, padding='valid')

    X = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(X, 192, 3, 3, strides=(2,2), padding='valid')
    branch_1 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(X)

    X = concatenate([branch_0, branch_1], axis=channel_axis, name='stem_out')
    return X


def Inception_detail(X, classes):
    #params
    num_classes = classes
    #num_classes = 1000
    dropout_rate = 0.2

    X = inception_A(X)
    X = inception_A(X)
    X = inception_A(X)
    X = inception_A(X)
    X = inception_reduction_A(X)
    X = Dropout(rate=dropout_rate)(X)

    X = inception_B(X)
    X = inception_B(X)
    X = inception_B(X)
    X = inception_B(X)
    X = inception_B(X)
    X = inception_B(X)
    X = inception_B(X)
    X = inception_reduction_B(X)
    X = Dropout(rate=dropout_rate)(X)

    X = inception_C(X)
    X = inception_C(X)
    X = inception_C(X)

    X = AveragePooling2D((8, 8), padding='valid')(X)
    X = Dropout(rate=dropout_rate)(X)
    X = Flatten()(X)
    X = Dense(units=num_classes, activation='softmax')(X)

    return X


def triplet_loss(y_true, y_pred, alpha=0.3):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


def Model_mixed(input_shape, num_classes):

    X_input = Input(input_shape, name='model_input')

    #AUTOENCODER
    '''
    X = Autoencoder(X_input)
    autoencoder = Model(X_input,
                        X(X_input), name='AE')
    autoencoder.compile(loss='mse', optimizer='adam')
    '''
    autoencoder = Autoencoder(X_input)
    model = Model(X_input,
                  autoencoder, name = 'stem_Model')
    model.summary()

    #SUPER_RESOLUTION
    '''
    X = Super_resolution(X_input)
    super_resolution = Model(X_input,
                             X(X_input), name='SUPresol')
    super_resolution.compile(loss='mse', optimizer='adam')
    '''
    super_resolution = Super_resolution(autoencoder)
    model = Model(X_input,
                  super_resolution, name = 'stem_Model')
    model.summary()

    #INCEPTION
    '''
    X = Stem_mode(X_input)
    #stem_model = Model(X_input, X, name='stem_model')
    #stem_model.compile(loss='mse', optimizer='adam')
    '''
    stem_model = Stem_model(super_resolution)
    model = Model(X_input,
                  stem_model, name = 'stem_Model')
    model.summary()

    #DETAIL OF INCEPTION MODEL
    '''
    X = Inception_detail(X)
    inception_detail = Model(inputs=X_input,
                             outputs=X(X_input), name='inception_Model')
    inception_detail.compile(loss='mse', optimizer='adam')
    '''
    inception_detail = Inception_detail(stem_model, num_classes)
    #FINAL_MODELING
    model = Model(inputs=X_input,
                  outputs = inception_detail, name = 'inception_Model')
    model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    model.summary()
    return model
