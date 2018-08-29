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
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Reshape, Conv2DTranspose, merge, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
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

    X1 = AveragePooling2D(pool_size=(3, 3), strides=(1,1), data_format='channels_first')(X)
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

    X1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(X)
    X1 = Conv2D(128, (1, 1), strides=(1, 1))(X1)

    X2 = Conv2D(384, (1, 1), strides=(1,1))(X)

    X3 = Conv2D(192, (1, 1), strides=(1,1))(X)
    X3 = Conv2D(224, (7, 1), strides=(1, 1))(X3)
    X3 = Conv2D(256, (1, 7), strides=(1, 1))(X3)

    X4 = Conv2D(192, (1, 1), strides=(1,1))(X)
    X4 = Conv2D(192, (1, 7), strides=(1, 1))(X4)
    X4 = Conv2D(224, (7, 1), strides=(1, 1))(X4)
    X4 = Conv2D(224, (1, 7), strides=(1, 1))(X4)
    X4 = Conv2D(256, (7, 1), strides=(1, 1))(X4)

    X = concatenate([X1, X2, X3, X4], axis=channel_axis)
    return X


def inception_C(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    X1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(X)
    X1 = Conv2D(256, (1, 1), strides=(1, 1))(X1)

    X2 = Conv2D(256, (1, 1), strides=(1,1))(X)

    X3 = Conv2D(384, (1, 1), strides=(1,1))(X)
    X3_1 = Conv2D(256, (3, 1), strides=(1, 1))(X3)
    X3_2 = Conv2D(256, (1, 3), strides=(1, 1))(X3)

    X4 = Conv2D(384, (1, 1), strides=(1,1))(X)
    X4 = Conv2D(448, (1, 3), strides=(1, 1))(X4)
    X4 = Conv2D(512, (3, 1), strides=(1, 1))(X4)
    X4_1 = Conv2D(256, (1, 3), strides=(1, 1))(X4)
    X4_2 = Conv2D(256, (3, 1), strides=(1, 1))(X4)

    X = concatenate([X1, X2, X3_1, X3_2, X4_1, X4_2], axis=channel_axis)
    return X

def inception_reduction_A(X):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    X1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), data_format='channels_first', padding='valid')(X)

    X2 = Conv2D(384, (3, 3), strides=(2,2))(X)

    X3 = Conv2D(192, (1, 1), strides=(1,1))(X)
    X3 = Conv2D(224, (3, 3), strides=(1, 1))(X3)
    X3 = Conv2D(256, (3, 3), strides=(2, 2))(X3)

    X = concatenate([X1, X2, X3], axis=channel_axis)

    return X

def inception_reduction_B(X):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    X1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), data_format='channels_first', padding='valid')(X)

    X2 = Conv2D(192, (1, 1), strides=(1,1))(X)
    X2 = Conv2D(192, (3, 3), strides=(2, 2), padding='valid')(X2)

    X3 = Conv2D(256, (1, 1), strides=(1,1))(X)
    X3 = Conv2D(256, (7, 1), strides=(1, 1))(X3)
    X3 = Conv2D(320, (1, 7), strides=(1, 1))(X3)
    X3 = Conv2D(320, (3, 3), strides=(2, 2), padding = 'valid')(X3)

    X = concatenate([X1, X2, X3], axis=channel_axis)

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
    out = Conv2D(3, (3, 3), activation='relu', padding='same', name='finaloutput')(outputs)

    return out


def Super_resolution(X):

    layer_filters = [(32,1),(32,3),(32,5),(32,7)]
    x = X
    output_layer = []

    x = Conv2D(filters=64,
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

    out = Conv2D(3, (3,3), activation='relu', padding='same', name ='finaloutput')(avg_output)

    return out


def Stem_model(X):
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


def Inception_detail(X):
    #params
    num_classes = 1000
    dropout_rate = 0.2

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

def Model_mixed(input_shape):

    X_input = Input(input_shape, name='model_input')

    #AUTOENCODER
    X = Autoencoder(X_input)
    autoencoder = Model(X_input,
                        X(X_input), name='AE')
    autoencoder.compile(loss='mse', optimizer='adam')

    #SUPER_RESOLUTION
    X = Super_resolution(X_input)
    super_resolution = Model(X_input,
                             X(X_input), name='SUPresol')
    super_resolution.compile(loss='mse', optimizer='adam')

    #INCEPTION
    X = Stem_mode(X_input)
    #stem_model = Model(X_input, X,name='stem_model')
    #stem_model.compile(loss='mse', optimizer='adam')

    X = inception_detail(X)
    inception_detail = Model(inputs=X_input,
                             outputs=X(X_input), name='inception_Model')
    inception_detail.compile(loss='mse', optimizer='adam')

    #FINAL_MODELING
    model = Model(inputs=X_input,
                  outputs=inception_detail(super_resolution(autoencoder(X_input))), name='inception_Model')
    model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    model.summary()
    return model

