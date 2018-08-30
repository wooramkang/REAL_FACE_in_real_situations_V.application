import os
import glob
import numpy as np
from numpy import genfromtxt
import cv2
from random import shuffle
import keras
from keras.models import Model


def Img_load(image_path=""):
    x_data = []
    y_data = []

    # glob.glob("images/*"):
    for file in glob.glob(image_path+"/*"):
        img_ = cv2.imread(file, 1)
        x_data.append(img_)

        identity = os.path.splitext(os.path.basename(file))[0]
        identity = str(identity).split('_')[0]
        y_data.append(identity)

    return x_data, y_data


def Data_shuffle(x_data, y_data):
    #params
    train_test_ratio = 0.7

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    data_len = len(x_data)
    train_len = int(data_len * train_test_ratio)

    train_list = shuffle(range(train_len))
    test_list = shuffle(range(train_len, data_len))

    for i in train_list:
        x_train.append(x_data[i])
        y_train.append(y_data[i])

    for i in test_list:
        x_test.append(x_data[i])
        y_test.append(y_data[i])

    return x_train, y_train, x_test, y_test


def Weight_load(model, weights_path):
    model = Model()
    model.load_weights(weights_path)
    return model



