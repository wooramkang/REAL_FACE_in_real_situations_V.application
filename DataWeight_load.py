from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
from numpy import genfromtxt
import cv2
from random import shuffle
import keras
from keras.models import Model


def Img_load(image_path, img_szie ):
    x_data = []
    y_data = []

    # glob.glob("images/*"):
    folders = os.listdir(image_path)
    for name in folders:
        print(name)
        for file in glob.glob(image_path+name+"/*"):
            img_ = cv2.imread(file)
            img_ = cv2.resize(img_, (img_szie, img_szie))
            #print(img_.shape)
            img_ = np.transpose(img_, (2, 0, 1))
            print(img_.shape)
            x_data.append(img_)
            identity = os.path.splitext(os.path.basename(file))[0]
            #identity = str(identity).split('_')[0]
            #y_data.append(identity)
            y_data.append(name)

    print(len(x_data))
    print(len(y_data))
    print(len(folders))
    print(y_data[:100])
    print("==============")
    t = np.array(x_data)
    x_data = t
    t = np.array(y_data)
    y_data = t
    print(x_data.shape[1])
    print(x_data.shape[2])
    print(x_data.shape[3])
    return x_data, y_data


def split_embed_groundtruth(raw_data):
    ans_set = []

    num = 0
    for i in range(raw_data):
        if i not in ans_set:
            ans_set.append(i)

    dist = []

    for j in raw_data:
        for i, ans in enumerate(ans_set):
            if j == ans:
                dist.append(i)
                break

    return ans_set, np.array(dist)


def Data_shuffle(x_data, y_data):
    #params
    train_test_ratio = 0.7

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    data_len = len(x_data)
    train_len = int(data_len * train_test_ratio)

    train_list = range(train_len)
    test_list = range(train_len, data_len)

    for i in train_list:
        x_train.append(x_data[i])
        y_train.append(y_data[i])

    for i in test_list:
        x_test.append(x_data[i])
        y_test.append(y_data[i])

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def Weight_load(model, weights_path):
    model = Model()
    model.load_weights(weights_path)
    return model



