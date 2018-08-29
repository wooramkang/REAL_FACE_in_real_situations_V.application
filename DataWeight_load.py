import os
import glob
import numpy as np
from numpy import genfromtxt
import cv2


def img_load(image_path=""):

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


def data_shuffle(x_data, y_data):
    #params
    train_test_ratio = 0.7

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    return x_train, y_train, x_test, y_test


def Weight_load(model):

    return -1