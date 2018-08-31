from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath((os.path.dirname(__file__)))+'/preprocessing')
print(os.path.abspath((os.path.dirname(__file__)))+'/preprocessing')
import histogram_equalization as hist
import detect_landmarks_plus_affineTransform as affine

def Removing_light(x_train=None, x_test=None):

    if x_train is not None:
        x_train_prime = []
        for _img in x_train:
            _img = hist.preprocessing_hist(_img)
            x_train_prime.append(_img)
        x_train = np.array(x_train_prime)
    else:
        pass

    if x_test is not None:
        x_test_prime = []
        for _img in x_test:
            _img = hist.preprocessing_hist(_img)
            x_test_prime.append(_img)

        x_test = np.array(x_test_prime)
    else:
        pass

    return x_train, x_test


def Make_embedding(x_train=None, x_test=None):
    if x_train is not None:
        x_train = x_train.astype('float32') / 255

    if x_test is not None:
        x_test = x_test.astype('float32') / 255

    return x_train, x_test


def Affine_transform(x_train=None, y_train=None, x_test=None, y_test=None):

    if x_train is not None:
        x_train_prime = []
        y_train_prime = []
        i = 0

        for _img in x_train:
            x_train_prime.append(_img)
            y_train_prime.append(y_train[i])

            _, _img = affine.make_transformed_faceset(_img)
            x_train_prime.append(_img)
            y_train_prime.append(y_train[i])
            i = i+1

        x_train = np.array(x_train_prime)
        y_train = np.array(y_train_prime)

    else:
        pass

    if x_test is not None:
        x_test_prime = []
        y_test_prime = []
        i = 0

        for _img in x_test:
            x_test_prime.append(_img)
            y_test_prime.append(y_test[i])

            _, _img = affine.make_transformed_faceset(_img)
            x_test_prime.append(_img)
            y_test_prime.append(y_test[i])
            i = i+1

        x_test = np.array(x_test_prime)
        y_test = np.array(y_test_prime)

    else:
        pass

    return x_train, y_train, x_test, y_test