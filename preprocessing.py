import preprocessing.histogram_equalization as hist


def removing_light(x_train=None, x_test=None):
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


def make_embedding(x_train=None, x_test=None):
    input_shape = (1, 1, 3)

    if x_train is not None:
        img_rows = x_train.shape[1]
        img_cols = x_train.shape[2]
        channels = x_train.shape[3]
        x_train = x_train.astype('float32') / 255
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, 3)
    if x_test is not None:
        img_rows = x_test.shape[1]
        img_cols = x_test.shape[2]
        channels = x_test.shape[3]
        x_test = x_test.astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, 3)

    return x_train, x_test, input_shape

def affine_trainsform(x_train=None, x_test=None):
    if x_train is not None:
        x_train_prime = []
        for _img in x_train:
            x_train_prime.append(_img)
        x_train = np.array(x_train_prime)
    else:
        pass

    if x_test is not None:
        x_test_prime = []
        for _img in x_test:
            x_test_prime.append(_img)

        x_test = np.array(x_test_prime)
    else:
        pass

    return x_train, x_test

