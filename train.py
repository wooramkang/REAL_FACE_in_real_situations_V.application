from making_Model import *
from DataWeight_load import *
from preprocessing import *


def train():
    '''
    #params
    weights_path = ""
    img_path = ""

    #DATA LOAD
    x_data, y_data = Img_load(img_path)
    x_train, y_train, x_test, y_test = Data_shuffle(x_data, y_data)

    #DATA PREPROCESSING
    x_train, y_train, x_test, y_test = Affine_transform(x_train, y_train, x_test, y_test)
    x_train, x_test = Removing_light(x_train, x_test)
    x_train, x_test, input_shape = Make_embedding(x_train, x_test)

    #MAKE LEARNING MODEL
    '''
    input_shape = (3, 299, 299)
    model = Model_mixed(input_shape)

#    model = Weight_load(model, weights_path)


if __name__ == "__main__":
    train()

