from Model import *
from DataWeight_load import *
from preprocessing import *
from Validation import *
import time


def train():
    # params
    weights_path = "/home/rd/recognition_reaserch/FACE/inception_v4+super_resolution+Face-align+denoisingAE+affineTransform/saved_models/REALFACE_final_facenn.h5"
    #img_path = "/home/rd/recognition_reaserch/FACE/Dataset/lfw/"
    #img_path = "/home/rd/recognition_reaserch/FACE/Dataset/VGGFace2/raw/"
    img_path = "/home/rd/recognition_reaserch/FACE/Dataset/integration/"
    img_size = 96  # target size
    num_classes = 64

    # DATA LOAD
    x_data, y_data = Img_load(img_path, img_size)
    x_train, y_train, x_test, y_test = Data_split(x_data, y_data)

    input_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
    x_train, x_test = Make_embedding(x_train, x_test)
    y_train_ans, y_train_embed = split_embed_groundtruth(y_train)
    y_test_ans, y_test_embed = split_embed_groundtruth(y_test)
    '''
    #DATA PREPROCESSING
    x_train, y_train, x_test, y_test = Affine_transform(x_train, y_train, x_test, y_test)
    x_train, x_test = Removing_light(x_train, x_test)
    x_train, x_test, input_shape = Make_embedding(x_train, x_test)
    '''
    '''
    written by wooram kang 2018.08.30
    img size minimum => 155 * 155
    '''
    # MAKE LEARNING MODEL
    # input_shape = (3, 155, 155)

    # model = Model_mixed(input_shape, num_classes)
    model = simpler_face_NN_residualnet(input_shape, num_classes)
    model_hint = hint_learn(input_shape, num_classes)

    '''
    written by wooramkang 2018.08.30
    numbers of params in networks

    shape / classes

    3 160 160 / 500
    ==================================================================================================
    Total params: 39,479,034
    Trainable params: 39,420,922
    Non-trainable params: 58,112

    3 160 160 / 1500
    ==================================================================================================
    Total params: 41,016,034
    Trainable params: 40,957,922
    Non-trainable params: 58,112
    __________________________________________________________________________________________________

    3 300 300 / 500
    ==================================================================================================
    Total params: 44,261,874
    Trainable params: 44,202,082
    Non-trainable params: 59,792
    ___________________________________________
    3 96 96 / 128

    15,000,0000_______________________________________________________
    there is no pretrained-weights

    '''
    '''
    try:
        model.load_weights(weights_path)
    except:
        print("there is no pretained-model for teacher-net")
    '''
    model.load_weights(weights_path)
    # SAVE MODEL ON LEARNING
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    # model_name = 'REALFACE_model_trippletloss_final.{epoch:03d}.h5'
    model_name = 'REALFACE_final_facenn.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)
    # OPTIONAL
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   verbose=1,
                                   min_lr=0.5e-6)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    # callbacks = [lr_reducer, early, checkpoint]
    callbacks = [lr_reducer, checkpoint]

    # TRAIN
    '''
    model.fit(x_train, y_train_embed,
              validation_data=(x_test, y_test_embed),
              epochs=15,
              batch_size=5,
              callbacks=callbacks)
    '''
    # TEST
    start_time = time.time()
    predict_test = model.predict(x_test)
    fin_time = time.time()

    print(predict_test)

    Validation(model, y_test, y_test_ans, y_test_embed, predict_test)

    print("======")
    print(fin_time - start_time)

    weights_path = "/home/rd/recognition_reaserch/FACE/inception_v4+super_resolution+Face-align+denoisingAE+affineTransform/saved_models/hintlearn_model_trippletloss_final.h5"
    #model_hint = Distilling_neural_networks(input_shape, num_classes)
    '''
    try:
        model_hint = Weight_load(model_hint, weights_path)
    except:
        print("there is no pretrained-weights")
    '''

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'hintlearn_model_trippletloss_final.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)
    # OPTIONAL
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   verbose=1,
                                   min_lr=0.5e-6)
    # early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    callbacks = [lr_reducer, checkpoint]

    predict_test_valid = predict_test
    predict_test = model.predict(x_train)

    model_hint.fit(x_train, predict_test,
                   validation_data=(x_test, predict_test_valid),
                   epochs=25,
                   batch_size=20,
                   callbacks=callbacks)

    start_time = time.time()
    predict_test = model_hint.predict(x_test)
    fin_time = time.time()

    Validation(model, y_test, y_test_ans, y_test_embed, predict_test)
    print("==================")
    print(predict_test)
    print("======")
    print(start_time)
    print(fin_time)
    print(fin_time - start_time)


if __name__ == "__main__":
    train()
