from Model import *
from DataWeight_load import *
from preprocessing import *
from Validation import *
import time


def train():
    # params
    weights_path = "/home/rd/recognition_reaserch/FACE/inception_v4+super_resolution+Face-align+denoisingAE+affineTransform/saved_models/model.h5"

    img_path = "/home/rd/recognition_reaserch/FACE/Dataset/lfw/"
    img_size = 96 #*96  # target size

    # DATA LOAD
    x_data, y_data = Img_load(img_path, img_size)
    x_train, y_train, x_test, y_test = Data_split(x_data, y_data)
    print(x_train)
    print(y_train)

    input_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
    x_train, x_test = Make_embedding(x_train, x_test)
    y_train_ans, y_train_embed = split_embed_groundtruth(y_train)
    print(y_train_embed)
    print(y_train_ans)
    print(x_train)
    y_test_ans, y_test_embed = split_embed_groundtruth(y_test)
    '''
    #DATA PREPROCESSING DEPENDING ON SITUATIONS
    x_train, y_train, x_test, y_test = Affine_transform(x_train, y_train, x_test, y_test)
    x_train, x_test = Removing_light(x_train, x_test)
    x_train, x_test, input_shape = Make_embedding(x_train, x_test)
    '''
    '''
    written by wooram kang 2018.08.30
    img size minimum => 155 * 155 for object-detection i guess
    
    '''
    # MAKE LEARNING MODEL
    model = Model_mixed(input_shape, 128)
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
    _________________________________________________________
    3 96 96 / 128
    Total params : 14,406,662_________________________________________
    there is no pretrained-weights
    
    2018.09.04
    conditions : 
        1. full networks of inception-v4
        2. skipped networks of it
    '''
    try:
        model = Weight_load(model, weights_path)
    except:
        print("there is no pretrained-weights")

    # SAVE MODEL ON LEARNING
    '''
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'REALFACE_model_trippletloss_final.{epoch:03d}.h5'
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

    #callbacks = [lr_reducer, early, checkpoint]
    callbacks = [lr_reducer, checkpoint]
    
    #TRAIN
    model.fit(x_train, y_train_embed,
              validation_data=(x_test, y_test_embed),
              epochs=20,
              batch_size=3,
              callbacks=callbacks)
    
    '''

    '''
    written by wooramkang 2018. 09.03
    be aware you need TiTan at least so far
    
    2018. 09. 05
    now i try to
    
        1. pruning
        2. KD networks
        3. born-again networks
        4. fitnet / hint net 
    '''

    # TEST

    start_time = time.time()
    predict_test = model.predict(x_test)
    fin_time = time.time()


    #Validation(model, y_test, y_test_ans, y_test_embed, predict_test)
    
    print(predict_test)
    print("======")
    print(start_time)
    print(fin_time)
    print(fin_time - start_time)
    #necessary running time
    '''
        written by wooramkang 2018.09.06
        
        kind of hint-net
        number of params get 1/3 times less than v-original as least 
    '''

    # DISTILLING NN from v-original

    weights_path = "/home/rd/recognition_reaserch/FACE/inception_v4+super_resolution+Face-align+denoisingAE+affineTransform/saved_models/hintlearn_model_trippletloss_final.h5"
    model_hint = Distilling_neural_networks(input_shape, 128)

    try:
        model = Weight_load(model_hint, weights_path)
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
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   verbose=1,
                                   min_lr=0.5e-6)

    callbacks = [lr_reducer, early, checkpoint]
    predict_test = model.predict(x_train)
    model_hint.fit(x_train, predict_test,
                   validation_data=(x_train, predict_test),
                   epochs=20,
                   batch_size=10,
                   callbacks=callbacks)
    '''
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
    '''
    written by wooramkang 2018.09.06
    distilling knowledge_Neural network
    
     the running time of v-original = 60 times of the running time of distilling NN 
    '''
if __name__ == "__main__":
    train()
