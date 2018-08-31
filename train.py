from Model import *
from DataWeight_load import *
from preprocessing import *


def train():

    #params
    weights_path = "/saved_models/"
    img_path = "/home/rd/recognition_reaserch/FACE/Dataset/lfw/"
    img_size = 299 #target size

    #DATA LOAD
    x_data, y_data = Img_load(img_path, img_size)
    x_train, y_train, x_test, y_test = Data_split(x_data, y_data)
    print(x_train)
    print(y_train)

    input_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
    x_train, x_test = Make_embedding(x_train, x_test)
    y_train_ans, y_train_embed = split_embed_groundtruth(y_train)
    print(y_train_embed)
    print(y_train_ans)

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
    #input_shape = (3, 155, 155)

    model = Model_mixed(input_shape, 6000)
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
    __________________________________________________________________________________________________
    there is no pretrained-weights
    
    '''
    try:
        model = Weight_load(model, weights_path)
    except:
        print("there is no pretrained-weights")


    #SAVE MODEL ON LEARNING
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'REALFACE_model.{epoch:03d}.h5'
    if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)
    #OPTIONAL
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   verbose=1,
                                   min_lr=0.5e-6)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    callbacks = [early, lr_reducer, checkpoint]

    #TRAIN
    model.fit(x_train, y_train_embed,
              validation_data=(x_test, y_test_embed),
              epochs=20,
              batch_size=15,
              callbacks=callbacks)

    #TEST
    predict_test = model.predict(x_train)
    print(predict_test)
    predicted_set = []
    count_cor = 0
    k = 0
    for i in range(predict_test):
        predicted_set.append(y_train_ans[i])
        if y_train_ans[i] == y_train[k]:
            count_cor = count_cor + 1
        k=k+1
    print(predicted_set)
    print(y_train)
    print(str(count_cor/len(y_train)))

if __name__ == "__main__":
    train()

