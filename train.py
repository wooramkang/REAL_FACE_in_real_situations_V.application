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
    '''

    # MAKE LEARNING MODEL
    input_shape = (3, 196, 196)
    model = Model_mixed(input_shape,5750)

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
    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=30,
              batch_size=32,
              callbacks=callbacks)
    #TEST
    predict_test = model.predict(x_test)

    #TEST RESULT
    imgs = predict_test[:100]
    print(imgs.shape)
    imgs = (imgs * 255).astype(np.uint8)
    imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    Image.fromarray(imgs).save('saved_images/sumof_img_gen.png')

if __name__ == "__main__":
    train()
