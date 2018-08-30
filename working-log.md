     to-do list
     
        1. for making embedding
            inception-v4
            triplet-loss
            + etc 
            => my work = Facenet - inception-v2 + inception-v4 -stn + stn else
        
        2. for denosing
            denosing autoencoder
            
        3. for making resolution clear
            super-resolution
            especially, ESRCNN
            
        4. for removing lights
            CLAHE
            
        5. for twisted face
            affine transform
                   
                   
     doing list
     
        1. Facenet - inception-v2 + inception-v4 (2018.08.28 ~)
        
        
    X1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(X)
    X1 = Conv2D(256, (1, 1), strides=(1, 1))(X1)

    X2 = Conv2D(256, (1, 1), strides=(1,1))(X)

    X3 = Conv2D(384, (1, 1), strides=(1,1))(X)
    X3_1 = Conv2D(256, (3, 1), strides=(1, 1))(X3)
    X3_2 = Conv2D(256, (1, 3), strides=(1, 1))(X3)

    X4 = Conv2D(384, (1, 1), strides=(1,1))(X)
    X4 = Conv2D(448, (1, 3), strides=(1, 1))(X4)
    X4 = Conv2D(512, (3, 1), strides=(1, 1))(X4)
    X4_1 = Conv2D(256, (1, 3), strides=(1, 1))(X4)
    X4_2 = Conv2D(256, (3, 1), strides=(1, 1))(X4)

    X = concatenate([X1, X2, X3_1, X3_2, X4_1, X4_2], axis=channel_axis)