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
    
    
    
            
errors

    i coded case 1. by myself, but there was a error about matching sizes of tensors
        
    case 1.
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
    
    case 2. is well-known codes
    
    case 2.
    branch_0 = conv2d_bn(X, 256, 1, 1)

    branch_1 = conv2d_bn(X, 384, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = concatenate([branch_10, branch_11], axis=channel_axis)


    branch_2 = conv2d_bn(X, 384, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = concatenate([branch_20, branch_21], axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(X)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1)

    X = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
