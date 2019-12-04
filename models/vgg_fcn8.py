import os
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras.models import Model, load_model
from keras.layers import Conv2D, Input, MaxPooling2D, Conv2DTranspose, Dropout, Lambda, Add
from keras.optimizers import Adam
from tensorflow.keras import backend as K

# simple pre-processing
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

# dice as a monitored metric
def dice(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def vgg_fcn8s(pretrained=False, model_path=None, n_classes=None, input_h=320, input_w=320, base=6):
    # Load pre-trained model instead
    if pretrained:
        if os.path.exists(model_path):
            model = load_model(model_path, custom_objects={'dice': dice, 'preprocess_input': preprocess_input})
            #model.summary()
            return model
        else:
            print(f'Failed to load  the existing model at {model_path}')

    b = base
    i = Input((input_h, input_w, 3))
    # Lambda layer to simple pre-process pixel values(such as Normalization, ...)
    s = Lambda(lambda x: preprocess_input(x))(i)

    ## Block 1
    x = Conv2D(2**b, (3, 3), activation='elu', padding='same', name='block1_conv1')(s)
    x = Conv2D(2**b, (3, 3), activation='elu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # Block 2
    x = Conv2D(2**(b+1), (3, 3), activation='elu', padding='same', name='block2_conv1')(x)
    x = Conv2D(2**(b+1), (3, 3), activation='elu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv1')(x)
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv2')(x)
    x = Conv2D(2**(b+2), (3, 3), activation='elu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    pool3 = x

    # Block 4
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv1')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv2')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block4_conv3')(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv1')(pool4)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv2')(x)
    x = Conv2D(2**(b+3), (3, 3), activation='elu', padding='same', name='block5_conv3')(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    conv6 = Conv2D(2048 , (7, 7) , activation='elu' , padding='same', name="conv6")(pool5)
    conv6 = Dropout(0.5)(conv6)
    conv7 = Conv2D(2048 , (1, 1) , activation='elu' , padding='same', name="conv7")(conv6)
    conv7 = Dropout(0.5)(conv7)

    pool4_n = Conv2D(n_classes, (1, 1), activation='elu', padding='same')(pool4)
    u2 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    # skip connection between pool_4(after 1x1 convolution) & conv7(upsampled 2 times)
    u2_skip = Add()([pool4_n, u2])


    pool3_n = Conv2D(n_classes, (1, 1), activation='elu', padding='same')(pool3)
    u4 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(u2_skip)
    # skip connection between pool_3(after 1x1 convolution) & the result of the previous upsampling(again upsampled 4 times)
    u4_skip = Add()([pool3_n, u4])

    o = Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), padding='same',
                        activation='softmax')(u4_skip)


    model = Model(inputs=i, outputs=o, name='vgg_fcn8s.model')
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=[dice])

    model.summary()

    return model