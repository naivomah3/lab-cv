import os
from keras.models import Model, load_model
from keras.layers import Conv2D, Input, MaxPooling2D, Conv2DTranspose, Dropout, Lambda, concatenate
from keras.optimizers import Adam
from tensorflow.keras import backend as K

# simple pre-processing
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


# dice as a metric
def dice(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def unet(pretrained=False, model_path=None, n_classes=None, input_h=320, input_w=320, base_exp=5):
    if pretrained:
        if os.path.exists(model_path):
            model = load_model(model_path, custom_objects={'dice': dice, 'preprocess_input': preprocess_input})
            # model.summary()
            return model
        else:
            print(f'Failed to load  the existing model at {model_path}')

    b = base_exp
    i = Input((input_h, input_w, 3))
    # Lambda layer to simple pre-processing element-wise(such as Normalization, ...)
    s = Lambda(lambda x: preprocess_input(x))(i)
    # Block 1
    c1 = Conv2D(2 ** b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(2 ** b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(2 ** (b + 1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(2 ** (b + 1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(2 ** (b + 2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(2 ** (b + 2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(2 ** (b + 3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(2 ** (b + 3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(2 ** (b + 4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(2 ** (b + 4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(2 ** (b + 3), (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(2 ** (b + 3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(2 ** (b + 3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(2 ** (b + 2), (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(2 ** (b + 2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(2 ** (b + 2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(2 ** (b + 1), (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(2 ** (b + 1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(2 ** (b + 1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(2 ** b, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(2 ** b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(2 ** b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    o = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=i, outputs=o, name="unet.model")
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=[dice])
    model.summary()

    return model