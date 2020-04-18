import os
from keras.models import Model  # Functional API
from keras.layers import (
    Conv2D, Input, MaxPooling2D,
    Conv2DTranspose, Lambda,
    concatenate, BatchNormalization,
    Dropout, Add)
from keras.optimizers import Adam

from src.metrics import (dice, jaccard)
from src.engine import scale_input


class FCN_8s:
    def __init__(self,
                 pre_trained=False,  # if True, set weights_path
                 weights_path=None,  # full-path to the pre-trained models_weights
                 n_classes=None,
                 input_h=None,
                 input_w=None,
                 activation='elu',
                 kernel_init='he_normal',
                 model_name=None
                 ):
        self.pre_trained = pre_trained
        self.weights_path = weights_path
        self.n_classes = n_classes
        self.input_h = input_h
        self.input_w = input_w
        self.activation = activation
        self.kernel_init = kernel_init
        self.model_name = model_name

    # Build FCN8s: original paper
    def build(self):
        inBlock = Input(shape=(self.input_h, self.input_w, 3), dtype='float32')
        # Lambda layer: scale input before feeding to the network
        inScaled = Lambda(lambda x: scale_input(x))(inBlock)

        # Block 1
        x = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(inScaled)
        x = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),)(x)
        f1 = x

        # Block 2
        x = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(x)
        x = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        f2 = x

        # Block 3
        x = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(x)
        x = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(x)
        x = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        pool3 = x

        # Block 4
        x = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(pool3)
        x = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(x)
        x = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        pool4 = x

        # Block 5
        x = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(pool4)
        x = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(x)
        x = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        pool5 = x

        conv6 = Conv2D(2048, (7, 7), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(pool5)
        conv7 = Conv2D(2048, (1, 1), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(conv6)

        pool4_n = Conv2D(self.n_classes, (1, 1), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(pool4)
        u2 = Conv2DTranspose(self.n_classes, kernel_size=(2, 2), strides=(2, 2), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(conv7)
        # skip connection between pool_4(after 1x1 convolution) & conv7(upsampled 2 times)
        u2_skip = Add()([pool4_n, u2])

        pool3_n = Conv2D(self.n_classes, (1, 1), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(pool3)
        u4 = Conv2DTranspose(self.n_classes, kernel_size=(2, 2), strides=(2, 2), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(u2_skip)
        # skip connection between pool_3(after 1x1 convolution) & the result of the previous upsampling(again upsampled 4 times)
        u4_skip = Add()([pool3_n, u4])

        # Output layer
        outBlock = Conv2DTranspose(self.n_classes, kernel_size=(8, 8), strides=(8, 8), padding='same', activation='softmax')(u4_skip)

        # Create model
        model = Model(inputs=inBlock, outputs=outBlock, name=self.model_name)
        model.compile(optimizer=Adam(),
                      loss="categorical_crossentropy",
                      metrics=[dice, jaccard, ]
                      )

        # Load models_weights if pre-trained
        if self.pre_trained:
            if os.path.exists(self.weights_path):
                model.load_weights(self.weights_path)
            else:
                raise Exception(f'Failed to load weights at {self.weights_path}')

        return model