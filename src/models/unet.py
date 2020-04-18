import os

from keras.models import Model  # Functional API
from keras.layers import (
    Conv2D, Input, MaxPooling2D,
    Conv2DTranspose, Lambda,
    concatenate, BatchNormalization)
from keras.optimizers import Adam

from src.metrics import (dice, jaccard)
from src.engine import scale_input

class UNet:
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


    # Build U-Net: original paper
    def build(self):
        # ======================================== INPUT ==========================================
        inBlock = Input(shape=(self.input_h, self.input_w, 3), dtype='float32')
        # Lambda layer: scale input before feeding to the network
        inScaled = Lambda(lambda x: scale_input(x))(inBlock)

        # ======================================== ENCODER ========================================
        # Block 1d
        convB1d = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(inScaled)
        convB1d = BatchNormalization()(convB1d)
        convB1d = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB1d)
        convB1d = BatchNormalization()(convB1d)
        poolB1d = MaxPooling2D(pool_size=(2, 2))(convB1d)
        # Block 2d
        convB2d = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(poolB1d)
        convB2d = BatchNormalization()(convB2d)
        convB2d = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB2d)
        convB2d = BatchNormalization()(convB2d)
        poolB2d = MaxPooling2D(pool_size=(2, 2))(convB2d)
        # Block 3d
        convB3d = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(poolB2d)
        convB3d = BatchNormalization()(convB3d)
        convB3d = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB3d)
        convB3d = BatchNormalization()(convB3d)
        poolB3d = MaxPooling2D(pool_size=(2, 2))(convB3d)
        # Block 4d
        convB4d = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(poolB3d)
        convB4d = BatchNormalization()(convB4d)
        convB4d = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB4d)
        convB4d = BatchNormalization()(convB4d)
        poolB4d = MaxPooling2D(pool_size=(2, 2))(convB4d)

        # ===================================== BOTTLENECK ======================================
        convBn = Conv2D(1024, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(poolB4d)
        convBn = BatchNormalization()(convBn)
        convBn = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convBn)
        convBn = BatchNormalization()(convBn)

        # ====================================== DECODER =======================================
        # Block 4u
        convB4u = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(convBn)
        convB4u = concatenate([convB4u, convB4d])
        convB4u = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB4u)
        convB4u = BatchNormalization()(convB4u)
        convB4u = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB4u)
        convB4u = BatchNormalization()(convB4u)
        # Block 3u
        convB3u = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(convB4u)
        convB3u = concatenate([convB3u, convB3d])
        convB3u = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB3u)
        convB3u = BatchNormalization()(convB3u)
        convB3u = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB3u)
        convB3u = BatchNormalization()(convB3u)
        # Block B2u
        convB2u = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(convB3u)
        convB2u = concatenate([convB2u, convB2d])
        convB2u = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB2u)
        convB2u = BatchNormalization()(convB2u)
        convB2u = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB2u)
        convB2u = BatchNormalization()(convB2u)
        # Block B1u
        convB1u = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(convB2u)
        convB1u = concatenate([convB1u, convB1d], axis=3)
        convB1u = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB1u)
        convB1u = BatchNormalization()(convB1u)
        convB1u = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB1u)
        convB1u = BatchNormalization()(convB1u)

        # ======================================== OUTPUT ==========================================
        if self.n_classes == 2:
            outBlock = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(convB1u)
        else:
            outBlock = Conv2D(self.n_classes, (1, 1), activation='softmax', padding='same')(convB1u)

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

