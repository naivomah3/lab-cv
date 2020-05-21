import os
from itertools import accumulate

from keras.models import Model  # Functional API
from keras.layers import (
    Conv2D, Input, MaxPooling2D,
    Conv2DTranspose, Lambda, ReLU,
    concatenate, BatchNormalization, DepthwiseConv2D)
from keras.optimizers import Adam

from src.metrics import (dice, jaccard)
from src.engine import scale_input

class DWUNet:
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


    def depthwise_block(self, layer, filter, stride=1):
        ''' Depthwise Separable convolution operation followed by batch-normalization and ReLU for non-linearity'''

        # Depthwise Convolution
        layer = DepthwiseConv2D((3, 3), strides=stride, padding='same', kernel_initializer=self.kernel_init)(layer)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)
        # Pointwise Convolution
        layer = Conv2D(filter, (1, 1), padding='same', kernel_initializer=self.kernel_init)(layer)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)

        return layer



    # Build U-Net: original paper
    def build(self):
        # ======================================== INPUT ==========================================
        inBlock = Input(shape=(self.input_h, self.input_w, 3), dtype='float32')
        # Lambda layer: scale input before feeding to the network
        inScaled = Lambda(lambda x: scale_input(x))(inBlock)

        # ======================================== ENCODER ========================================
        # Input Block
        inConvB = Conv2D(32, (3, 3), padding='same')(inScaled)
        inConvB = BatchNormalization()(inConvB)
        inConvB = ReLU()(inConvB)

        # Block 1
        convB1d = self.depthwise_block(inConvB, 64)
        convB1d = self.depthwise_block(inConvB, 64)
        poolB1d = MaxPooling2D(pool_size=(2, 2))(convB1d)
        # block 2
        convB2d = self.depthwise_block(poolB1d, 128)
        convB2d = self.depthwise_block(convB2d, 128)
        poolB2d = MaxPooling2D(pool_size=(2, 2))(convB2d)
        # block 3
        convB3d = self.depthwise_block(poolB2d, 256)
        convB3d = self.depthwise_block(convB3d, 256)
        poolB3d = MaxPooling2D(pool_size=(2, 2))(convB3d)
        # block 4
        convB4d = self.depthwise_block(poolB3d, 512)
        convB4d = self.depthwise_block(convB4d, 512)
        poolB4d = MaxPooling2D(pool_size=(2, 2))(convB4d)

        # ===================================== BOTTLENECK ======================================
        convBn = self.depthwise_block(poolB4d, 1024)
        convBn = self.depthwise_block(convBn, 512)

        # ====================================== DECODER =======================================
        # Block 4u
        convB4u = Conv2DTranspose(258, (2, 2), strides=(2, 2), padding='same')(convBn)
        convB4u = concatenate([convB4u, convB4d])
        convB4u = self.depthwise_block(convB4u, 512)
        convB4u = self.depthwise_block(convB4u, 256)
        # Block 3u
        convB3u = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(convB4u)
        convB3u = concatenate([convB3u, convB3d])
        convB3u = self.depthwise_block(convB3u, 256)
        convB3u = self.depthwise_block(convB3u, 128)
        # Block 2u
        convB2u = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(convB3u)
        convB2u = concatenate([convB2u, convB2d])
        convB2u = self.depthwise_block(convB2u, 128)
        convB2u = self.depthwise_block(convB2u, 64)
        # Block 1u
        convB1u = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(convB2u)
        convB1u = concatenate([convB1u, convB1d], axis=3)
        convB1u = self.depthwise_block(convB1u, 64)
        convB1u = self.depthwise_block(convB1u, 64)

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

