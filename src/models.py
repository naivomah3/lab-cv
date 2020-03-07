import os
import numpy as np
import tensorflow as tf
from keras.models import Model  # Functional API
from keras import layers
from keras.layers import (
    Conv2D, Input, MaxPooling2D,
    Conv2DTranspose, Dropout, Lambda, Concatenate,
    concatenate, BatchNormalization, Activation, AveragePooling2D,
    Reshape, ConvLSTM2D, ZeroPadding2D, DepthwiseConv2D, Add)
from keras.optimizers import Adam
from keras.activations import relu
from keras.regularizers import l2
from keras import backend as K

from src.metrics import (dice, jaccard)
from src.engine import scale_input
from src.utils import relu6, make_divisible


class UNet:
    def __init__(self,
                 pre_trained=False,  # if True, set weights_path
                 weights_path=None,  # full-path to the pre-trained weights
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
        inBlock = Input(shape=(self.input_h, self.input_w, 3), dtype='float32')
        # Lambda layer: scale input before feeding to the network
        inScaled = Lambda(lambda x: scale_input(x))(inBlock)
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
        # Bottleneck
        convBn = Conv2D(1024, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(poolB4d)
        convBn = BatchNormalization()(convBn)
        convBn = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convBn)
        convBn = BatchNormalization()(convBn)
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

        # Output layer
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

        # Load weights if pre-trained
        if self.pre_trained:
            if os.path.exists(self.weights_path):
                model.load_weights(self.weights_path)
            else:
                raise Exception(f'Failed to load weights at {self.weights_path}')

        return model


class BCD_UNet_D3:
    def __init__(self,
                 pre_trained=False,  # if True, set weights_path
                 weights_path=None,  # full-path to the pre-trained weights
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

    def build(self):
        # Compile model
        inBlock = Input(shape=(self.input_h, self.input_w, 3), dtype='float32')
        # Lambda layer: scale input before feeding to the network
        inScaled = Lambda(lambda x: scale_input(x))(inBlock)
        # =============================================== ENCODING ==================================================
        # Block 1d
        convB1d = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(inScaled)
        convB1d = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB1d)
        poolB1d = MaxPooling2D(pool_size=(2, 2))(convB1d)
        # Block 2d
        convB2d = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(poolB1d)
        convB2d = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB2d)
        poolB2d = MaxPooling2D(pool_size=(2, 2))(convB2d)
        # Block 3d
        convB3d = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(poolB2d)
        convB3d = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB3d)
        #dropB3d = Dropout(0.5)(convB3d)
        poolB3d = MaxPooling2D(pool_size=(2, 2))(convB3d)

        # =============================================== BOTTLENECK =================================================
        # Bottleneck - Block D1
        convBnd1 = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(poolB3d)
        convBnd1 = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convBnd1)
        dropBnd1 = Dropout(0.5)(convBnd1)
        # Bottlenbeck - Block D2
        convBnd2 = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convBnd1)
        convBnd2 = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convBnd2)
        dropBnd2 = Dropout(0.5)(convBnd2)
        # Bottlenbeck - Block D3
        merge_d1_d2 = concatenate([convBnd1, convBnd2], axis=3)
        convBnd3 = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(merge_d1_d2)
        convBnd3 = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convBnd3)
        dropBnd3 = Dropout(0.5)(convBnd3)

        # =============================================== DECODING ==================================================
        # Block 4u
        convB4u = Conv2DTranspose(256, kernel_size=2, strides=2, kernel_initializer=self.kernel_init, padding='same')(convBnd3)
        convB4u = BatchNormalization(axis=3)(convB4u)
        convB4u = Activation(self.activation)(convB4u)
        dropB3d = Reshape(target_shape=(1, np.int32(self.input_h/4), np.int32(self.input_w/4), 256))(convB3d) # just to make sure about shape, but I think the size is already okay :P
        convB4u = Reshape(target_shape=(1, np.int32(self.input_h/4), np.int32(self.input_w/4), 256))(convB4u)
        merge_3d_4u = concatenate([dropB3d, convB4u], axis=1)
        merge_3d_4u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=self.kernel_init)(merge_3d_4u)
        convB4u = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(merge_3d_4u)
        convB4u = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB4u)

        # Block 3u
        convB3u = Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=self.kernel_init, padding='same')(convB4u)
        convB3u = BatchNormalization(axis=3)(convB3u)
        convB3u = Activation(self.activation)(convB3u)
        convB2d = Reshape(target_shape=(1, np.int32(self.input_h/2), np.int32(self.input_w/2), 128))(convB2d)
        convB3u = Reshape(target_shape=(1, np.int32(self.input_h/2), np.int32(self.input_w/2), 128))(convB3u)
        merge_2d_3u = concatenate([convB2d, convB3u], axis=1)
        merge_2d_3u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=self.kernel_init)(merge_2d_3u)
        convB3u = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(merge_2d_3u)
        convB3u = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB3u)

        # Block B2u
        convB2u = Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=self.kernel_init, padding='same')(convB3u)
        convB2u = BatchNormalization(axis=3)(convB2u)
        convB2u = Activation(self.activation)(convB2u)
        convB1d = Reshape(target_shape=(1, self.input_h, self.input_w, 64))(convB1d)
        convB2u = Reshape(target_shape=(1, self.input_h, self.input_w, 64))(convB2u)
        merge_1d_2u = concatenate([convB1d, convB2u], axis=1)
        merge_1d_2u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=self.kernel_init)(merge_1d_2u)
        convB2u = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(merge_1d_2u)
        convB2u = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB2u)

        # ================================================== OUTPUT =====================================================
        # Output layer
        if self.n_classes == 2:
            outBlock = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(convB2u)
        else:
            outBlock = Conv2D(self.n_classes, (1, 1), activation='softmax', padding='same')(convB2u)

        # Create model
        model = Model(inputs=inBlock, outputs=outBlock, name=self.model_name)
        model.compile(optimizer=Adam(),
                      loss="categorical_crossentropy",
                      metrics=[dice, jaccard, ]
                      )

        # Load weights if pre-trained
        if self.pre_trained:
            if os.path.exists(self.weights_path):
                model.load_weights(self.weights_path)
            else:
                raise Exception(f'Failed to load weights at {self.weights_path}')

        return model


class BCD_UNet_D1:
    def __init__(self,
                 pre_trained=False,  # if True, set weights_path
                 weights_path=None,  # full-path to the pre-trained weights
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

    def build(self):
        # Compile model
        inBlock = Input(shape=(self.input_h, self.input_w, 3), dtype='float32')
        # Lambda layer: scale input before feeding to the network
        inScaled = Lambda(lambda x: scale_input(x))(inBlock)
        # =============================================== ENCODING ==================================================
        # Block 1d
        convB1d = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(inScaled)
        convB1d = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB1d)
        poolB1d = MaxPooling2D(pool_size=(2, 2))(convB1d)
        # Block 2d
        convB2d = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(poolB1d)
        convB2d = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB2d)
        poolB2d = MaxPooling2D(pool_size=(2, 2))(convB2d)
        # Block 3d
        convB3d = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(poolB2d)
        convB3d = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB3d)
        dropB3d = Dropout(0.5)(convB3d)
        poolB3d = MaxPooling2D(pool_size=(2, 2))(convB3d)

        # =============================================== BOTTLENECK =================================================
        # Bottleneck - Block D1
        convBnd1 = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(poolB3d)
        convBnd1 = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convBnd1)
        dropBnd1 = Dropout(0.5)(convBnd1)

        # =============================================== DECODING ==================================================
        # Block 4u
        convB4u = Conv2DTranspose(256, kernel_size=2, strides=2, kernel_initializer=self.kernel_init, padding='same')(dropBnd1)
        convB4u = BatchNormalization(axis=3)(convB4u)
        convB4u = Activation(self.activation)(convB4u)
        dropB3d = Reshape(target_shape=(1, np.int32(self.input_h/4), np.int32(self.input_w/4), 256))(dropB3d) # just to make sure about shape, but I think the size is already okay :P
        convB4u = Reshape(target_shape=(1, np.int32(self.input_h/4), np.int32(self.input_w/4), 256))(convB4u)
        merge_3d_4u = concatenate([dropB3d, convB4u], axis=1)
        merge_3d_4u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=self.kernel_init)(merge_3d_4u)
        convB4u = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(merge_3d_4u)
        convB4u = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB4u)

        # Block 3u
        convB3u = Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=self.kernel_init, padding='same')(convB4u)
        convB3u = BatchNormalization(axis=3)(convB3u)
        convB3u = Activation(self.activation)(convB3u)
        convB2d = Reshape(target_shape=(1, np.int32(self.input_h/2), np.int32(self.input_w/2), 128))(convB2d)
        convB3u = Reshape(target_shape=(1, np.int32(self.input_h/2), np.int32(self.input_w/2), 128))(convB3u)
        merge_2d_3u = concatenate([convB2d, convB3u], axis=1)
        merge_2d_3u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=self.kernel_init)(merge_2d_3u)
        convB3u = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(merge_2d_3u)
        convB3u = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB3u)

        # Block B2u
        convB2u = Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=self.kernel_init, padding='same')(convB3u)
        convB2u = BatchNormalization(axis=3)(convB2u)
        convB2u = Activation(self.activation)(convB2u)
        convB1d = Reshape(target_shape=(1, self.input_h, self.input_w, 64))(convB1d)
        convB2u = Reshape(target_shape=(1, self.input_h, self.input_w, 64))(convB2u)
        merge_1d_2u = concatenate([convB1d, convB2u], axis=1)
        merge_1d_2u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=self.kernel_init)(merge_1d_2u)
        convB2u = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(merge_1d_2u)
        convB2u = Conv2D(64, (3, 3), activation=self.activation, kernel_initializer=self.kernel_init, padding='same')(convB2u)

        # ================================================== OUTPUT =====================================================
        # Output layer
        if self.n_classes == 2:
            outBlock = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(convB2u)
        else:
            outBlock = Conv2D(self.n_classes, (1, 1), activation='softmax', padding='same')(convB2u)

        # Create model
        model = Model(inputs=inBlock, outputs=outBlock, name=self.model_name)
        model.compile(optimizer=Adam(),
                      loss="categorical_crossentropy",
                      metrics=[dice, jaccard, ]
                      )

        # Load weights if pre-trained
        if self.pre_trained:
            if os.path.exists(self.weights_path):
                model.load_weights(self.weights_path)
            else:
                raise Exception(f'Failed to load weights at {self.weights_path}')

        return model


class FC_DenseNet:
    def __init__(self,
                 pre_trained=False,  # if True, set weights_path
                 weights_path=None,  # full-path to the pre-trained weights
                 n_classes=None,
                 input_h=None,
                 input_w=None,
                 activation='relu',
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
        self.filters = 48   # this value is the number of filter of the first convolution block --> this will be updated at the beginning of each dense block
        # Local setting
        if model_name == 'fcn_densenet_56':
            self.n_pool = 5    # Corresponds to number of dense block --> number of transition up/down
            self.growth_rate = 12  # used to calculate the number of filters for each layer in each dense block/transition up/down --> filters = nb_layers * growth_rate
            self.n_layers_per_dense_block = 4  # number of layer in each dense block
        elif model_name == 'fcn_densenet_67':
            self.n_pool = 5   # Corresponds to number of dense block --> number of transition up/down
            self.growth_rate = 16  # used to calculate the number of filters for each layer in each dense block/transition up/down --> filters = nb_layers * growth_rate
            self.n_layers_per_dense_block = 4  # number of layer in each dense block
        elif model_name == 'fcn_densenet_103':
            self.n_pool = 5   # Corresponds to number of dense block --> number of transition up/down
            self.growth_rate = 16  # used to calculate the number of filters for each layer in each dense block/transition up/down --> filters = nb_layers * growth_rate
            self.n_layers_per_dense_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]  # number of layer in each dense block
        else:
            pass
            # raise Exception(f"Model name not understood `{model_name}`")


    def dense_block(self, layer, n_filters):
        layer = BatchNormalization()(layer)
        layer = Activation(self.activation)(layer)
        layer = Conv2D(n_filters, kernel_size=(3, 3), kernel_initializer=self.kernel_init, padding='same', data_format='channels_last')(layer)
        layer = Dropout(0.2)(layer)
        return layer

    def transition_down(self, layer, n_filters):
        layer = BatchNormalization()(layer)
        layer = Activation(self.activation)(layer)
        layer = Conv2D(n_filters, kernel_size=(1, 1), kernel_initializer=self.kernel_init, padding='same', data_format='channels_last')(layer)
        layer = Dropout(0.2)(layer)
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last')(layer)
        return layer

    def transition_up(self, skip_connection, block_to_upsample, n_filters):
        layer = concatenate(block_to_upsample)
        layer = Conv2DTranspose(n_filters, kernel_size=(3, 3), strides=(2, 2), kernel_initializer=self.kernel_init,padding='same')(layer)
        layer = concatenate([layer, skip_connection])
        return layer

    def build(self):
        # ======================================== INPUT ==========================================
        # Input layer
        input_layer = Input(shape=(self.input_h, self.input_w, 3), dtype='float32')
        # # Lambda layer: scale input before feeding to the network
        # inScaled = Lambda(lambda x: scale_input(x))(inBlock)
        # First block
        stack = Conv2D(self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last')(input_layer)

        # ======================================== ENCODER ========================================
        # To save the output of each dense block of the down-sampling path to later concatenate to the transition up
        skip_connections = list()
        for i in range(self.n_pool):
            # Dense block
            for j in range(self.n_layers_per_dense_block[i]):
                l = self.dense_block(stack, self.growth_rate)
                stack = concatenate([stack, l])
                # Update filters
                self.filters += self.growth_rate
                # save the current output
            skip_connections.append(stack)
            # TD
            stack = self.transition_down(stack, self.filters)

        # ===================================== BOTTLENECK ======================================
        # store the output of each dense block and upsample them all as well
        block_to_upsample = []
        # Create the bottleneck dense block
        for i in range(self.n_layers_per_dense_block[self.n_pool]):
            l = self.dense_block(stack, self.growth_rate)
            block_to_upsample.append(l)
            stack = concatenate([stack, l])

        # ====================================== DECODER =======================================
        # Revert the order within the skip-connections to get the last stacked layer at index=0
        skip_connections = skip_connections[::-1]

        for j in range(self.n_pool):
            # Updating filters is specific for each variant
            if self.model_name == 'fcn_densenet_56' or self.model_name == 'fcn_densenet_67':
                keep_filters = self.n_layers_per_dense_block * self.growth_rate
            else:
                keep_filters = self.n_layers_per_dense_block[self.n_pool + j] * self.growth_rate
            # TU
            stack = self.transition_up(skip_connections[j], block_to_upsample, keep_filters)
            # Dense Block
            block_to_upsample = []
            for k in range(self.n_layers_per_dense_block[self.n_pool + j + 1]):
                l = self.dense_block(stack, self.growth_rate)
                block_to_upsample.append(l)
                stack = concatenate([stack, l])

        # ======================================== OUTPUT ==========================================
        # Output layer
        if self.n_classes == 2:
            stack = Conv2D(1, (1, 1), activation='sigmoid', padding='same', data_format='channels_last')(stack)
        else:
            stack = Conv2D(self.n_classes, (1, 1), activation='softmax', padding='same', data_format='channels_last')(stack)

        # Create model
        model = Model(inputs=input_layer, outputs=stack, name=self.model_name)
        model.compile(optimizer=Adam(),
                      loss="categorical_crossentropy",
                      metrics=[dice, jaccard, ]
                      )

        # Load weights if pre-trained
        if self.pre_trained:
            if os.path.exists(self.weights_path):
                model.load_weights(self.weights_path)
            else:
                raise Exception(f'Failed to load weights at {self.weights_path}')

        return model


# Deeplab-v3+
class FDSC_DeepLabNet:
    def __init__(self,
                 pre_trained=False,  # if True, set weights_path
                 weights_path=None,  # full-path to the pre-trained weights
                 backbone='mobilenetv2',
                 n_classes=None,
                 input_h=None,
                 input_w=None,
                 activation='relu',
                 kernel_init='he_normal',
                 model_name=None
                 ):
        self.pre_trained = pre_trained
        self.weights_path = weights_path
        self.backbone = backbone
        self.n_classes = n_classes
        self.input_h = input_h
        self.input_w = input_w
        self.activation = activation
        self.kernel_init = kernel_init
        self.model_name = model_name
        self.OS = 16
        self.alpha = 1.

    def depth_conv(self, x, filters, stride=1, kernel_size=3, rate=1, depth_activation=False, eps=1e-3):
        """
        Depth-wise Convolution:
        1. Depth-wise Separable Convolution
        2. Point-wise Convolution

        """
        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'

        if not depth_activation:
            x = Activation(self.activation)(x)
        x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate), padding=depth_padding, use_bias=False)(x)
        x = BatchNormalization(epsilon=eps)(x)
        if depth_activation:
            x = Activation(self.activation)(x)
        x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization(epsilon=eps)(x)
        if depth_activation:
            x = Activation(self.activation)(x)

        return x

    def conv2d_same(self, x, filters, stride=1, kernel_size=3, rate=1):

        if stride == 1:
            return Conv2D(filters,
                          (kernel_size, kernel_size),
                          strides=(stride, stride),
                          padding='same', use_bias=False,
                          dilation_rate=(rate, rate))(x)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            return Conv2D(filters,
                          (kernel_size, kernel_size),
                          strides=(stride, stride),
                          padding='valid', use_bias=False,
                          dilation_rate=(rate, rate))(x)


    def xception_block(self, inputs, depth_list, skip_connect_type, stride, rate=1, depth_activation=False, return_skip=False):

        residual = inputs
        for i in range(3):
            residual = self.depth_conv(residual,
                                  depth_list[i],
                                  stride=stride if i == 2 else 1,
                                  rate=rate,
                                  depth_activation=depth_activation)
            if i == 1:
                skip = residual
        if skip_connect_type == 'conv':
            shortcut = self.conv2d_same(inputs, depth_list[-1],
                                    kernel_size=1,
                                    stride=stride)
            shortcut = BatchNormalization()(shortcut)
            outputs = layers.add([residual, shortcut])
        elif skip_connect_type == 'sum':
            outputs = layers.add([residual, inputs])
        elif skip_connect_type == 'none':
            outputs = residual
        if return_skip:
            return outputs, skip
        else:
            return outputs

    def inverted_res_block(self, inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1, eps=1e-3):
        in_channels = inputs._keras_shape[-1]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = make_divisible(pointwise_conv_filters, 8)
        x = inputs
        if block_id:
            # Expand
            x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                       use_bias=False, activation=None)(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
            x = Lambda(lambda x: relu(x, max_value=6.))(x)
        # Depthwise
        x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same', dilation_rate=(rate, rate))(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        x = Lambda(lambda x: relu(x, max_value=6.))(x)

        x = Conv2D(pointwise_filters,kernel_size=1, padding='same', use_bias=False, activation=None)(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

        if skip_connection:
            return Add()([inputs, x])

        # if in_channels == pointwise_filters and stride == 1:
        #    return Add(name='res_connect_' + str(block_id))([inputs, x])
        return x

    def build(self):
        if not (self.backbone in {'xception', 'mobilenetv2'}):
            raise ValueError('The `backbone` argument should be either `xception`  or `mobilenetv2` ')

        img_input = Input(shape=(self.input_h, self.input_w, 3))
        # Lambda layer: scale input before feeding to the network
        img_input = Lambda(lambda x: scale_input(x))(img_input)

        if self.backbone == 'xception':
            if self.OS == 8:
                entry_block3_stride = 1
                middle_block_rate = 2  # ! Not mentioned in paper, but required
                exit_block_rates = (2, 4)
                atrous_rates = (12, 24, 36)
            else:
                entry_block3_stride = 2
                middle_block_rate = 1
                exit_block_rates = (1, 2)
                atrous_rates = (6, 12, 18)
            x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same')(img_input)

            x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
            x = Activation(self.activation)(x)

            x = self.conv2d_same(x, 64, kernel_size=3, stride=1)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)

            x = self.xception_block(x, [128, 128, 128], skip_connect_type='conv', stride=2, depth_activation=False)
            x, skip1 = self.xception_block(x, [256, 256, 256], skip_connect_type='conv', stride=2, depth_activation=False, return_skip=True)

            x = self.xception_block(x, [728, 728, 728], skip_connect_type='conv', stride=entry_block3_stride, depth_activation=False)
            for i in range(16):
                x = self.xception_block(x, [728, 728, 728], skip_connect_type='sum', stride=1, rate=middle_block_rate, depth_activation=False)

            x = self.xception_block(x, [728, 1024, 1024], skip_connect_type='conv', stride=1, rate=exit_block_rates[0], depth_activation=False)
            x = self.xception_block(x, [1536, 1536, 2048], skip_connect_type='none', stride=1, rate=exit_block_rates[1], depth_activation=True)


        # Backbone='mobilenetv2'
        else:
            self.OS = 8
            first_block_filters = make_divisible(32 * self.alpha, 8)
            x = Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='same', use_bias=False)(img_input)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

            x = Lambda(lambda x: relu(x, max_value=6.))(x)

            x = self.inverted_res_block(x, filters=16, alpha=self.alpha, stride=1, expansion=1, block_id=0, skip_connection=False)

            x = self.inverted_res_block(x, filters=24, alpha=self.alpha, stride=2,
                                    expansion=6, block_id=1, skip_connection=False)
            x = self.inverted_res_block(x, filters=24, alpha=self.alpha, stride=1,
                                    expansion=6, block_id=2, skip_connection=True)

            x = self.inverted_res_block(x, filters=32, alpha=self.alpha, stride=2,
                                    expansion=6, block_id=3, skip_connection=False)
            x = self.inverted_res_block(x, filters=32, alpha=self.alpha, stride=1,
                                    expansion=6, block_id=4, skip_connection=True)
            x = self.inverted_res_block(x, filters=32, alpha=self.alpha, stride=1,
                                    expansion=6, block_id=5, skip_connection=True)

            # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
            x = self.inverted_res_block(x, filters=64, alpha=self.alpha, stride=1,  # 1!
                                    expansion=6, block_id=6, skip_connection=False)
            x = self.inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=7, skip_connection=True)
            x = self.inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=8, skip_connection=True)
            x = self.inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=9, skip_connection=True)

            x = self.inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=10, skip_connection=False)
            x = self.inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=11, skip_connection=True)
            x = self.inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=12, skip_connection=True)

            x = self.inverted_res_block(x, filters=160, alpha=self.alpha, stride=1, rate=2,  # 1!
                                    expansion=6, block_id=13, skip_connection=False)
            x = self.inverted_res_block(x, filters=160, alpha=self.alpha, stride=1, rate=4,
                                    expansion=6, block_id=14, skip_connection=True)
            x = self.inverted_res_block(x, filters=160, alpha=self.alpha, stride=1, rate=4,
                                    expansion=6, block_id=15, skip_connection=True)

            x = self.inverted_res_block(x, filters=320, alpha=self.alpha, stride=1, rate=4,
                                    expansion=6, block_id=16, skip_connection=False)
        # Image Feature branch
        b4 = AveragePooling2D(pool_size=(int(np.ceil(self.input_h / self.OS)), int(np.ceil(self.input_w / self.OS))))(x)

        b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
        b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation(self.activation)(b4)

        b4 = Lambda(lambda x: K.tf.image.resize_bilinear(x, size=(int(np.ceil(self.input_h / self.OS)), int(np.ceil(self.input_w / self.OS)))))(b4)

        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation(self.activation)(b0)

        # there are only 2 branches in mobilenetV2. not sure why
        if self.backbone == 'xception':
            # rate = 6 (12)
            b1 = self.depth_conv(x, 256, rate=atrous_rates[0], depth_activation=True, eps=1e-5)
            # rate = 12 (24)
            b2 = self.depth_conv(x, 256, rate=atrous_rates[1], depth_activation=True, eps=1e-5)
            # rate = 18 (36)
            b3 = self.depth_conv(x, 256, rate=atrous_rates[2], depth_activation=True, eps=1e-5)

            # concatenate ASPP branches & project
            x = Concatenate()([b4, b0, b1, b2, b3])
        else:
            x = Concatenate()([b4, b0])

        x = Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization(epsilon=1e-5)(x)
        x = Activation(self.activation)(x)
        x = Dropout(0.1)(x)


        # DeepLab v.3+ decoder
        if self.backbone == 'xception':
            # Feature projection
            # x4 (x2) block
            x = Lambda(lambda x: K.tf.image.resize_bilinear(x, size=(int(np.ceil(self.input_h / 4)), int(np.ceil(self.input_w / 4)))))(x)

            dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False)(skip1)
            dec_skip1 = BatchNormalization(epsilon=1e-5)(dec_skip1)
            dec_skip1 = Activation(self.activation)(dec_skip1)
            x = Concatenate()([x, dec_skip1])
            x = self.depth_conv(x, 256, depth_activation=True, eps=1e-5)
            x = self.depth_conv(x, 256, depth_activation=True, eps=1e-5)


        # Output layer
        x = Conv2D(self.n_classes, (1, 1), padding='same')(x)
        x = Lambda(lambda x: K.tf.image.resize_bilinear(x, size=(self.input_h, self.input_w)))(x)
        x = Activation('softmax')(x)

        # Create model
        model = Model(inputs=img_input, outputs=x)
        model.compile(optimizer=Adam(),
                      loss="categorical_crossentropy",
                      metrics=[dice, jaccard, ]
                      )

        # Load weights if pre-trained
        if self.pre_trained:
            if os.path.exists(self.weights_path):
                model.load_weights(self.weights_path)
            else:
                raise Exception(f'Failed to load weights {self.weights_path}')

        return model

