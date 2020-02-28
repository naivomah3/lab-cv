import os
import numpy as np
from keras.models import Model  # Functional API
from keras.layers import (
    Conv2D, Input, MaxPooling2D,
    Conv2DTranspose, Dropout, Lambda, Concatenate,
    concatenate, BatchNormalization, Activation,
    Reshape, ConvLSTM2D)
from keras.optimizers import Adam
from keras.regularizers import l2

from src.metrics import (
    scale_input, dice, jaccard)


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
            raise Exception(f"Model name not understood {model_name}")


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
        # First block
        stack = Conv2D(self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last')(input_layer)

        # ======================================== ENCODER ========================================
        # To save the output of each dense block of the down-sampling path to later concate to the transition up sampling
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
            # Update filters
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















