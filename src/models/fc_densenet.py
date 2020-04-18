import os
from keras.models import Model  # Functional API
from keras.layers import (
    Conv2D, Input, MaxPooling2D,
    Conv2DTranspose, Dropout, Lambda,
    concatenate, BatchNormalization, Activation)
from keras.optimizers import Adam
#from tensorflow.keras import backend as K

from src.metrics import (dice, jaccard)
from src.engine import scale_input

class FC_DenseNet:
    def __init__(self,
                 pre_trained=False,  # if True, set weights_path
                 weights_path=None,  # full-path to the pre-trained models_weights
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
        self.input_filters = 48   # this value is the number of filter of the first convolution block --> this will be updated at the beginning of each dense block
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
        input_layer = Input(shape=(self.input_h, self.input_w, 3), dtype='float32')
        # Lambda layer: scale input before feeding to the network
        inScaled = Lambda(lambda x: scale_input(x))(input_layer)
        # First block
        stack = Conv2D(self.input_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last')(inScaled)

        # ======================================== ENCODER ========================================
        # To save the output of each dense block of the down-sampling path to later concatenate to the transition up
        skip_connections = list()
        for i in range(self.n_pool):
            # DB: Dense-Block
            for j in range(self.n_layers_per_dense_block[i]):
                l = self.dense_block(stack, self.growth_rate)
                stack = concatenate([stack, l])
                # Update filters
                self.input_filters += self.growth_rate
                # save the current output
            skip_connections.append(stack)
            # TD: Transition-Up
            stack = self.transition_down(stack, self.input_filters)

        # ===================================== BOTTLENECK ======================================
        # store the output of each dense block and upsample them all as well
        block_to_upsample = []
        # Create the bottleneck dense block
        for i in range(self.n_layers_per_dense_block[self.n_pool]):
            # DB: Dense-Block
            l = self.dense_block(stack, self.growth_rate)
            block_to_upsample.append(l)
            stack = concatenate([stack, l])

        # ====================================== DECODER =======================================
        # Revert the order of layers within the skip-connections to get the last pooling-layer at index=0
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
        if self.n_classes == 2:
            stack = Conv2D(1, (1, 1), activation='sigmoid', padding='same', data_format='channels_last')(stack)
        else:
            stack = Conv2D(self.n_classes, (1, 1), activation='softmax', padding='same', data_format='channels_last')(stack)

        # Compile model
        model = Model(inputs=input_layer, outputs=stack, name=self.model_name)
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