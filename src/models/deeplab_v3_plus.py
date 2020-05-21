import os
import numpy as np
import tensorflow as tf
from keras.models import Model  # Functional API
from keras import layers
from keras.layers import (
    Conv2D, Input, Dropout, Lambda, Concatenate,
    BatchNormalization, Activation, AveragePooling2D,
    ZeroPadding2D, DepthwiseConv2D, Add)
from keras.optimizers import Adam
from keras.activations import relu
#from tensorflow.keras import backend as K

from src.metrics import (dice, jaccard)
from src.engine import scale_input
from src.utils import make_divisible


# Deeplab-v3+
class DSC_DeepLab_v3_plus:
    def __init__(self,
                 pre_trained=False,  # if True, set weights_path
                 weights_path=None,  # full-path to the pre-trained models_weights
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

    def depth_sep_conv(self, x, filters, stride=1, kernel_size=3, rate=1, depth_activation=False, eps=1e-3):
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
            residual = self.depth_sep_conv(residual,
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
        return x

    def build(self):
        if not (self.backbone in {'xception', 'mobilenetv2'}):
            raise ValueError('The `backbone` argument should be either `xception`  or `mobilenetv2` ')

        img_input = Input(shape=(self.input_h, self.input_w, 3))
        # Lambda layer: scale input before feeding to the network
        batches_input = Lambda(lambda x: scale_input(x))(img_input)

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
            x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same')(batches_input)

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
            x = Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='same', use_bias=False)(batches_input)
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
        b4 = BatchNormalization(epsilon=1e-5)(b4)
        b4 = Activation(self.activation)(b4)

        b4 = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, size=(int(np.ceil(self.input_h / self.OS)), int(np.ceil(self.input_w / self.OS)))))(b4)

        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = BatchNormalization(epsilon=1e-5)(b0)
        b0 = Activation(self.activation)(b0)

        # there are only 2 branches in mobilenetV2. not sure why
        if self.backbone == 'xception':
            # rate = 6 (12)
            b1 = self.depth_sep_conv(x, 256, rate=atrous_rates[0], depth_activation=True, eps=1e-5)
            # rate = 12 (24)
            b2 = self.depth_sep_conv(x, 256, rate=atrous_rates[1], depth_activation=True, eps=1e-5)
            # rate = 18 (36)
            b3 = self.depth_sep_conv(x, 256, rate=atrous_rates[2], depth_activation=True, eps=1e-5)

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
            x = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, size=(int(np.ceil(self.input_h / 4)), int(np.ceil(self.input_w / 4)))))(x)

            dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False)(skip1)
            dec_skip1 = BatchNormalization(epsilon=1e-5)(dec_skip1)
            dec_skip1 = Activation(self.activation)(dec_skip1)
            x = Concatenate()([x, dec_skip1])
            x = self.depth_sep_conv(x, 256, depth_activation=True, eps=1e-5)
            x = self.depth_sep_conv(x, 256, depth_activation=True, eps=1e-5)


        # Output layer
        x = Conv2D(self.n_classes, (1, 1), padding='same')(x)
        x = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, size=(self.input_h, self.input_w)))(x)
        x = Activation('softmax')(x)

        # Create model
        model = Model(inputs=img_input, outputs=x)
        model.compile(optimizer=Adam(),
                      loss="categorical_crossentropy",
                      metrics=[dice, jaccard, ]
                      )

        # Load models_weights if pre-trained
        if self.pre_trained:
            if os.path.exists(self.weights_path):
                model.load_weights(self.weights_path)
            else:
                raise Exception(f'Failed to load weights {self.weights_path}')

        return model