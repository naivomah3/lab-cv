import os
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.models import Model  # Functional API
from keras.layers import (
        Conv2D, Input, MaxPooling2D,
        Conv2DTranspose, Dropout, Lambda,
        concatenate, BatchNormalization, Activation,
        Reshape, ConvLSTM2D)
from keras.optimizers import Adam
import numpy as np

def scale_input(x):
    '''pre-processing: scaling/normalization
    rtype: x = {(x / 255) - 0.5} * 2 <- to be proved
    '''
    x /= 255.
    # x -= 0.5
    # x *= 2.
    return x

# Binary DSC - IoU
def dice(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection) / (union + smooth)
    # y_true_shape = tf.shape(y_true)
    # y_true = tf.reshape(y_true, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])
    # y_pred = tf.reshape(y_pred, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])
    # intersection = tf.reduce_sum(y_true * y_pred, axis=1)
    # union = tf.reduce_sum(y_true + y_pred, axis=1)
    # return (2. * intersection) / (union + smooth)
def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)

# Binary JSC - F1_score
def jaccard(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    jaccard = intersection / (union - intersection + smooth)
    return jaccard
def jaccard_loss(y_true, y_pred):
  return 1 - jaccard(y_true, y_pred)

# Multi-label DSC - IoU
def dice_multilabel_loss(y_true, y_pred, no_classes=4, eps=1e-6):
    ####Got negative dices with this
    # total_dice = 0
    # for index in range(no_classes):
    #     total_dice -= dice(y_true[:,:,:, index], y_pred[:,:,:, index]) # [n_sample, x, y, labels/channels]
    # return total_dice
    # [b, h, w, classes]

    ###Got stationary dice loss when computing again softmax out here.
    #pred_tensor = tf.nn.softmax(y_pred) # I don't think it should again compute this activation here, this work has been done at the last layer
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(y_pred, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    # count how many of each class are present in
    # each image, if there are zero, then assign
    # them a fixed weight of eps
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / (counts ** 2)
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights * multed, axis=-1)
    denom = tf.reduce_sum(weights * summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)

# Multi-label JSC - F1_score
# Not yet tested
def jaccard_multilabel_loss(y_true, y_pred, no_classes=4):
    total_jaccard = 0
    for index in range(no_classes):
        total_jaccard -= jaccard(y_true[:,:,:,index], y_pred[:,:,:,index]) # [n_sample, x, y, labels/channels]
    return total_jaccard


# Build U-Net: original paper
def unet(pre_trained=False,  # if True, set weights_path
         weights_path=None,  # full-path to the pre-trained weights
         n_classes=None,
         input_h=None,
         input_w=None,
         activation='elu',
         kernel_init='he_normal',
         model_name=None):

    # Compile model
    inBlock = Input(shape=(input_h, input_w, 3), dtype='float32')
    # Lambda layer: scale input before feeding to the network
    inScaled = Lambda(lambda x: scale_input(x))(inBlock)
    # Block 1d
    convB1d = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(inScaled)
    convB1d = BatchNormalization()(convB1d)
    convB1d = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB1d)
    convB1d = BatchNormalization()(convB1d)
    poolB1d = MaxPooling2D(pool_size=(2, 2))(convB1d)
    # Block 2d
    convB2d = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(poolB1d)
    convB2d = BatchNormalization()(convB2d)
    convB2d = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB2d)
    convB2d = BatchNormalization()(convB2d)
    poolB2d = MaxPooling2D(pool_size=(2, 2))(convB2d)
    # Block 3d
    convB3d = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(poolB2d)
    convB3d = BatchNormalization()(convB3d)
    convB3d = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB3d)
    convB3d = BatchNormalization()(convB3d)
    poolB3d = MaxPooling2D(pool_size=(2, 2))(convB3d)
    # Block 4d
    convB4d = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(poolB3d)
    convB4d = BatchNormalization()(convB4d)
    convB4d = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB4d)
    convB4d = BatchNormalization()(convB4d)
    poolB4d = MaxPooling2D(pool_size=(2, 2))(convB4d)
    # Bottleneck
    convBn = Conv2D(1024, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(poolB4d)
    convBn = BatchNormalization()(convBn)
    convBn = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convBn)
    convBn = BatchNormalization()(convBn)
    # Block 4u
    convB4u = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(convBn)
    convB4u = concatenate([convB4u, convB4d])
    convB4u = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB4u)
    convB4u = BatchNormalization()(convB4u)
    convB4u = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB4u)
    convB4u = BatchNormalization()(convB4u)
    # Block 3u
    convB3u = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(convB4u)
    convB3u = concatenate([convB3u, convB3d])
    convB3u = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB3u)
    convB3u = BatchNormalization()(convB3u)
    convB3u = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB3u)
    convB3u = BatchNormalization()(convB3u)
    # Block B2u
    convB2u = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(convB3u)
    convB2u = concatenate([convB2u, convB2d])
    convB2u = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB2u)
    convB2u = BatchNormalization()(convB2u)
    convB2u = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB2u)
    convB2u = BatchNormalization()(convB2u)
    # Block B1u
    convB1u = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(convB2u)
    convB1u = concatenate([convB1u, convB1d], axis=3)
    convB1u = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB1u)
    convB1u = BatchNormalization()(convB1u)
    convB1u = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB1u)
    convB1u = BatchNormalization()(convB1u)

    # Output layer
    if n_classes == 2:
        outBlock = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(convB1u)
    else:
        outBlock = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(convB1u)

    # Create model
    model = Model(inputs=inBlock, outputs=outBlock, name=model_name)
    model.compile(optimizer=Adam(),
                  loss="categorical_crossentropy",
                  metrics=[dice, jaccard, ]
                  )

    # Load weights if pre-trained
    if pre_trained:
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        else:
            raise Exception(f'Failed to load weights at {weights_path}')

    # Get summary after loading weights
    model.summary()

    return model

# Build Bi-Directional ConvLSTM U-Net with Densely Connected Convolutions
# Dense Block = 3
def bcd_unet_d3(pre_trained=False,  # if True, set weights_path
         weights_path=None,  # full-path to the pre-trained weights
         n_classes=None,
         input_h=None,
         input_w=None,
         activation='elu',
         kernel_init='he_normal',
         model_name=None):

    # Compile model
    inBlock = Input(shape=(input_h, input_w, 3), dtype='float32')
    # Lambda layer: scale input before feeding to the network
    inScaled = Lambda(lambda x: scale_input(x))(inBlock)
    # =============================================== ENCODING ==================================================
    # Block 1d
    convB1d = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(inScaled)
    convB1d = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB1d)
    poolB1d = MaxPooling2D(pool_size=(2, 2))(convB1d)
    # Block 2d
    convB2d = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(poolB1d)
    convB2d = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB2d)
    poolB2d = MaxPooling2D(pool_size=(2, 2))(convB2d)
    # Block 3d
    convB3d = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(poolB2d)
    convB3d = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB3d)
    dropB3d = Dropout(0.5)(convB3d)
    poolB3d = MaxPooling2D(pool_size=(2, 2))(convB3d)

    # =============================================== BOOTLENECK =================================================
    # Bottleneck - Block D1
    convBnd1 = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(poolB3d)
    convBnd1 = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convBnd1)
    dropBnd1 = Dropout(0.5)(convBnd1)
    # Bottlenbeck - Block D2
    convBnd2 = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(dropBnd1)
    convBnd2 = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convBnd2)
    dropBnd2 = Dropout(0.5)(convBnd2)
    # Bottlenbeck - Block D3
    merge_d1_d2 = concatenate([dropBnd1, dropBnd2], axis=3)
    convBnd3 = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(merge_d1_d2)
    convBnd3 = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convBnd3)
    dropBnd3 = Dropout(0.5)(convBnd3)

    # =============================================== DECODING ==================================================
    # Block 4u
    convB4u = Conv2DTranspose(256, kernel_size=2, strides=2, kernel_regularizer=kernel_init, padding='same')(dropBnd3)
    convB4u = BatchNormalization(axis=3)(convB4u)
    convB4u = Activation(activation)(convB4u)
    dropB3d = Reshape(target_shape=(1, np.int32(input_h/4), np.int32(input_h/4), 256))(dropB3d) # just to make sure about shape, but I think the size is already okay :P
    convB4u = Reshape(target_shape=(1, np.int32(input_h/4), np.int32(input_h/4), 256))(convB4u)
    merge_3d_4u = concatenate([dropB3d, convB4u], axis=1)
    merge_3d_4u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=kernel_init)(merge_3d_4u)
    convB4u = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(merge_3d_4u)
    convB4u = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB4u)

    # Block 3u
    convB3u = Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=kernel_init, padding='same')(convB4u)
    convB3u = BatchNormalization(axis=3)(convB3u)
    convB3u = Activation(activation)(convB3u)
    convB2d = Reshape(target_shape=(1, np.int32(input_h/2), np.int32(input_h/2), 128))(convB2d)
    convB3u = Reshape(target_shape=(1, np.int32(input_h/2), np.int32(input_h/2), 128))(convB3u)
    merge_2d_3u = concatenate([convB2d, convB3u], axis=1)
    merge_2d_3u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=kernel_init)(merge_2d_3u)
    convB3u = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(merge_2d_3u)
    convB3u = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB3u)

    # Block B2u
    convB2u = Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=kernel_init, padding='same')(convB3u)
    convB2u = BatchNormalization(axis=3)(convB2u)
    convB2u = Activation(activation)(convB2u)
    convB1d = Reshape(target_shape=(1, np.int32(input_h), np.int32(input_h), 128))(convB1d)
    convB2u = Reshape(target_shape=(1, np.int32(input_h), np.int32(input_h), 128))(convB2u)
    merge_1d_2u = concatenate([convB1d, convB2u], axis=1)
    merge_1d_2u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=kernel_init)(merge_1d_2u)
    convB2u = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(merge_1d_2u)
    convB2u = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB2u)

    # ================================================== OUTPUT =====================================================
    # Output layer
    if n_classes == 2:
        outBlock = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(convB2u)
    else:
        outBlock = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(convB2u)

    # Create model
    model = Model(inputs=inBlock, outputs=outBlock, name=model_name)
    model.compile(optimizer=Adam(),
                  loss="categorical_crossentropy",
                  metrics=[dice, jaccard, ]
                  )

    # Load weights if pre-trained
    if pre_trained:
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        else:
            raise Exception(f'Failed to load weights at {weights_path}')

    # Get summary after loading weights
    model.summary()

    return model

# Build Bi-Directional ConvLSTM U-Net with Densely Connected Convolutions
# Dense Block = 1
def bcd_unet_d1(pre_trained=False,  # if True, set weights_path
         weights_path=None,  # full-path to the pre-trained weights
         n_classes=None,
         input_h=None,
         input_w=None,
         activation='elu',
         kernel_init='he_normal',
         model_name=None):

    # Compile model
    inBlock = Input(shape=(input_h, input_w, 3), dtype='float32')
    # Lambda layer: scale input before feeding to the network
    inScaled = Lambda(lambda x: scale_input(x))(inBlock)
    # =============================================== ENCODING ==================================================
    # Block 1d
    convB1d = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(inScaled)
    convB1d = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB1d)
    poolB1d = MaxPooling2D(pool_size=(2, 2))(convB1d)
    # Block 2d
    convB2d = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(poolB1d)
    convB2d = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB2d)
    poolB2d = MaxPooling2D(pool_size=(2, 2))(convB2d)
    # Block 3d
    convB3d = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(poolB2d)
    convB3d = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB3d)
    dropB3d = Dropout(0.5)(convB3d)
    poolB3d = MaxPooling2D(pool_size=(2, 2))(convB3d)


    # =============================================== BOOTLENECK =================================================
    # Bottleneck - Block D1
    convBnd1 = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(poolB3d)
    convBnd1 = Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convBnd1)
    dropBnd1 = Dropout(0.5)(convBnd1)

    # =============================================== DECODING ==================================================
    # Block 4u
    convB4u = Conv2DTranspose(256, kernel_size=2, strides=2, kernel_regularizer=kernel_init, padding='same')(dropBnd1)
    convB4u = BatchNormalization(axis=3)(convB4u)
    convB4u = Activation(activation)(convB4u)
    dropB3d = Reshape(target_shape=(1, np.int32(input_h/4), np.int32(input_h/4), 256))(dropB3d) # just to make sure about shape, but I think the size is already okay :P
    convB4u = Reshape(target_shape=(1, np.int32(input_h/4), np.int32(input_h/4), 256))(convB4u)
    merge_3d_4u = concatenate([dropB3d, convB4u], axis=1)
    merge_3d_4u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=kernel_init)(merge_3d_4u)
    convB4u = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(merge_3d_4u)
    convB4u = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB4u)

    # Block 3u
    convB3u = Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=kernel_init, padding='same')(convB4u)
    convB3u = BatchNormalization(axis=3)(convB3u)
    convB3u = Activation(activation)(convB3u)
    convB2d = Reshape(target_shape=(1, np.int32(input_h/2), np.int32(input_h/2), 128))(convB2d)
    convB3u = Reshape(target_shape=(1, np.int32(input_h/2), np.int32(input_h/2), 128))(convB3u)
    merge_2d_3u = concatenate([convB2d, convB3u], axis=1)
    merge_2d_3u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=kernel_init)(merge_2d_3u)
    convB3u = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(merge_2d_3u)
    convB3u = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB3u)

    # Block B2u
    convB2u = Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=kernel_init, padding='same')(convB3u)
    convB2u = BatchNormalization(axis=3)(convB2u)
    convB2u = Activation(activation)(convB2u)
    convB1d = Reshape(target_shape=(1, np.int32(input_h), np.int32(input_h), 128))(convB1d)
    convB2u = Reshape(target_shape=(1, np.int32(input_h), np.int32(input_h), 128))(convB2u)
    merge_1d_2u = concatenate([convB1d, convB2u], axis=1)
    merge_1d_2u = ConvLSTM2D(128, kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer=kernel_init)(merge_1d_2u)
    convB2u = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(merge_1d_2u)
    convB2u = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, padding='same')(convB2u)

    # ================================================== OUTPUT =====================================================
    # Output layer
    if n_classes == 2:
        outBlock = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(convB2u)
    else:
        outBlock = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(convB2u)

    # Create model
    model = Model(inputs=inBlock, outputs=outBlock, name=model_name)
    model.compile(optimizer=Adam(),
                  loss="categorical_crossentropy",
                  metrics=[dice, jaccard, ]
                  )

    # Load weights if pre-trained
    if pre_trained:
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        else:
            raise Exception(f'Failed to load weights at {weights_path}')

    # Get summary after loading weights
    model.summary()

    return model