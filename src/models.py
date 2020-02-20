import os
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.models import Model, load_model
from keras.layers import Conv2D, Input, MaxPooling2D, Conv2DTranspose, Dropout, Lambda, concatenate, BatchNormalization
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
def dice(y_true, y_pred):
    intersection = tf.reduce_sum(y_true*y_pred, axis=(1, 2))
    union = tf.reduce_sum(y_true+y_pred, axis=(1, 2))
    dice = 2 * intersection / (union + 1e-4)
    return dice

def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)

# Binary JSC - F1_score
def jaccard(y_true, y_pred):
  intersection = tf.reduce_sum(y_true*y_pred, axis=(1, 2))
  union = tf.reduce_sum(y_true+y_pred, axis=(1, 2))
  jaccard = intersection / (union - intersection + 1e-4)
  return jaccard

def jaccard_loss(y_true, y_pred):
  return 1 - jaccard(y_true, y_pred)

# Multi-label DSC - IoU
def dice_multilabel(y_true, y_pred, no_classes=4):
    total_dice = 0
    for index in range(no_classes):
        total_dice -= dice(y_true[:,:,:,index], y_pred[:,:,:,index]) # [n_sample, x, y, labels/channels]
    return total_dice

def dice_multilabel_loss(y_true, y_pred):
    return  1 - dice_multilabel(y_true, y_pred)

# Multi-label JSC - F1_score
def jaccard_multilabel(y_true, y_pred, no_classes=4):
    total_jaccard = 0
    for index in range(no_classes):
        total_jaccard -= jaccard(y_true[:,:,:,index], y_pred[:,:,:,index]) # [n_sample, x, y, labels/channels]
    return total_jaccard

def jaccard_multilabel(y_true, y_pred, no_classes=4):
    return 1 - jaccard_multilabel(y_true, y_pred)


# Build U-Net: original paper
def unet(pre_trained=False,
         model_path=None,
         n_classes=None,
         input_h=None,
         input_w=None,
         activation='elu',
         kernel_init='he_normal',
         model_name=None):
    if pre_trained:
        if os.path.exists(model_path):
            model = load_model(model_path,
                               custom_objects={'dice_multilabel': dice_multilabel,
                                               'jaccard_multilabel': jaccard_multilabel,
                                               'scale_input': scale_input}
                               )
            model.compile(optimizer=Adam(),
                          loss=dice_multilabel_loss,
                          metrics=[dice_multilabel,
                                   jaccard_multilabel,
                                   ]
                          )
            model.summary()
            return model
        else:
            raise Exception(f'Failed to load the existing model at {model_path}')

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
                  loss=dice_multilabel_loss,
                  metrics=[dice_multilabel,
                           jaccard_multilabel,
                           ]
                  )
    # Kamariya
    model.summary()

    return model