from tensorflow.keras import backend as K
import tensorflow as tf

# Binary DSC - IoU
def dice(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection) / (union + smooth)
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
    ####Got negative dices with the following
    # total_dice = 0
    # for index in range(no_classes):
    #     total_dice -= dice(y_true[:,:,:, index], y_pred[:,:,:, index]) # [n_sample, x, y, labels/channels]
    # return total_dice
    # [b, h, w, classes]

    ###Got stationary dice loss when computing again softmax out here. Softmax is done at the last last layer
    #pred_tensor = tf.nn.softmax(y_pred)
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

