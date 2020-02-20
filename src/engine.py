import random, string
import numpy as np

def scale_input(x):
    '''
    pre-processing: feature scaling/normalization
    rtype: x = {(x / 255) - 0.5} * 2 <- to be proved
    '''
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def get_rand_name(size=30, chars=string.ascii_letters + string.digits):
    '''
    Generate Random filename
    rtype: a random string of length 30
    '''
    return ''.join(random.choice(chars) for x in range(size))

def unison_shuffled(a, b):
    '''
    a & b: 2 numpy arrays
    '''
    if len(a) != len(b):
        raise Exception('Dimensions do not match')
    p = np.random.permutation(len(a))
    return a[p], b[p]


def color_img(frame, n_classes):
    '''
    Pixel wise coloring of each class label
    frame: 2-d matrix of all respective classes
    '''
    # create channels
    mask_channels = np.zeros((frame.shape[0], frame.shape[1], n_classes)).astype("float")
    # Create color palette
    # black =  [  0,   0,   0]
    # red =    [230,   0,   0]
    # green =  [  0, 230,   0]
    # blue =   [  0,  38, 230]
    # violet = [222,   0, 292]
    # yellow = [245, 255,  82]
    R = [0, 230,   0,   0, 222, 245]
    G = [0,   0, 230,  38,   0, 255]
    B = [0,   0,   0, 230, 292,  82]

    if n_classes == 2:
        # Color labels for binary classification
        mask_channels *= 255
    else:
        # Color labels for multi label classification
        # Assuming the first label is Background and be colored with Black(0,0,0)
        for i in range(n_classes):
            mask_color = (frame == i)
            mask_channels[:, :, 0] += (mask_color * B[i])
            mask_channels[:, :, 1] += (mask_color * G[i])
            mask_channels[:, :, 2] += (mask_color * R[i])

    return (mask_channels)