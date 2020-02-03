'''
    Data generator
'''
import numpy as np
import cv2
import os
from random import shuffle


# load image of desired size
def get_frame(frame_path, height, width, is_resizable=False):
    frame = cv2.imread(frame_path, 1)
    if is_resizable:
        frame = np.float32(cv2.resize(frame, (height, width)))
    return frame

# load image of desired size
def get_mask(mask_path=None, height=None, width=None, n_classes=None, is_resizable=False):
    # Mask Encoding: OneHotEncoding
    # Mask depth should be the same as the number of classes
    mask_ohe_mapping = np.zeros((height, width, n_classes))  # Each depth is fed one-by-one corresponding each class
    mask = cv2.imread(mask_path, 1)
    if is_resizable:
        mask = cv2.resize(mask, (height, width))
    # Make sure having the map of all classes in the first channel
    mask = mask[:, :, 0]

    for i in range(n_classes):
        # Depth wise encoding
        mask_ohe_mapping[:, :, i] = (mask == i).astype(int)  # OneHotEncode True:1-False:0

    return mask_ohe_mapping


# Batch generator for Training and Validation
def data_generator(frames_path=None,
                   masks_path=None,
                   fnames=None,
                   n_classes=None,
                   input_h=None,
                   input_w=None,
                   batch_size=32,
                   is_resizable=False):
    '''
    fnames: a list of all frames file name == list of all mask file names
    '''
    while True:
        batch_frames, batch_masks = list(), list()

        # Get random list of filenames within loaded 'fnames'
        rand_filenames = np.random.choice(a=fnames, size=batch_size)

        for filename in rand_filenames:
            frame = get_frame(os.path.join(frames_path, filename), input_h, input_w, is_resizable)
            mask = get_mask(os.path.join(masks_path, filename), input_h, input_w, n_classes, is_resizable)
            batch_frames.append(frame)
            batch_masks.append(mask)

        X = np.array(batch_frames)
        Y = np.array(batch_masks)

        yield (X, Y)

# Build data directly without generator
def data_loader(frames_path=None,
                masks_path=None,
                input_h=None,
                input_w=None,
                n_classes=None,
                fnames=None,
                is_resizable=False,
                ):

    # X is of depth=3, Y of depth=n_classes
    X_test, Y_test = list(), list()
    for file_name in fnames:
        X_test.append(get_frame(os.path.join(frames_path, file_name), input_h, input_w, is_resizable))
        Y_test.append(get_mask(os.path.join(masks_path, file_name), input_h, input_w, n_classes, is_resizable))

    return np.array(X_test), np.array(Y_test)