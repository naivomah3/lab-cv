import numpy as np
import cv2
import os
from random import shuffle

# Create dataset with the appropriate size
def get_frame(frame_path, height, width):
    frame = cv2.imread(frame_path, 1)
    frame = np.float32(cv2.resize(frame, (height, width)))
    return frame


def get_mask(mask_path=None, height=None, width=None, n_classes=None):
    # Mask depth should be the same as the number of classes
    mask_depth = np.zeros((height, width, n_classes))  # Each depth will be fed one-by-one corresponding to the classs label
    mask = cv2.imread(mask_path, 1)
    mask = cv2.resize(mask, (height, width))
    mask = mask[:, :, 0]  # take only the first channel

    for i in range(n_classes):
        # One depth
        mask_depth[:, :, i] = (mask == i).astype(int)  # OneHotEncode True-False then 0-1

    return mask_depth


# Batch generator for Training and Validation
def image_data_generator(frames_path=None, masks_path=None, fnames=None, n_classes=None, batch_size=32):
    '''
    fnames: a list of all frames file name == list of all mask file names
    '''
    while True:
        batch_frames, batch_masks = list(), list()

        # Get random list of filenames from frame_ids of size 'batch_size'
        rand_filenames = np.random.choice(a=fnames, size=batch_size)

        for filename in rand_filenames:
            frame = get_frame(os.path.join(frames_path, filename), 320, 320, n_classes)
            mask = get_mask(os.path.join(masks_path, filename), 320, 320)
            batch_frames.append(frame)
            batch_masks.append(mask)

        X = np.array(batch_frames)
        Y = np.array(batch_masks)

        yield (X, Y)


# Build data without generator for Testing
def image_data_builder(frames_path=None, masks_path=None, fnames=None):
    # X[i]: Matrix of input image with depth=3, Y Matrix of label image with depth=n_classes
    X_test, Y_test = list(), list()
    for file_name in fnames:
        X_test.append(get_frame(os.path.join(frames_path, file_name), 320, 320))
        Y_test.append(get_mask(os.path.join(masks_path, file_name), 320, 320))

    X_test, Y_test = shuffle(X_test, Y_test)
    return np.array(X_test), np.array(Y_test)