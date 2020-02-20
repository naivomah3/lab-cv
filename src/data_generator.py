'''
    Data loader/generator
'''
import numpy as np
import cv2
import os

# load image of desired size
def get_frame(frame_path, height, width, is_resizable=False):
    frame = cv2.imread(frame_path, 1)
    return frame

# load image of desired size
def get_mask(mask_path=None, height=None, width=None, n_classes=None, is_resizable=False):
    mask = cv2.imread(mask_path)
    if n_classes == 2:
        return mask
    else:
        # Mask Encoding: OneHotEncoding
        # Mask depth should be the same as the number of classes
        mask_ohe_mapping = np.zeros((height, width, n_classes))  # Each depth is fed one-by-one corresponding each class
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
                   batch_size=None,
                   is_resizable=False,
                   training=True # If prediction, load X only
                   ):
    '''
    fnames: a list of all frames file name == list of all mask file names
    '''
    while True:
        batch_frames, batch_masks = list(), list()

        # Get random list of filenames within loaded 'fnames'
        rand_filenames = np.random.choice(a=fnames, size=batch_size)

        for filename in rand_filenames:
            frame = get_frame(os.path.join(frames_path, filename), input_h, input_w, is_resizable)
            #print(f"FRAME: {input_h}")
            mask = get_mask(os.path.join(masks_path, filename), input_h, input_w, n_classes, is_resizable)
            batch_frames.append(frame)
            batch_masks.append(mask)

        X = np.array(batch_frames)
        Y = np.array(batch_masks)
        # If binary tensor shape is 3, expand to 4; else already 4
        # if n_classes == 2:
        #     Y = np.expand_dims(Y, axis=-1)

        if training:
            yield (X, Y)
        else:
            yield (X)

# Build data directly without generator
def data_loader(frames_path=None,
                masks_path=None,
                input_h=None,
                input_w=None,
                n_classes=None,
                fnames=None,
                is_resizable=False,
                training=True # If prediction, load X only
                ):

    # X is of depth=3, Y of depth=n_classes
    X, Y = list(), list()
    for file_name in fnames:
        X.append(get_frame(os.path.join(frames_path, file_name), input_h, input_w, is_resizable))
        if training:
            Y.append(get_mask(os.path.join(masks_path, file_name), input_h, input_w, n_classes, is_resizable))

    X = np.array(X)
    Y = np.array(Y)
    # If binary tensor shape is 3, expand to 4; else already 4
    if n_classes == 2:
        Y = np.array(Y)
        Y = np.expand_dims(Y, axis=-1)
    # If training, return X, Y; else only X
    if training:
        return X, Y

    return X


