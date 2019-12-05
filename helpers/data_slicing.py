import os, random, string
import cv2
import numpy as np

def get_rand_name(size=30, chars=string.ascii_letters + string.digits):
    '''
    Random filename generator'''
    return ''.join(random.choice(chars) for x in range(size))


def gen_masks(src_path, dst_path):
    '''
    Generate masks from .png files
    upon to the corresponding classes
    [0: Background, 1: occluded_road, 2: road]'''

    for img_id in os.listdir(src_path):
        # Load png
        mask_in = cv2.imread(os.path.join(src_path, img_id))
        # Read pixel values
        G_mask = mask_in[:,:,1] == 128
        mask_in[G_mask] = 1
        R_mask = mask_in[:,:,2] == 128
        mask_in[R_mask] = 2
        # Save image as png
        cv2.imwrite(os.path.join(dst_path, img_id), mask_in)

def inv_gen_masks(src_path, dst_path):
    '''
    inverse of generate_mask for visualization only'''

    for img_id in os.listdir(src_path):
        # Load mask in png | 1 as RGB
        mask_in = cv2.imread(os.path.join(src_path, img_id), 1)
        # Split channels
        B, G, R = cv2.split(mask_in)
        G_mask = G == 1
        G[G_mask] = 128
        R_mask = R == 2
        R[R_mask] = 128
        mask_out = cv2.merge((B, G, R))
        # Save new normalized mask
        cv2.imwrite(os.path.join(dst_path, img_id), mask_out)


def img_crop_4(src_frames, src_masks, dst_frames, dst_masks):
    '''
    crop a (frame & mask) into 4: from 720x1280x1 to 360x640x4
    - run once
    '''
    for img_id in os.listdir(src_frames):
        img_fname = img_id.split('.')[0]

        # Load frames(png) and masks(npy)
        img_in = cv2.imread(os.path.join(src_frames, img_fname + '.png'))
        mask_in = np.load(os.path.join(src_masks, img_fname + '.npy'))

        for w in range(0, img_in.shape[1], img_in.shape[1]//2):
            for h in range (0, img_in.shape[0], img_in.shape[0]//2):
                img_out = img_in[h:h+(img_in.shape[0]//2), w:w+(img_in.shape[1]//2)]
                mask_out = mask_in[h:h+h+(img_in.shape[0]//2), w:w+(img_in.shape[1]//2)]
                fname = get_rand_name()
                cv2.imwrite(os.path.join(dst_frames, fname + '.png'), img_out)
                cv2.imwrite(os.path.join(dst_masks, fname + '.png'), mask_out)


def img_crop_8(src_frames, src_masks, dst_frames, dst_masks):
    '''
    crop a (frame & mask) into 8: from 720x1280x1 to 360x320x8
    - run once
    '''
    for img_id in os.listdir(src_frames):
        img_fname = img_id.split('.')[0]
        # Load frames and masks from npy source
        img_in = cv2.imread(os.path.join(src_frames, img_fname + '.png'))
        mask_in = np.load(os.path.join(src_masks, img_fname + '.npy'))

        for w in range(0, 1280, 320):
            for h in range (0, 720, 360):
                img_out = img_in[h:h+360, w:w+320]
                mask_out = mask_in[h:h+360, w:w+320]
                fname = get_rand_name()
                cv2.imwrite(os.path.join(dst_frames, fname + '.png'), img_out)
                cv2.imwrite(os.path.join(dst_masks, fname + '.png'), mask_out)

# Create dataset with the appropriate size
def get_frame(frame_path, height, width):
    frame = cv2.imread(frame_path, 1)
    frame = np.float32(cv2.resize(frame, (height, width)))
    return frame

def get_mask(mask_path, height, width, n_classes):
    # Mask depth should be the same as the number of classes
    mask_depth = np.zeros((height, width, n_classes))  # Each depth will be fed one-by-one corresponding to the classs label
    mask = cv2.imread(mask_path, 1)
    mask = cv2.resize(mask, (height, width))
    mask = mask[:, :, 0]  # take only the first channel

    for i in range(n_classes):
        # One depth
        mask_depth[:, :, i] = (mask == i).astype(int)  # OneHotEncode True-False then 0-1

    return mask_depth

