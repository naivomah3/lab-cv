import os
import numpy as np
import cv2

from src.data_generator import data_generator, data_loader
from src.engine import get_rand_name, color_img
from src.unet import UNet


# input set
FRAMES_TEST_IN_PATH = os.environ.get("FRAMES_TEST_IN_PATH")
MASKS_TEST_IN_PATH = os.environ.get("MASKS_TEST_IN_PATH")
# output set
FRAMES_TEST_OUT_PATH = os.environ.get("FRAMES_TEST_OUT_PATH")
MASKS_TEST_OUT_PATH = os.environ.get("MASKS_TEST_OUT_PATH")
MASKS_PREDICT_OUT_PATH = os.environ.get("MASKS_PREDICT_OUT_PATH")
# Model path
UNET_MODEL_PATH = os.environ.get("UNET_MODEL_PATH")
# Number of classes [2:'binary', >2:'multilabel']
NO_CLASSES = int(os.environ.get("NO_CLASSES"))
# Frames&masks input dimension
INPUT_HEIGHT = int(os.environ.get("INPUT_HEIGHT"))
INPUT_WIDTH = int(os.environ.get("INPUT_WIDTH"))

if __name__ == '__main__':
    is_generator = False
    test_frames = None
    test_masks = None

    if not os.path.isfile(UNET_MODEL_PATH):
        quit(f"Model file not found {UNET_MODEL_PATH}")
    if not os.path.isdir(FRAMES_TEST_IN_PATH) or not os.path.isdir(MASKS_TEST_IN_PATH):
        quit(f"Directory not found")

    # Create model
    model = UNet.build(pre_trained=True,
                       model_path=UNET_MODEL_PATH,
                       n_classes=NO_CLASSES,
                       input_h=INPUT_HEIGHT,
                       input_w=INPUT_WIDTH)

    # If images are loaded from generator
    if is_generator:
        test_data_generated = data_generator(frames_path=FRAMES_TEST_IN_PATH,
                                             masks_path=MASKS_TEST_IN_PATH,
                                             fnames=os.listdir(MASKS_TEST_IN_PATH),
                                             input_h=INPUT_HEIGHT,
                                             input_w=INPUT_WIDTH,
                                             n_classes=NO_CLASSES,
                                             batch_size=10,
                                             is_resizable=True)
        predicted_masks = model.predict_generator(test_data_generated, steps=15)

    # If images are loaded from builder(no generator)
    else:
        test_frames, test_masks = data_loader(frames_path=FRAMES_TEST_IN_PATH,
                                              masks_path=MASKS_TEST_IN_PATH,
                                              input_h=INPUT_HEIGHT,
                                              input_w=INPUT_WIDTH,
                                              n_classes=NO_CLASSES,
                                              fnames=os.listdir(MASKS_TEST_IN_PATH),
                                              is_resizable=True)
        # Predictive probs
        predicted_masks = model.predict(test_frames, batch_size=10)

    # Map mask & predicted mask as 2-D matrix of the respective class
    # dim: [n_samples, h, w, 1]
    test_masks = np.argmax(test_masks, axis=3)
    predicted_masks = np.argmax(predicted_masks, axis=3)

    # Save Frames, GroundTruth and Predicted masks
    for i in range(test_frames.shape[0]):
        fname = get_rand_name()
        # Save original frame
        cv2.imwrite(os.path.join(FRAMES_TEST_OUT_PATH, fname + '.png'),
                    test_frames[i])
        # Save ground truth mask
        cv2.imwrite(os.path.join(MASKS_TEST_OUT_PATH, fname + '.png'),
                    color_img(test_masks[i], NO_CLASSES))
        # Save predicted mask
        cv2.imwrite(os.path.join(MASKS_PREDICT_OUT_PATH, fname + '.png'),
                    color_img(predicted_masks[i], NO_CLASSES))





