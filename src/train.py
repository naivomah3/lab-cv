import os
from datetime import datetime
from src.data_generator import data_generator, data_loader
from src.callbacks import get_callbacks
from src.models import UNet
from src.dispatcher import MODELS

# Load all environment variables
FRAMES_TRAIN_PATH = os.environ.get("FRAMES_TRAIN_PATH")
MASKS_TRAIN_PATH = os.environ.get("MASKS_TRAIN_PATH")
FRAMES_VAL_PATH = os.environ.get("FRAMES_VAL_PATH")
MASKS_VAL_PATH = os.environ.get("MASKS_VAL_PATH")
# MODEL&BACKBONE
MODEL = os.environ.get("MODEL")
BACKBONE = os.environ.get("BACKBONE")
WEIGHTS_OUT_PATH = os.environ.get("WEIGHTS_OUT_PATH")
WEIGHTS_IN_PATH = os.environ.get("WEIGHTS_IN_PATH")
# Batch-size
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
VAL_BATCH_SIZE = int(os.environ.get("VAL_BATCH_SIZE"))
# Steps per epoch:
TRAIN_STEPS_PER_EPOCH = int(os.environ.get("TRAIN_STEPS_PER_EPOCH"))
VAL_STEPS_PER_EPOCH = int(os.environ.get("VAL_STEPS_PER_EPOCH"))
# Number of epochs
# Frames&masks input dimensions
IN_HEIGHT = int(os.environ.get("IN_HEIGHT"))
IN_WIDTH = int(os.environ.get("IN_WIDTH"))
NO_EPOCHS = int(os.environ.get("NO_EPOCHS"))

# get problem name for naming history/model
P_NAME = os.environ.get("PROBLEM")

# load file containing list of classes
LABELS_FILE = os.environ.get("LABELS_FILE")
with open(LABELS_FILE, 'r') as file:
    CLASSES = len(list(file))
if not CLASSES:
    raise Exception("Unable to load label file")

if __name__ == '__main__':
    generator = True

    # define where to save the model
    model_name = f"{P_NAME}_{MODEL}_{BACKBONE}_{IN_HEIGHT}_{IN_WIDTH}_weights_{datetime.now().strftime('%d_%m_%y-%H_%M_%p')}"
    callbacks = get_callbacks(weights_path=WEIGHTS_OUT_PATH, model_name=model_name)

    # Load model from dispatcher and build
    network = MODELS[MODEL]
    model = network.build()
    model.summary()


    # # Load data using generator
    # if generator:
    #     train_generator = data_generator(frames_path=FRAMES_TRAIN_PATH,
    #                                      masks_path=MASKS_TRAIN_PATH,
    #                                      fnames=os.listdir(FRAMES_TRAIN_PATH),
    #                                      n_classes=CLASSES,
    #                                      input_h=IN_HEIGHT,
    #                                      input_w=IN_WIDTH,
    #                                      batch_size=TRAIN_BATCH_SIZE,
    #                                      is_resizable=False,
    #                                      training=True)
    #
    #     val_generator = data_generator(frames_path=FRAMES_VAL_PATH,
    #                                    masks_path=MASKS_VAL_PATH,
    #                                    fnames=os.listdir(FRAMES_VAL_PATH),
    #                                    n_classes=CLASSES,
    #                                    input_h=IN_HEIGHT,
    #                                    input_w=IN_WIDTH,
    #                                    batch_size=VAL_BATCH_SIZE,
    #                                    is_resizable=False,
    #                                    training=True)
    #
    #     history = model.fit_generator(generator=train_generator,
    #                                   steps_per_epoch=TRAIN_STEPS_PER_EPOCH,  # train_len(800 images) = batch_size(20) * steps_per_epoch(40)
    #                                   validation_data=val_generator,
    #                                   validation_steps=VAL_STEPS_PER_EPOCH,  # val_len(150 images) = batch_size * validation_steps
    #                                   epochs=NO_EPOCHS,
    #                                   verbose=1,
    #                                   callbacks=callbacks)
    #
    # # Load data without generator
    # else:
    #     test_frames, test_masks = data_loader(frames_path=FRAMES_TRAIN_PATH,
    #                                           masks_path=MASKS_TRAIN_PATH,
    #                                           input_h=IN_HEIGHT,
    #                                           input_w=IN_WIDTH,
    #                                           n_classes=CLASSES,
    #                                           fnames=os.listdir(FRAMES_TRAIN_PATH),
    #                                           is_resizable=False,
    #                                           training=True)
    #
    #     val_frames, val_masks = data_loader(frames_path=FRAMES_VAL_PATH,
    #                                           masks_path=MASKS_VAL_PATH,
    #                                           input_h=IN_HEIGHT,
    #                                           input_w=IN_WIDTH,
    #                                           n_classes=CLASSES,
    #                                           fnames=os.listdir(FRAMES_VAL_PATH),
    #                                           is_resizable=False,
    #                                           training=True)
    #
    #     history = model.fit(test_frames,
    #                         test_masks,
    #                         validation_data=(val_frames, val_masks),
    #                         epochs=NO_EPOCHS,
    #                         batch_size=TRAIN_BATCH_SIZE,
    #                         verbose=1,
    #                         callbacks=callbacks)
        # Predictive probs

