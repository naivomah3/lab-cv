import os

from datetime import date
from keras.callbacks import ModelCheckpoint, CSVLogger

from src.data_generator import data_generator
from src.unet import UNet

# Load Env. variables
FRAMES_TRAIN_PATH = os.environ.get("FRAMES_TRAIN_PATH")
MASKS_TRAIN_PATH = os.environ.get("MASKS_TRAIN_PATH")
FRAMES_VAL_PATH = os.environ.get("FRAMES_VAL_PATH")
MASKS_VAL_PATH = os.environ.get("MASKS_VAL_PATH")
#MODEL = os.environ.get("MODEL")
MODELS_OUT_PATH = os.environ.get("MODELS_OUT_PATH")
# Number of classes [2:'binary', >2:'multilabel']
NO_CLASSES = int(os.environ.get("NO_CLASSES"))
# Frames&masks input dimension
INPUT_HEIGHT = int(os.environ.get("INPUT_HEIGHT"))
INPUT_WIDTH = int(os.environ.get("INPUT_WIDTH"))

if __name__ == '__main__':
    # Image generator
    train_generator = data_generator(frames_path=FRAMES_TRAIN_PATH,
                                       masks_path=MASKS_TRAIN_PATH,
                                       fnames=os.listdir(FRAMES_TRAIN_PATH),
                                       n_classes=NO_CLASSES,
                                       input_h=INPUT_HEIGHT,
                                       input_w=INPUT_WIDTH,
                                       batch_size=25,
                                       is_resizable=True)

    val_generator = data_generator(frames_path=FRAMES_VAL_PATH,
                                       masks_path=MASKS_VAL_PATH,
                                       fnames=os.listdir(FRAMES_VAL_PATH),
                                       n_classes=NO_CLASSES,
                                       input_h=INPUT_HEIGHT,
                                       input_w=INPUT_WIDTH,
                                       batch_size=10,
                                       is_resizable=True)

    # Save model
    model_checkpoint_path = os.path.join(MODELS_OUT_PATH, f"unet_train_on_{date.today().strftime('%m_%d_%y')}.h5")

    # callback 1: saving history
    cb_csvlogger = CSVLogger(os.path.join(MODELS_OUT_PATH, f"unet_log_on_{date.today().strftime('%m_%d_%y')}.csv"), append=True)
    # callback 2: saving model
    cb_checkpoint = ModelCheckpoint(model_checkpoint_path,
                                    monitor='dice',
                                    verbose=1,
                                    mode='max',
                                    save_best_only=True,
                                    save_weights_only=False,
                                    period=2)  # monitor every 2 epochs


    callback_list = [cb_checkpoint, cb_csvlogger]

    # Create model
    model = UNet.build(pre_trained=False,
                       model_path=MODELS_OUT_PATH,
                       n_classes=NO_CLASSES,
                       input_h=INPUT_HEIGHT,
                       input_w=INPUT_WIDTH)

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=32,  # train_len(850 images) = batch_size * steps_per_epoch
                                  validation_data=val_generator,
                                  validation_steps=20,  # val_len(150 images) = batch_size * validation_steps
                                  epochs=200,
                                  verbose=2,
                                  callbacks=callback_list)
