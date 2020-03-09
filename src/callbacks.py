import os
import keras
from src.utils import mk_dir
from src.utils import lr_decay

def get_callbacks(weights_path=None, model_name=None, ):

    if not mk_dir(weights_path, model_name):
        raise Exception("Provide path to store the model")

    callbacks = [
        # CB 1: saving history
        keras.callbacks.CSVLogger(os.path.join(weights_path, model_name, f"{model_name}.csv"),
                                  append=True),
        # CB 2: saving model/models_weights
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_path, model_name, model_name) + "_{epoch:02d}.h5",
                                        monitor='dice',
                                        verbose=1,
                                        mode='max',
                                        #save_best_only=True,
                                        save_weights_only=True,
                                        period=1),  # save models_weights each epoch
        # CB 3: Early stopping if no improvement within 20 epochs
        keras.callbacks.EarlyStopping(monitor='val_dice', mode='max', patience=20),

        # CB 4: LR decay
        # keras.callbacks.LearningRateScheduler(lr_decay),

        # CB 5: Reduce if no improvement
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        #                                   factor=0.9,
        #                                   patience=5,
        #                                   min_lr=1e-5,
        #                                   verbose=1),
    ]

    return callbacks