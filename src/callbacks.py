import os
import keras

from src.utils import mk_dir

def get_callbacks(model_path=None, model_name=None,):

    if not mk_dir(model_path, model_name):
        raise Exception("Provide path to store the model")

    callbacks = [
        # callback 1: saving history
        keras.callbacks.CSVLogger(os.path.join(model_path, model_name, f"{model_name}.csv"),
                                  append=True),
        # callback 2: saving model
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, model_name, f"{model_name}.h5"),
                                        monitor='dice_multilabel',
                                        verbose=1,
                                        mode='max',
                                        save_best_only=True,
                                        save_weights_only=True,
                                        period=2),  # monitor every 2 epochs
        # Callback 3: LR decay
        # cb_lrdecay = LearningRateScheduler(step_decay)

        # Callback 4: Early stopping if no improvement within 30 epochs
        keras.callbacks.EarlyStopping(monitor='val_dice', mode='max', patience=50),

        # Callback 5: Reduce if no improvement
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        #                                   factor=0.9,
        #                                   patience=5,
        #                                   min_lr=1e-5,
        #                                   verbose=1),
    ]


    return callbacks