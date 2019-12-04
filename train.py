import os
from keras.callbacks import ModelCheckpoint
from  helpers.data_generator import image_data_generator
from models.vgg_fcn8 import vgg_fcn8s


def train_model(train_frames_path, train_masks_path, val_frames_path, val_masks_path, root_path):

    train_fnames = os.listdir(train_frames_path)
    val_fnames = os.listdir(val_frames_path)

    train_generator = image_data_generator(train_frames_path, train_masks_path, train_fnames, batch_size=25)
    val_generator = image_data_generator(val_frames_path, val_masks_path, val_fnames, batch_size=10)

    # Create checkpoint callbacks for history
    check_point_path = os.path.join(root_path, "vgg_fcn8s.model")
    callbacks = ModelCheckpoint(check_point_path, monitor='dice',
                                verbose=1, mode='max',
                                save_best_only=True,
                                save_weights_only=False,
                                period=2)  # every 2 epochs
    callbacks_list = [callbacks]

    # Create the model
    model = vgg_fcn8s()
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=34,  # 850 images = batch_size * steps_per_epoch
                                  validation_data=val_generator,
                                  validation_steps=15,  # 150 images = batch_size * validation_steps
                                  epochs=100,
                                  verbose=2,
                                  callbacks=callbacks_list
                                  )

    return history