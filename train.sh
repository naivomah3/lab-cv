#======================================================
# TRAINING
#======================================================
# --------------------
# Load environment variables
# --------------------
export FRAMES_TRAIN_PATH=input/training/images
export MASKS_TRAIN_PATH=input/training/mask
export FRAMES_VAL_PATH=input/validation/images
export MASKS_VAL_PATH=input/validation/mask
export NO_CLASSES=3
export INPUT_HEIGHT=320
export INPUT_WIDTH=320
export MODELS_OUT_PATH=models/unet_train_on_11_26_2019.h5
# --------------------
# Define script loader
# --------------------
python -m src.train
#======================================================