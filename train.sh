#======================================================
# TRAINING
#======================================================
# --------------------
# Load environment variables
# --------------------
export FRAMES_TRAIN_PATH=input/training/images # Where are located training frames
export MASKS_TRAIN_PATH=input/training/mask # Where are located training masks
export FRAMES_VAL_PATH=input/validation/images # Where are located validation frame
export MASKS_VAL_PATH=input/validation/mask # Where are located validation masks
export NO_CLASSES=3 # Number of classes
export INPUT_HEIGHT=320 # Image input height(frames+masks/training+validation sets)
export INPUT_WIDTH=320 # Image input width(frames+masks/training+validation sets)
export MODELS_OUT_PATH=models/unet_train_on_11_26_2019.h5 # Where is located model callback
# --------------------
# Define launcher
# --------------------
python -m src.train
#======================================================
