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
export TRAIN_BATCH_SIZE=2
export VAL_BATCH_SIZE=4
export TRAIN_STEPS_PER_EPOCH=1000  # train_len(ex. 800 images) = batch_size(20) * steps_per_epoch(40)
export VAL_STEPS_PER_EPOCH=120   # val_len(200) = batch_size(5) * steps_per_epoch(40)
export NO_EPOCHS=100
export IN_HEIGHT=320
export IN_WIDTH=320 #
export MODELS_OUT_PATH=models/
export PROBLEM=road_seg
export LABELS_FILE=labels.txt
export MODEL=unet
export BACKBONE=vgg16
# --------------------
# Define script loader
# --------------------
python -m src.train
#======================================================