#======================================================
# TRAINING
#======================================================
# --------------------
# Load environment variables
# --------------------
#export FRAMES_TRAIN_PATH=input/training/images
#export MASKS_TRAIN_PATH=input/training/mask
#export FRAMES_VAL_PATH=input/validation/images
#export MASKS_VAL_PATH=input/validation/mask
# --------------------
# Define script loader
# --------------------
#python -m src.train
# --------------------
# How can I train?
# --------------------
#sh src.train
#======================================================


#======================================================
# PREDICTION / TESTING
#======================================================
# --------------------
# Load environment variables
# --------------------
# Input
export FRAMES_TEST_IN_PATH=input/prediction/images
export MASKS_TEST_IN_PATH=input/prediction/mask
# Output
export FRAMES_TEST_OUT_PATH=output/prediction/images
export MASKS_TEST_OUT_PATH=output/prediction/mask
export MASKS_PREDICT_OUT_PATH=output/prediction/predict
#export MODEL=$1
export UNET_MODEL_PATH=models/unet_train_on_11_26_2019.h5
# --------------------
# Define script loader
# --------------------
python -m src.predict
# --------------------
# How can I predict?
# --------------------
#sh src.predict
#======================================================
