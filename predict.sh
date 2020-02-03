#======================================================
# PREDICTION / TESTING
#======================================================
# --------------------
# Load environment variables
# --------------------
# Input
export FRAMES_TEST_IN_PATH=input/testing/images
export MASKS_TEST_IN_PATH=input/testing/mask

# Output
export FRAMES_TEST_OUT_PATH=output/prediction/images
export MASKS_TEST_OUT_PATH=output/prediction/mask
export MASKS_PREDICT_OUT_PATH=output/prediction/prediction
#export MODEL=$1
export NO_CLASSES=3
export INPUT_HEIGHT=320
export INPUT_WIDTH=320
export UNET_MODEL_PATH=models/unet_train_on_11_26_2019.h5
# --------------------
# Define script loader
# --------------------
python -m src.predict
#======================================================
