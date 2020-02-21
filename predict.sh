#======================================================
# PREDICTION / TESTING
#======================================================
# --------------------
# Load environment variables
# --------------------
# Input
export FRAMES_TEST_IN_PATH=input/testing/images # Where are located testing frames(input)
export MASKS_TEST_IN_PATH=input/testing/mask # Where are located testing masks(not colored)

# Output
export FRAMES_TEST_OUT_PATH=output/prediction/images # Where are located testing frames(output)
export MASKS_TEST_OUT_PATH=output/prediction/mask # Where are located testing masks(colored)
export MASKS_PREDICT_OUT_PATH=output/prediction/prediction # Where are located predicted masks(colored)
#export MODEL=$1
export NO_CLASSES=3 # Number of classes
export INPUT_HEIGHT=320 # Image input height(frames+masks/training+validation sets)
export INPUT_WIDTH=320 # Image input width(frames+masks/training+validation sets)
export WEIGHTS_PATH=models/unet_train_on_11_26_2019.h5 # Where is located the model
# --------------------
# Define launcher
# --------------------
python -m src.predict
#======================================================
