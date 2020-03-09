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
export FRAMES_TEST_OUT_PATH=input/prediction/images/ # Where are located testing frames(output)
export MASKS_TEST_OUT_PATH=input/prediction/mask # Where are located testing masks(colored)
export MASKS_PREDICT_OUT_PATH=input/prediction/prediction # Where are located predicted masks(colored)
#export MODEL=$1
export IN_HEIGHT=320 # Image input height(frames+masks/training+validation sets)
export IN_WIDTH=320 # Image input width(frames+masks/training+validation sets)
export PRE_TRAINED=True
export WEIGHTS_IN_PATH=models_weights/road_seg_udd_fcn_densenet_103_vgg16_320_320_weights_06_03_20-20_17_PM/road_seg_udd_fcn_densenet_103_vgg16_320_320_weights_06_03_20-20_17_PM_36.h5 # Where is located the model
export LABELS_FILE=labels.txt
# MODEL:
# {unet, bcd_unet_d1, bcd_unet_d3, fcn_densenet_56, fcn_densenet_67, fcn_densenet_103}
export MODEL=fcn_densenet_103
# --------------------
# Define launcher
# --------------------
python -m src.predict
#======================================================
