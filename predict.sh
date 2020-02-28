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
export FRAMES_TEST_OUT_PATH=input/prediction/images # Where are located testing frames(output)
export MASKS_TEST_OUT_PATH=input/prediction/mask # Where are located testing masks(colored)
export MASKS_PREDICT_OUT_PATH=input/prediction/prediction # Where are located predicted masks(colored)
#export MODEL=$1
export INPUT_HEIGHT=512 # Image input height(frames+masks/training+validation sets)
export INPUT_WIDTH=512 # Image input width(frames+masks/training+validation sets)
export WEIGHTS_PATH=models/road_seg_nit_unet_vgg16_320_320_weights_24_02_20-00_32_AM/road_seg_nit_unet_vgg16_320_320_weights_24_02_20-00_32_AM_99.h5 # Where is located the model
export LABELS_FILE=labels.txt
# MODEL:
# {unet, bcd_unet_d1, bcd_unet_d3, fcn_densenet_56, fcn_densenet_67, fcn_densenet_103}
export MODEL=fcn_densenet_103
# --------------------
# Define launcher
# --------------------
python -m src.predict
#======================================================
