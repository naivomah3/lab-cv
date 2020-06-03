#======================================================
# PREDICTION / TESTING
#======================================================
# --------------------
# Load environment variables
# --------------------
# INPUT
export FRAMES_TEST_IN_PATH=input/testing/images # Where testing frames are located (input)
export MASKS_TEST_IN_PATH=input/testing/mask # Where testing masks are located (not colored)
# OUTPUT
export FRAMES_TEST_OUT_PATH=input/prediction/images/ # Where testing frames will be generated(output)
export MASKS_TEST_OUT_PATH=input/prediction/mask # Where testing masks are will be generated(colored)
export MASKS_PREDICT_OUT_PATH=input/prediction/prediction # Where predicted masks are will be generated(colored)
#export MODEL=$1
export IN_HEIGHT=512 # Image height
export IN_WIDTH=512 # Image width
# If PRE_TRAINED is True, set the variable `WEIGHTS_IN_PATH` to the weights(.h5) full path
export PRE_TRAINED=True
export WEIGHTS_IN_PATH=models_weights/road_seg_udd_fcn_densenet_103_vgg16_320_320_weights_06_03_20-20_17_PM/road_seg_udd_fcn_densenet_103_vgg16_320_320_weights_06_03_20-20_17_PM_36.h5 # Where is located the model
export LABELS_FILE=labels.txt
# MODEL:
# unet,
# fcn_8s,
# dense_unet,
# dw_unet,
# dw_dense_unet_d10, dw_dense_unet_d15, dw_dense_unet_d20
# seg_net,
# bcd_unet_d1, bcd_unet_d3,
# fcn_densenet_56, fcn_densenet_67, fcn_densenet_103,
# deeplab_v3_plus
export MODEL=fcn_densenet_103
# --------------------
# Define launcher
# --------------------
python -m src.predict
#======================================================
