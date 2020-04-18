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
export TRAIN_BATCH_SIZE=4
export VAL_BATCH_SIZE=6
export TRAIN_STEPS_PER_EPOCH=4131  # train_len(ex. 800) = batch_size(20) * steps_per_epoch(40)
export VAL_STEPS_PER_EPOCH=918     # val_len(200) = batch_size(5) * steps_per_epoch(40)
export NO_EPOCHS=100
export IN_HEIGHT=320
export IN_WIDTH=320 #
export WEIGHTS_OUT_PATH=models_weights/
export PRE_TRAINED=False
export WEIGHTS_IN_PATH=models_weights/road_seg_nit_unet_vgg16_320_320_weights_24_02_20-00_32_AM/road_seg_nit_unet_vgg16_320_320_weights_24_02_20-00_32_AM_99.h5    # Full path of pre-trained models_weights
export PROBLEM=road_seg_udd
export LABELS_FILE=labels.txt
# MODEL:
# {unet, fcn_8s, dense_unet, seg_net, bcd_unet_d1, bcd_unet_d3, fcn_densenet_56, fcn_densenet_67, fcn_densenet_103, deeplab_v3_plus}
export MODEL=seg_net
# BACKBONE:
# {default, vgg16, xception, mobilenetv2}
export BACKBONE=default
# --------------------
# Define script loader
# --------------------
python -m src.train
#======================================================