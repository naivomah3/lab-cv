import os
from src.models import UNet, BCD_UNet_D1, BCD_UNet_D3, FC_DenseNet, DSC_DeepLab_v3_plus

WEIGHTS_IN_PATH = os.environ.get("WEIGHTS_IN_PATH") # should be the pre-trained weights full path
IN_HEIGHT = int(os.environ.get("IN_HEIGHT"))
IN_WIDTH = int(os.environ.get("IN_WIDTH"))
LABELS_FILE = os.environ.get("LABELS_FILE")
MODEL = os.environ.get("MODEL")
BACKBONE = os.environ.get("BACKBONE")
is_pretrained = False if os.environ.get("PRE_TRAINED") == "False" else True
with open(LABELS_FILE, 'r') as file:
    CLASSES = len(list(file))
if not CLASSES:
    raise Exception("Unable to load label file")

MODELS = {
    # Fully Convolutional Network
    'unet': UNet(pre_trained=is_pretrained,
                weights_path=WEIGHTS_IN_PATH,
                n_classes=CLASSES,
                input_h=IN_HEIGHT,
                input_w=IN_WIDTH,
                model_name=f"{MODEL}"),
    # Bi-Directional ConvLSTM U-Net with Densely Connected Convolutions
    'bcd_unet_d3': BCD_UNet_D3(pre_trained=is_pretrained,
                                weights_path=WEIGHTS_IN_PATH,
                                n_classes=CLASSES,
                                input_h=IN_HEIGHT,
                                input_w=IN_WIDTH,
                                model_name=f"{MODEL}"),
    'bcd_unet_d1': BCD_UNet_D1(pre_trained=is_pretrained,
                                    weights_path=WEIGHTS_IN_PATH,
                                    n_classes=CLASSES,
                                    input_h=IN_HEIGHT,
                                    input_w=IN_WIDTH,
                                    model_name=f"{MODEL}"),
    # Fully Convolutional Network
    'fcn_densenet_103': FC_DenseNet(pre_trained=is_pretrained,
                                weights_path=WEIGHTS_IN_PATH,
                                n_classes=CLASSES,
                                input_h=IN_HEIGHT,
                                input_w=IN_WIDTH,
                                model_name=f"{MODEL}"),
    'fcn_densenet_56': FC_DenseNet(pre_trained=is_pretrained,
                                    weights_path=WEIGHTS_IN_PATH,
                                    n_classes=CLASSES,
                                    input_h=IN_HEIGHT,
                                    input_w=IN_WIDTH,
                                    model_name=f"{MODEL}"),
    'fcn_densenet_67': FC_DenseNet(pre_trained=is_pretrained,
                                    weights_path=WEIGHTS_IN_PATH,
                                    n_classes=CLASSES,
                                    input_h=IN_HEIGHT,
                                    input_w=IN_WIDTH,
                                    model_name=f"{MODEL}"),
    # Depthwise Separable Convolution with 2 backbones available: Mobilenetv2 and Xception
    'deeplab_v3_plus': DSC_DeepLab_v3_plus(pre_trained=is_pretrained,
                                           weights_path=WEIGHTS_IN_PATH,
                                           n_classes=CLASSES,
                                           backbone=BACKBONE,
                                           input_h=IN_HEIGHT,
                                           input_w=IN_WIDTH,
                                           model_name=f"{MODEL}"),

    }
