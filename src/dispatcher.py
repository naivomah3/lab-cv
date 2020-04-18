import os
from src.models.unet import UNet
from src.models.bcd_unet import BCD_UNet_D1, BCD_UNet_D3
from src.models.fc_densenet import FC_DenseNet
from src.models.deeplab_v3_plus import DSC_DeepLab_v3_plus
from src.models.fcn_8s import FCN_8s
from src.models.dense_unet import Dense_UNet
from src.models.segnet import SegNet

WEIGHTS_IN_PATH = os.environ.get("WEIGHTS_IN_PATH") # should be the pre-trained models_weights full path
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

    # Fully Convolutional Network
    'fcn_8s': FCN_8s(pre_trained=is_pretrained,
                    weights_path=WEIGHTS_IN_PATH,
                    n_classes=CLASSES,
                    input_h=IN_HEIGHT,
                    input_w=IN_WIDTH,
                    model_name=f"{MODEL}"),

    # Dense-U-Net(proposed model for experiments)
    'dense_unet': Dense_UNet(pre_trained=is_pretrained,
                    weights_path=WEIGHTS_IN_PATH,
                    n_classes=CLASSES,
                    input_h=IN_HEIGHT,
                    input_w=IN_WIDTH,
                    model_name=f"{MODEL}"),

    # SegNet: A deep convolutional encoder-decoder architecture for image segmentation
    'seg_net': SegNet(pre_trained=is_pretrained,
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
    # Fully Convolutional Network: 3 variants
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
    # DeepLabV3+: Depthwise Separable Convolution with 2 backbones available: Mobilenetv2 and Xception
    'deeplab_v3_plus': DSC_DeepLab_v3_plus(pre_trained=is_pretrained,
                                           weights_path=WEIGHTS_IN_PATH,
                                           n_classes=CLASSES,
                                           backbone=BACKBONE,
                                           input_h=IN_HEIGHT,
                                           input_w=IN_WIDTH,
                                           model_name=f"{MODEL}"),

    }
