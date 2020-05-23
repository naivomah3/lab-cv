#### Summary
>This repository is holding experiments related to my M.Tech thesis.
The project is aiming to address the road extraction problem in 
>an end-to-end fashion. Our dataset is a set of aerial images taken 
>from UAV(drones) from local areas within the NIT Rourkela campus. 
>Throughout the experiments, we are benchmarking different 
>state-of-the-art models and taking advantage of the technique being used to tackle our problem. 
>The main objective of this project is to build an effective CNN model, 
>being able to distinguish roads from occlusion and background and able 
>to generalize to later extension as well as to build our own dataset. 

> **Please note:** the development is undergoing and details will 
> gradually be provided below. 

## Prerequisites
There are a few but important dependencies that need to be installed before making any change. 
Note that the code has been tested using `python==3.6.7`. 
```bash 
pip install -r requirements.txt
```
Few parameters(environment variables) have to be set according to the need.
* In `train.sh` and `predict.sh`
```bash
export PRE_TRAINED=True   # set to False if not loading the pre-trained weight 
export WEIGHTS_IN_PATH=path/to/the/weights.h5    # full path of the weight
``` 

### Create new [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) environment 
```bash
conda create --name ENV_NAME python=3.6.7
source activate ENV_NAME
```
### Install required packages


## Dataset directory structure
* `/input` contains images/frames and their respective masks within each 
set(training/validation/testing). Within `mask`, the ground truth set would be a 2D matrix of dimension `(height x width)` and each pixel's value maps the spatial distribution of each respective label. 
As an example, please find [here](https://github.com/naivomah3/lab-cv/blob/master/notebooks/image-preprocessing.ipynb)

```bash
/input
├── testing # Testing data 
│   ├── images
│   └── mask
├── training   # Training data 
│   ├── images
│   └── mask
└── validation # Validation data
    ├── images
    └── mask
```
* `/output` contains predicted masks referred from input/prediction along with their respective frames within each respective subfolders and named the same way to avoid confusion. `masks` and `prediction` sets would be a 2D matrix of dimension `(height x width x 3)` colored with respect to the spatial distribution of each class/label. As an example, please find here
```bash
/output   
└── prediction # Prediction data 
    ├── images
    ├── mask
    └── prediction
```
* `/models` where to store and load from the model weights. 



## Usage
* For training and prediction, refer to `train.sh` and `predict.sh` respectively to define necessary environment variables. 
```bash
# How to train? 
sh train.sh
# How to predict?
sh predict.sh
```

##### Major update
| Timeline |         Comments           | Source                                                                                       |        Reference                          |
| -------- | -------------------------- | -------------------------------------------------                                            | ----------------------------------------  | 
| 18-12-2019 | U-Net model added        |    [source](https://github.com/naivomah3/lab-cv/blob/master/src/models/unet.py)              | [paper](https://arxiv.org/abs/1505.04597) | 
| 22-02-2020 | BCDU-Net model added     |    [source](https://github.com/naivomah3/lab-cv/blob/master/src/models/bcd_unet.py)          | [paper](https://arxiv.org/abs/1909.00166) |
| 29-02-2020 | FC-DenseNet model added  |    [source](https://github.com/naivomah3/lab-cv/blob/master/src/models/fc_densenet.py)        | [paper](https://arxiv.org/abs/1611.09326) |
| 08-03-2020 | DeepLab-v3+ model added                          |   [source](https://github.com/naivomah3/lab-cv/blob/master/src/models/deeplab_v3_plus.py)     | [paper](https://arxiv.org/abs/1802.02611) |
| 08-03-2020 | FCN model added                                  |  [source](https://github.com/naivomah3/lab-cv/blob/master/src/models/fcn_8s.py)               | [paper](https://arxiv.org/abs/1411.4038)  |
| 18-04-2020 | SegNet model added                               |   [source](https://github.com/naivomah3/lab-cv/blob/master/src/models/segnet.py)              | [paper](https://arxiv.org/abs/1511.00561) |
| 21-05-2020 | Depth-wise Separable UNet model added             |   [source](https://github.com/naivomah3/lab-cv/blob/master/src/models/dw_unet.py)              | proposed model  |
| 21-05-2020 | Depth-wise Separable Dense UNet model added       |   [source](https://github.com/naivomah3/lab-cv/blob/master/src/models/dw_dense_unet.py)              | proposed model  |


