#### Summary
This repository is holding experiments related to my M.Tech thesis. <br>
The project is aiming to address the road extraction problem in end-to-end fashion. 
Our dataset is a set of aerial images taken from UAV(drones) from local areas within the NIT Rourkela campus. 
Throughout the experiments, we are benchmarking different state-of-the-art models and taking advantages to address our problem. The main objective of this project is to build an effective CNN model, being able to distinguish roads from occlusion and background
and able to generalize to later extension as well as to build our own dataset. 

> **Please note:** the development is undergoing and details will 
> gradually be provided below. 

## Prerequisites
There are a few but important prerequisites that (may) need to be installed before making any change. Note that the code has been tested with `python==3.6.7` 

### Create new [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) environment 
```bash
conda create --name ENV_NAME python=3.6.7
source activate ENV_NAME
```
### Install required packages
```bash 
pip install -r requirements.txt
```

## Directory structure
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
* `/models` where to store in and load from the models.  


## Usage
* For training and prediction, refer to `train.sh` and `predict.sh` respectively to define necessary environment variables. 
```bash
# How to train? 
sh train.sh
# How to predict?
sh predict.sh
```

##### Major update
| Timeline | Comments |  Reference |
| -------- | -------- | -----------| 
| 18-12-2019 | Add U-Net model    | [paper](https://arxiv.org/abs/1505.04597) 
| 22-02-2020 | Add BCDU-Net model | [paper](https://arxiv.org/abs/1909.00166)
| 29-02-2020 | FC-DenseNet model  | [paper](https://arxiv.org/abs/1611.09326)

