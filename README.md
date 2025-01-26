# RPMC

Official implementation of *Semi-supervised Medical Image Segmentation Using Reliable Pseudo-label-based Mixed Consistency Learning*  

**Authors:**  

> Yuanbin Chen, Hui Tang, Tao Wang, Longxuan Zhao, Tao Tan, Xinlin Zhang, Tong Tong


## Requirements
This repository is based on PyTorch 1.12.1, CUDA 11.6 and Python 3.8; All experiments in our paper were conducted on a single NVIDIA GeForce RTX 3090 24GB GPU.

## Data 

Following previous works, we have validated our method on three benchmark datasets, including 2018 Atrial Segmentation Challenge, Pancreas-CT dataset and Multimodal Brain Tumor Segmentation Challenge 2019.  
It should be noted that we do not have permissions to redistribute the data. Thus, for those who are interested, please follow the instructions below and process the data, or you will get a mismatching result compared with ours.

### Data Preparation

#### Download

Atrial Segmentation: http://atriaseg2018.cardiacatlas.org/  
Pancreas-CT dataset: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT  
Multimodal Brain Tumor Segmentation Challenge 2019: https://www.med.upenn.edu/cbica/brats-2019/

#### Data Split

We split the data following previous works. Detailed split could be found in folder `data`, which are stored in .list files.

#### Data Preprocessing

Download the data from the url above, then run the script `./code/dataloaders/la_heart_processing.py` and `./code/dataloaders/Pre_processing.ipynb` by passing the arguments of data location.

### Prepare Your Own Data

Our DCPA could be extended to other datasets with some modifications.  

### You can also download our prepared dataset from https://pan.baidu.com/s/1s58oHoIFoE1lijLuNVm2kA?pwd=u4fj.

### Pretrained Weight
You can download our weights from https://pan.baidu.com/s/1Jp3C-w_w4BwgWD4tG4Ez0Q?pwd=frub.

## Usage
1. Clone the repo.;
```
git clone https://github.com/BinYCn/RPMC.git
cd RPMC
```
2. Put the data in `./data`;

3. Train the model;
```
cd code
bash train.sh
```
4. Test the model;
```
cd code
bash test.sh
```

## Acknowledgements:
Our code is adapted from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [MC-Net](https://github.com/ycwu1997/MC-Net.git). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

## Questions
If any questions, feel free to contact me at 'binycn904363330@gmail.com'
