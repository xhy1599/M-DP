# Multimodal Fusion-Guided Diffusion Policy for Motion Planning in Unstructured Environments

## Installation Guide

### Prerequisites
- PyTorch version: 1.12.0+cu113

### Install CUDA-based KNN and Random Search Modules
These modules are from [RegFormer](https://github.com/IRMVLab/RegFormer):
```bash
cd src
cd fused_conv_random_k
python setup.py install
cd ../
cd fused_conv_select_k
python setup.py install
cd ../
```
### Install Diffusion Policy Package
```BASH
cd src/mdptest
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
pip install diffusers==0.11.1 warmup-scheduler efficientnet-pytorch vit-pytorch
cd ../../
```
### Running Instructions
Clone this repository to your catkin workspace:
```BASH
cd catkin_ws/src
git clone https://github.com/xhy1599/M-DP.git
#Build and run the launch file
roslaunch ros_pointnet mdp.launch
```
### Acknowledgments
This project is built upon the following open-source works:

[RegFormer](https://github.com/IRMVLab/RegFormer)
[NoMaD](https://github.com/robodhruv/visualnav-transformer.git)
[diffusion_policy](https://github.com/real-stanford/diffusion_policy.git)
