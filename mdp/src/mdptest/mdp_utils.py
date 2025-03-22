
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import yaml

import cv2
from typing import List, Tuple, Dict, Optional, Callable

from PIL import Image as PILImage
import argparse
import yaml
import time
# models

from vit import ViT
from Resnet50 import ResNet50Encoder
from base_model import BaseModel
from nomad import NoMaD











# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def merge(x_vec, y_vec):
	merged_list = [(int(x_vec[i]), int(y_vec[i])) for i in range(0, len(x_vec))]
	return merged_list
def coverLabelToGroup6(seg):
    seg=np.max(seg,axis=2)
    palette = [[0, 0, 0], [0, 128, 0], [255, 255, 0], [255, 128, 0], [255, 0, 0], [0, 0, 128]]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    return color_seg


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()




def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)
    

def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cuda"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    model_type = config["model_type"]
    

    if config["model_type"] == "nomad":
        if config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
               # patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "Resnet50": 
            vision_encoder = ResNet50Encoder(
                encoding_size=config["encoding_size"],
                context_size=config["context_size"],
            )
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["dp_encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        
        #pc_net =  PointNetSaModule(batch_size = config["batch_size"], K_sample = config["K_sample"], kernel_size = config["kernel_size"], H = config["out_H"], W = config["out_W"], 
         #                              stride_H = config["stride_H"], stride_W = config["stride_W"], distance = config["pc_distance"], in_channels = config["in_channels"],
        #                               mlp = config["mlp"], bn_decay = config["bn_decay"])


        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,

        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model





VISUALIZATION_IMAGE_SIZE = (640, 360) 
IMAGE_ASPECT_RATIO = (
    16 / 9
)  # all images are centered cropped to a 16:9 aspect ratio in training






