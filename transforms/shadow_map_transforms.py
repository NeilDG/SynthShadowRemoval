"""
Customized transforms using kornia and torch for faster data augmentation

@author: delgallegon
"""
import random

import torch
import torch.nn as nn
import kornia
import numpy as np
import torchvision.transforms as transforms

from config import iid_server_config
from utils import tensor_utils


class ShadowMapTransforms():
    def __init__(self):
        super(ShadowMapTransforms, self).__init__()


    def generate_shadow_map(self, rgb_tensor_ws, rgb_tensor_ns, one_channel = True):
        shadow_tensor = rgb_tensor_ns - rgb_tensor_ws

        shadow_matte = kornia.color.rgb_to_grayscale(shadow_tensor)

        if(one_channel == True):
            shadow_tensor = kornia.color.rgb_to_grayscale(shadow_tensor)

        return rgb_tensor_ws, rgb_tensor_ns, shadow_tensor, shadow_matte

    def remove_rgb_shadow(self, rgb_tensor_ws, shadow_tensor, tozeroone=True):
        if (tozeroone):
            rgb_tensor_ws = tensor_utils.normalize_to_01(rgb_tensor_ws)
            shadow_tensor = tensor_utils.normalize_to_01(shadow_tensor)

        rgb_recon = rgb_tensor_ws + shadow_tensor
        rgb_recon = torch.clip(rgb_recon, 0.0, 1.0)
        return rgb_recon