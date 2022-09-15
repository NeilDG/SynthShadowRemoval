"""
Customized transforms using kornia and torch for faster data augmentation

@author: delgallegon
"""

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
        self.transform_op = transforms.Normalize((0.5,), (0.5,))

    def extract_shadow(self, rgb_tensor_ws, rgb_tensor_ns, one_channel = True):
        min = torch.min(rgb_tensor_ws)
        max = torch.max(rgb_tensor_ws)

        ws_refined = rgb_tensor_ws
        ns_refined = rgb_tensor_ns

        shadow_tensor = ns_refined - ws_refined

        if(one_channel == True):
            shadow_tensor = kornia.color.rgb_to_grayscale(shadow_tensor)

        shadow_tensor = torch.clip(shadow_tensor, min, max)
        return shadow_tensor

    def remove_rgb_shadow(self, rgb_tensor, shadow_tensor, tozeroone=True):
        if (tozeroone):
            rgb_tensor = tensor_utils.normalize_to_01(rgb_tensor)
            shadow_tensor = tensor_utils.normalize_to_01(shadow_tensor)

        rgb_recon = rgb_tensor + shadow_tensor
        return rgb_recon

    def add_rgb_shadow(self, rgb_tensor, shadow_tensor, tozeroone=True):
        if (tozeroone):
            rgb_tensor = tensor_utils.normalize_to_01(rgb_tensor)
            shadow_tensor = tensor_utils.normalize_to_01(shadow_tensor)

        rgb_recon = rgb_tensor - shadow_tensor
        return rgb_recon