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

        self.MIN_SHADOW_INTENSITY = 0.5
        self.MAX_SHADOW_INTENSITY = 1.0
        self.shadow_intensity = 1.0


    def generate_shadow_map(self, rgb_tensor_ws, rgb_tensor_ns, one_channel = True):
        # min = torch.min(rgb_tensor_ws)
        # max = torch.max(rgb_tensor_ws)

        shadow_tensor = rgb_tensor_ns - rgb_tensor_ws

        self.shadow_intensity = np.random.uniform(self.MIN_SHADOW_INTENSITY, self.MAX_SHADOW_INTENSITY)
        shadow_tensor = shadow_tensor * self.shadow_intensity
        shadow_tensor = torch.clip(shadow_tensor, 0.0, 1.0)

        ws_refined = rgb_tensor_ns - shadow_tensor
        ws_refined = torch.clip(ws_refined, 0.0, 1.0)
        ns_refined = ws_refined + shadow_tensor
        ns_refined = torch.clip(ns_refined, 0.0, 1.0)

        if(one_channel == True):
            shadow_tensor = kornia.color.rgb_to_grayscale(shadow_tensor)

        return ws_refined, ns_refined, shadow_tensor

    def remove_rgb_shadow(self, rgb_tensor_ws, shadow_tensor, tozeroone=True):
        if (tozeroone):
            rgb_tensor_ws = tensor_utils.normalize_to_01(rgb_tensor_ws)
            shadow_tensor = tensor_utils.normalize_to_01(shadow_tensor)

        rgb_recon = rgb_tensor_ws + shadow_tensor
        rgb_recon = torch.clip(rgb_recon, 0.0, 1.0)
        return rgb_recon