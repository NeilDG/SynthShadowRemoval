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

class SDNetDecomposition():
    @staticmethod
    def add_shadow(rgb_ns, shadow_matte, gamma, beta):
        min = 0.0
        max = 1.0
        upper_bound = max - min

        darkened = (rgb_ns - beta) / gamma
        rgb_ws = (rgb_ns * shadow_matte) + (darkened * (upper_bound - shadow_matte))

        rgb_ws = torch.clip(rgb_ws, min, max)
        return rgb_ws

    @staticmethod
    def remove_shadow(rgb_ws, rgb_ws_relit, shadow_matte):
        min = 0.0
        max = 1.0
        upper_bound = max - min

        shadow_free = (rgb_ws * shadow_matte) + (rgb_ws_relit * (upper_bound - shadow_matte))
        return torch.clip(shadow_free, min, max)

    @staticmethod
    def extract_shadow(rgb_tensor_ws, rgb_tensor_ns):
        min = torch.min(rgb_tensor_ws)
        max = torch.max(rgb_tensor_ws)

        ws_refined = rgb_tensor_ws
        ns_refined = rgb_tensor_ns

        shadow_tensor = ns_refined - ws_refined
        shadow_tensor = kornia.color.rgb_to_grayscale(shadow_tensor)

        shadow_tensor = torch.clip(shadow_tensor, min, max)
        return shadow_tensor

    @staticmethod
    def extract_relit(rgb_ws, gamma, beta):
        min = torch.min(rgb_ws)
        max = ShadowMapTransforms.MAX_GAMMA

        relit_ws = (gamma * rgb_ws) + beta
        relit_ws = torch.clip(relit_ws, min, max)
        return relit_ws

    @staticmethod
    def extract_relit_batch(rgb_ws, gamma, beta):
        relit_ws = torch.zeros_like(rgb_ws)
        for i in range(np.shape(rgb_ws)[0]):
            relit_ws[i] = SDNetDecomposition.extract_relit(rgb_ws[i], gamma[i], beta[i])

        return relit_ws

class ShadowMapTransforms():
    MIN_BETA = 0.05
    MAX_BETA = 1.2
    MAX_GAMMA = 1.5

    def __init__(self):
        super(ShadowMapTransforms, self).__init__()

        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        network_config = sc_instance.interpret_network_config_from_version()


    def decompose_shadow(self, rgb_ws, rgb_ns):
        beta = torch.tensor(np.random.uniform(self.MIN_BETA, self.MAX_BETA), dtype=torch.float)
        gamma = self.MAX_GAMMA - beta

        shadow_matte = 1.0 - SDNetDecomposition.extract_shadow(rgb_ws, rgb_ns)
        rgb_ws_refined = SDNetDecomposition.add_shadow(rgb_ns, shadow_matte, gamma, beta)

        rgb_ws_relit = SDNetDecomposition.extract_relit(rgb_ws_refined, gamma, beta)
        rgb_ns_refined = SDNetDecomposition.remove_shadow(rgb_ws_refined, rgb_ws_relit, shadow_matte)

        return rgb_ws_refined, rgb_ns_refined, shadow_matte, gamma, beta


    def generate_shadow_map(self, rgb_tensor_ws, rgb_tensor_ns, one_channel = True):
        shadow_tensor = rgb_tensor_ns - rgb_tensor_ws

        shadow_gray = kornia.color.rgb_to_grayscale(shadow_tensor)
        # shadow_mask = (shadow_gray != 0.0).float()
        shadow_mask = shadow_gray

        if(one_channel == True):
            shadow_tensor = kornia.color.rgb_to_grayscale(shadow_tensor)

        return rgb_tensor_ws, rgb_tensor_ns, shadow_tensor, shadow_mask

    def remove_rgb_shadow(self, rgb_tensor_ws, shadow_tensor, tozeroone=True):
        if (tozeroone):
            rgb_tensor_ws = tensor_utils.normalize_to_01(rgb_tensor_ws)
            shadow_tensor = tensor_utils.normalize_to_01(shadow_tensor)

        rgb_recon = rgb_tensor_ws + shadow_tensor
        rgb_recon = torch.clip(rgb_recon, 0.0, 1.0)
        return rgb_recon