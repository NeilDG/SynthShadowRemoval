"""
Customized transforms using kornia for faster data augmentation

@author: delgallegon
"""

import torch
import torch.nn as nn
import kornia
import numpy as np
import torchvision.transforms as transforms

from config import iid_server_config

class IIDTransform(nn.Module):
    # GAMMA = 0.95
    # BETA = 0.55
    # GAMMA = 1.25
    # BETA = 0.95
    MIN_GAMMA = 1.35
    MIN_BETA = 0.45
    MAX_GAMMA = 1.75
    MAX_BETA = 1.55

    def __init__(self):
        super(IIDTransform, self).__init__()

        self.transform_op = transforms.Normalize((0.5,), (0.5,))

        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        network_config = sc_instance.interpret_network_config_from_version()

        # self.MIN_GAMMA = network_config["min_gamma"]
        # self.MIN_BETA = network_config["min_beta"]
        # 
        # self.MAX_GAMMA = network_config["max_gamma"]
        # self.MAX_BETA = network_config["max_beta"]


    def mask_fill_nonzeros(self, input_tensor):
        output_tensor = torch.clone(input_tensor)
        masked_tensor = (input_tensor <= 0.01)
        return output_tensor.masked_fill(masked_tensor, 1.0)

    def revert_mask_fill_nonzeros(self, input_tensor):
        output_tensor = torch.clone(input_tensor)
        masked_tensor = (input_tensor >= 1.0)
        return output_tensor.masked_fill_(masked_tensor, 0.0)

    def create_sky_reflection_masks(self, albedo_tensor, tozeroone = True):
        #assume sky areas/reflections are 1 in albedo tensor
        if (tozeroone):
            albedo_tensor = (albedo_tensor * 0.5) + 0.5

        albedo_gray = kornia.color.rgb_to_grayscale(albedo_tensor)
        # albedo_gray = kornia.filters.median_blur(albedo_gray, (3, 3))

        output_tensor = torch.ones_like(albedo_gray)
        masked_tensor = (albedo_gray >= 1.0)
        return output_tensor.masked_fill_(masked_tensor, 0)

    def decompose_shadow(self, rgb_ws, rgb_ns):
        gamma = torch.tensor(np.random.uniform(self.MIN_GAMMA, self.MAX_GAMMA), dtype=torch.float)
        beta = torch.tensor(np.random.uniform(self.MIN_BETA, self.MAX_BETA), dtype=torch.float)

        shadow_matte = 1.0 - self.extract_shadow(rgb_ws, rgb_ns, True)
        rgb_ws = self.add_shadow(rgb_ns, shadow_matte, gamma, beta)
        rgb_ws_relit = self.extract_relit(rgb_ws, gamma, beta)
        rgb_ns = self.remove_shadow(rgb_ws, rgb_ws_relit, shadow_matte)

        rgb_ws = self.transform_op(rgb_ws)
        rgb_ns = self.transform_op(rgb_ns)
        shadow_matte = self.transform_op(shadow_matte)
        rgb_ws_relit = self.transform_op(rgb_ws_relit)

        return rgb_ws, rgb_ns, shadow_matte, rgb_ws_relit, gamma, beta

    # def forward(self, rgb_ws, rgb_ns, albedo_tensor):
    #     #extract shadows
    #     rgb_ws, rgb_ns, shadow_matte, _, _, _ = self.decompose_shadow(rgb_ws, rgb_ns)
    #
    #     albedo_refined, shading_refined = self.decompose(rgb_ns, albedo_tensor, True)
    #     # rgb_recon = self.produce_rgb(albedo_refined, shading_refined, shadow_matte, False)
    #
    #     # loss_op = nn.L1Loss()
    #     # print("Difference between RGB vs Recon: ", loss_op(rgb_recon, rgb_ws).item())
    #
    #     albedo_refined = self.transform_op(albedo_refined)
    #     shading_refined = self.transform_op(shading_refined)
    #
    #     return rgb_ws, rgb_ns, albedo_refined, shading_refined, shadow_matte #return original RGB

    def produce_rgb(self, albedo_tensor, shading_tensor, shadow_tensor, tozeroone = True):
        if(tozeroone):
            albedo_tensor = (albedo_tensor * 0.5) + 0.5
            shading_tensor = (shading_tensor * 0.5) + 0.5
            shadow_tensor = (shadow_tensor * 0.5) + 0.5

        rgb_recon = (albedo_tensor * shading_tensor) - shadow_tensor
        rgb_recon = torch.clip(rgb_recon, 0.0, 1.0)
        return rgb_recon

    # def remove_rgb_shadow(self, rgb_tensor, shadow_tensor, tozeroone = True):
    #     if (tozeroone):
    #         rgb_tensor = (rgb_tensor * 0.5) + 0.5
    #         shadow_tensor = (shadow_tensor * 0.5) + 0.5
    #
    #     rgb_recon = rgb_tensor + shadow_tensor
    #     return rgb_recon
    #
    # def add_rgb_shadow(self, rgb_tensor, shadow_tensor, tozeroone=True):
    #     if (tozeroone):
    #         rgb_tensor = (rgb_tensor * 0.5) + 0.5
    #         shadow_tensor = (shadow_tensor * 0.5) + 0.5
    #
    #     rgb_recon = rgb_tensor - shadow_tensor
    #     return rgb_recon

    def decompose(self, rgb_tensor, albedo_tensor, one_channel = False):
        min = torch.min(rgb_tensor)
        max = torch.max(rgb_tensor)

        final_shading = self.extract_shading(rgb_tensor, albedo_tensor, one_channel)
        # final_shading = self.mask_fill_nonzeros(final_shading)

        final_albedo = rgb_tensor / self.mask_fill_nonzeros(final_shading)

        final_albedo = torch.clip(final_albedo, min, max)
        final_shading = torch.clip(final_shading, min, max)

        return final_albedo, final_shading

    def extract_shading(self, rgb_tensor, albedo_tensor, one_channel = False, mask_fill = True):
        min = torch.min(rgb_tensor)
        max = torch.max(rgb_tensor)

        if(mask_fill):
            albedo_refined = self.mask_fill_nonzeros(albedo_tensor)
        else:
            albedo_refined = albedo_tensor

        shading_tensor = rgb_tensor / albedo_refined

        if(one_channel == True):
            shading_tensor = kornia.color.rgb_to_grayscale(shading_tensor)

        shading_tensor = torch.clip(shading_tensor, min, max)
        return shading_tensor

    def extract_relit(self, rgb_ws, gamma, beta):
        min = torch.min(rgb_ws)
        max = torch.tensor(self.MAX_GAMMA)

        relit_ws = (gamma * rgb_ws) + beta
        relit_ws = torch.clip(relit_ws, min, max)
        return relit_ws

    def add_shadow(self, rgb_ns, shadow_matte, gamma, beta):
        # min = torch.min(rgb_ws)
        # max = torch.max(rgb_ws)
        min = 0.0
        max = 1.0
        upper_bound = max - min

        darkened = (rgb_ns - beta) / gamma
        rgb_ws = (rgb_ns * shadow_matte) + (darkened * (upper_bound - shadow_matte))

        rgb_ws = torch.clip(rgb_ws, min, max)
        return rgb_ws

    def remove_shadow(self, rgb_ws, rgb_ws_relit, shadow_matte):
        # min = torch.min(rgb_ws)
        # max = torch.max(rgb_ws)
        min = 0.0
        max = 1.0
        upper_bound = max - min

        shadow_free = (rgb_ws * shadow_matte) + (rgb_ws_relit * (upper_bound - shadow_matte))
        return torch.clip(shadow_free, min, max)

    def extract_shadow(self, rgb_tensor_ws, rgb_tensor_ns, one_channel = False):
        min = torch.min(rgb_tensor_ws)
        max = torch.max(rgb_tensor_ws)

        ws_refined = rgb_tensor_ws
        ns_refined = rgb_tensor_ns

        shadow_tensor = ns_refined - ws_refined

        if(one_channel == True):
            shadow_tensor = kornia.color.rgb_to_grayscale(shadow_tensor)

        shadow_tensor = torch.clip(shadow_tensor, min, max)
        return shadow_tensor

    def extract_shadow_matte(self, rgb_ws, rgb_ns, rgb_ws_relit):
        min = torch.min(rgb_ws)
        max = torch.max(rgb_ws)

        a = rgb_ns - rgb_ws_relit
        b = rgb_ws - rgb_ws_relit

        matte = a / b
        matte = torch.clip(matte, min, max)
        matte = kornia.color.rgb_to_grayscale(matte)
        return matte

    def extract_albedo(self, rgb_tensor, shading_tensor, shadow_tensor, tozeroone = True):
        min = torch.min(rgb_tensor)
        max = torch.max(rgb_tensor)
        if(tozeroone):
            rgb_tensor = (rgb_tensor * 0.5) + 0.5
            shading_tensor = (shading_tensor * 0.5) + 0.5
            shadow_tensor = (shadow_tensor * 0.5) + 0.5

        shading_tensor = self.mask_fill_nonzeros(shading_tensor)
        # shadow_tensor = self.mask_fill_nonzeros(shadow_tensor)

        albedo_refined = (rgb_tensor + shadow_tensor) / shading_tensor
        # albedo_refined = rgb_tensor / shading_tensor
        albedo_tensor = torch.clip(albedo_refined, min, max)
        # albedo_tensor = albedo_refined

        return albedo_tensor

    # used for viewing an albedo tensor and for metric measurement
    def view_albedo(self, albedo_tensor, tozeroone=True):
        if (tozeroone):
            albedo_tensor = (albedo_tensor * 0.5) + 0.5
        return self.revert_mask_fill_nonzeros(albedo_tensor)


class CGITransform(IIDTransform):
    
    def __init__(self):
        super(CGITransform, self).__init__()

    def decompose_cgi(self, rgb_tensor, albedo_tensor):
        albedo_refined, shading_refined = self.decompose(rgb_tensor, albedo_tensor, True)

        rgb_recon = albedo_refined * shading_refined
        rgb_recon = torch.clip(rgb_recon, 0.0, 1.0)

        # loss_op = nn.L1Loss()
        # print("Difference between RGB vs Recon: ", loss_op(rgb_recon, rgb_tensor).item())


        return rgb_recon, albedo_refined, shading_refined

    


