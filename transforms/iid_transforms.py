"""
Customized transforms using kornia for faster data augmentation

@author: delgallegon
"""

import torch
import torch.nn as nn
import kornia
import numpy as np
import torchvision.transforms as transforms

class IIDTransform(nn.Module):

    def __init__(self):
        super(IIDTransform, self).__init__()

        self.transform_op = transforms.Normalize((0.5,), (0.5,))

    def mask_fill_nonzeros(self, input_tensor):
        output_tensor = torch.clone(input_tensor)
        masked_tensor = (input_tensor <= 0.0)
        return output_tensor.masked_fill(masked_tensor, 1.0)

    def revert_mask_fill_nonzeros(self, input_tensor):
        output_tensor = torch.clone(input_tensor)
        masked_tensor = (input_tensor >= 1.0)
        return output_tensor.masked_fill_(masked_tensor, 0.0)

    def define_albedo_mask(self, albedo_tensor):
        masked_tensor = (albedo_tensor < 1.0)
        return masked_tensor

    def forward(self, rgb_ws, rgb_ns, albedo_tensor):
        min = 0.0
        max = 1.0

        shading_tensor = self.extract_shading(rgb_ns, albedo_tensor, False)

        #refine albedo
        shading_refined = self.mask_fill_nonzeros(shading_tensor)
        albedo_refined = rgb_ns / shading_refined
        albedo_refined = torch.clip(albedo_refined, min, max)

        #extract shadows
        shadows_refined = self.extract_shadow(rgb_ws, rgb_ns)

        # albedo_refined = albedo_tensor
        # shading_refined = shading_tensor

        rgb_recon = self.produce_rgb(albedo_refined, shading_refined, shadows_refined, False)

        # loss_op = nn.L1Loss()
        # print("Difference between RGB vs Recon: ", loss_op(rgb_recon, rgb_ws).item()) #0.011102916672825813

        rgb_recon = self.transform_op(rgb_recon)
        albedo_refined = self.transform_op(albedo_refined)
        shading_tensor = self.transform_op(shading_tensor)
        shadow_tensor = self.transform_op(shadows_refined)

        return rgb_recon, albedo_refined, shading_tensor, shadow_tensor

    def extract_shading(self, rgb_tensor, albedo_tensor, one_channel = False):
        min = 0.0
        max = 1.0

        albedo_refined = self.mask_fill_nonzeros(albedo_tensor)
        shading_tensor = rgb_tensor / albedo_refined

        if(one_channel == True):
            shading_tensor = kornia.color.rgb_to_grayscale(shading_tensor)

        shading_tensor = torch.clip(shading_tensor, min, max)
        return shading_tensor

    def extract_shadow(self, rgb_tensor_ws, rgb_tensor_ns, one_channel = False):
        min = 0.0
        max = 1.0

        ws_refined = self.mask_fill_nonzeros(rgb_tensor_ws)
        ns_refined = self.mask_fill_nonzeros(rgb_tensor_ns)

        shadow_tensor = ws_refined / ns_refined

        if(one_channel == True):
            shadow_tensor = kornia.color.rgb_to_grayscale(shadow_tensor)

        shadow_tensor = torch.clip(shadow_tensor, min, max)
        return shadow_tensor

    def remove_rgb_shadow(self, rgb_tensor_ws, shadow_map):
        min = 0.0
        max = 1.0

        ws_refined = self.mask_fill_nonzeros(rgb_tensor_ws)
        shadow_map = self.mask_fill_nonzeros(shadow_map)

        rgb_tensor_ns = ws_refined / (shadow_map * 1.0)

        shadow_tensor = torch.clip(rgb_tensor_ns, min, max)
        return shadow_tensor

    #used for viewing an albedo tensor and for metric measurement
    def view_albedo(self, albedo_tensor, tozeroone = True):
        if (tozeroone):
            albedo_tensor = (albedo_tensor * 0.5) + 0.5
        return self.revert_mask_fill_nonzeros(albedo_tensor)

    def extract_albedo(self, rgb_tensor, shading_tensor, shadow_tensor, tozeroone = True):
        min = 0.0
        max = 1.0
        if(tozeroone):
            rgb_tensor = (rgb_tensor * 0.5) + 0.5
            shading_tensor = (shading_tensor * 0.5) + 0.5
            shadow_tensor = (shadow_tensor * 0.5) + 0.5

        # shading_tensor = self.mask_fill_nonzeros(shading_tensor)
        # shadow_tensor = self.mask_fill_nonzeros(shadow_tensor)

        albedo_refined = rgb_tensor / (shading_tensor * shadow_tensor)
        # albedo_tensor = torch.clip(albedo_refined, min, max)
        albedo_tensor = albedo_refined

        return albedo_tensor

    def produce_rgb(self, albedo_tensor, shading_tensor, shadow_tensor, tozeroone = True):
        if(tozeroone):
            albedo_tensor = (albedo_tensor * 0.5) + 0.5
            shading_tensor = (shading_tensor * 0.5) + 0.5
            shadow_tensor = (shadow_tensor * 0.5) + 0.5

        # albedo_tensor = self.mask_fill_nonzeros(albedo_tensor)
        # shading_tensor = self.mask_fill_nonzeros(shading_tensor)
        rgb_recon = albedo_tensor * shading_tensor * shadow_tensor
        rgb_recon = torch.clip(rgb_recon, 0.0, 1.0)
        return rgb_recon

