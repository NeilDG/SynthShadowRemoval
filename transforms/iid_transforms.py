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
        masked_tensor = (input_tensor == 0.0)
        return input_tensor.masked_fill_(masked_tensor, 1.0)

    def revert_mask_fill_nonzeros(self, input_tensor):
        masked_tensor = (input_tensor == 1.0)
        return input_tensor.masked_fill_(masked_tensor, 0.0)

    def forward(self, rgb_tensor, albedo_tensor):
        min = 0.0
        max = 1.0

        albedo_refined = self.mask_fill_nonzeros(albedo_tensor)
        shading_tensor = rgb_tensor / albedo_refined
        shading_tensor = kornia.color.rgb_to_grayscale(shading_tensor)
        shading_tensor = torch.clip(shading_tensor, min, max)

        #refine albedo
        shading_refined = self.mask_fill_nonzeros(shading_tensor)
        albedo_tensor = rgb_tensor / shading_refined
        albedo_tensor = torch.clip(albedo_tensor, min, max)

        rgb_recon = albedo_tensor * shading_tensor

        # loss_op = nn.L1Loss()
        # print("Difference between RGB vs Recon: ", loss_op(rgb_recon, rgb_tensor).item()) #0.0017016564961522818

        rgb_tensor = self.transform_op(rgb_tensor)
        albedo_tensor = self.transform_op(albedo_tensor)
        shading_tensor = self.transform_op(shading_tensor)

        return rgb_recon, albedo_tensor, shading_tensor

    # def produce_albedo(self, rgb_tensor, shading_tensor, tozeroone = True):
    #     if(tozeroone):
    #         rgb_tensor = (rgb_tensor * 0.5) + 0.5
    #         shading_tensor = (shading_tensor * 0.5) + 0.5
    #
    #     shading_tensor = torch.clip(shading_tensor, 0.0, 1.0)
    #     shading_tensor = self.mask_fill_nonzeros(shading_tensor)
    #     albedo_tensor = rgb_tensor / shading_tensor
    #     albedo_tensor = torch.clip(albedo_tensor, 0.0, 1.0)
    #
    #     return albedo_tensor

    def produce_rgb(self, albedo_tensor, shading_tensor, tozeroone = True):
        if(tozeroone):
            albedo_tensor = (albedo_tensor * 0.5) + 0.5
            shading_tensor = (shading_tensor * 0.5) + 0.5

        rgb_recon = albedo_tensor * shading_tensor
        rgb_recon = torch.clip(rgb_recon, 0.0, 1.0)
        return rgb_recon

