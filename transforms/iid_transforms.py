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

    def forward(self, rgb_tensor, albedo_tensor):
        min = 0.0
        max = 1.0

        shading_tensor = self.extract_shading(rgb_tensor, albedo_tensor, False)

        #refine albedo
        shading_refined = self.mask_fill_nonzeros(shading_tensor)
        # shading_refined = shading_tensor
        albedo_refined = rgb_tensor / shading_refined
        albedo_refined = torch.clip(albedo_refined, min, max)
        # albedo_refined = self.revert_mask_fill_nonzeros(albedo_refined)

        # albedo_refined = albedo_tensor
        # shading_refined = shading_tensor

        rgb_recon = self.produce_rgb(albedo_refined, shading_refined, False)

        # loss_op = nn.L1Loss()
        # print("Difference between RGB vs Recon: ", loss_op(rgb_recon, rgb_tensor).item()) #0.011102916672825813

        rgb_recon = self.transform_op(rgb_recon)
        albedo_refined = self.transform_op(albedo_refined)
        shading_tensor = self.transform_op(shading_tensor)

        return rgb_recon, albedo_refined, shading_tensor

    def extract_shading(self, rgb_tensor, albedo_tensor, one_channel = False):
        min = 0.0
        max = 1.0

        albedo_refined = self.mask_fill_nonzeros(albedo_tensor)
        shading_tensor = rgb_tensor / albedo_refined

        if(one_channel == True):
            shading_tensor = kornia.color.rgb_to_grayscale(shading_tensor)

        shading_tensor = torch.clip(shading_tensor, min, max)
        return shading_tensor

    #used for viewing an albedo tensor and for metric measurement
    def view_albedo(self, albedo_tensor, tozeroone = True):
        if (tozeroone):
            albedo_tensor = (albedo_tensor * 0.5) + 0.5
        return self.revert_mask_fill_nonzeros(albedo_tensor)

    # def extract_albedo(self, rgb_tensor, shading_tensor, tozeroone = True):
    #     min = 0.0
    #     max = 1.0
    #     if(tozeroone):
    #         rgb_tensor = (rgb_tensor * 0.5) + 0.5
    #         shading_tensor = (shading_tensor * 0.5) + 0.5
    #
    #     albedo_refined = rgb_tensor / shading_tensor
    #     # albedo_tensor = torch.clip(albedo_refined, min, max)
    #     albedo_tensor = albedo_refined
    #
    #     return albedo_tensor

    def produce_rgb(self, albedo_tensor, shading_tensor, tozeroone = True):
        if(tozeroone):
            albedo_tensor = (albedo_tensor * 0.5) + 0.5
            shading_tensor = (shading_tensor * 0.5) + 0.5

        # albedo_tensor = self.mask_fill_nonzeros(albedo_tensor)
        # shading_tensor = self.mask_fill_nonzeros(shading_tensor)
        rgb_recon = albedo_tensor * shading_tensor
        rgb_recon = torch.clip(rgb_recon, 0.0, 1.0)
        return rgb_recon

