"""
Customized transforms using kornia for faster data augmentation

@author: delgallegon
"""

import torch
import torch.nn as nn
import kornia
import numpy as np

class CycleGANTransform(nn.Module):

    def __init__(self, patch_size):
        super(CycleGANTransform, self).__init__()

        self.patch_size = (patch_size, patch_size)

        self.transform_op = kornia.augmentation.ImageSequential(
            kornia.augmentation.RandomCrop(self.patch_size),
            kornia.augmentation.RandomVerticalFlip(p=0.5),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            same_on_batch=True)

        # self.stride_choices = [4, 8, 12, 16, 24, 32]


    def extract_patch_tensors(self, x, num_patches= -1):
        patch_extract_op = kornia.contrib.ExtractTensorPatches(window_size=self.patch_size, stride=self.patch_size)
        out_tensor = patch_extract_op(x)
        out_tensor = torch.flatten(out_tensor, 0, 1)

        if(num_patches == -1):
            out_tensor_size = len(out_tensor)
            indices = torch.randperm(out_tensor_size)
        else:
            indices = torch.randperm(num_patches)

        return out_tensor[indices]

    def forward(self, x):
        # stride = np.random.choice(self.stride_choices)

        out_tensor = self.transform_op(x)
        out_tensor = self.extract_patch_tensors(x)
        # print("Shape: ", np.shape(out_tensor))
        return out_tensor
