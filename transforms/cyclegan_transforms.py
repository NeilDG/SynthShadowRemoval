"""
Customized transforms using kornia for faster data augmentation

@author: delgallegon
"""

import torch
import torch.nn as nn
import kornia
import numpy as np

class CycleGANTransform(nn.Module):

    def __init__(self, opts):
        super(CycleGANTransform, self).__init__()

        self.patch_size = (opts.patch_size, opts.patch_size)

        self.transform_op = kornia.augmentation.ImageSequential(
            kornia.augmentation.RandomVerticalFlip(p=0.5),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            same_on_batch=True)


    def forward(self, x):
        stride = np.random.randint(4, self.patch_size)
        patch_extract_op = kornia.contrib.ExtractTensorPatches(window_size=self.patch_size, stride=stride)

        out_tensor = patch_extract_op(x)
        out_tensor = torch.flatten(out_tensor, 0, 1)

        # out_tensor_size = len(out_tensor)

        indices = torch.randperm(512)
        out_tensor = out_tensor[indices]
        out_tensor = self.transform_op(out_tensor)

        # print("Shape: ", np.shape(out_tensor))
        return out_tensor
