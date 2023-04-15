import torchvision.transforms as transforms
import torchvision.transforms.functional
from torch import nn


class Img2ImgBasicTransform(nn.Module):
    def __init__(self, patch_size):
        super(Img2ImgBasicTransform, self).__init__()

        self.patch_size = (patch_size, patch_size)

        self.initial_op = transforms.Compose([
            transforms.RandomCrop(self.patch_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ])

    def forward(self, x):
        return self.initial_op(x)