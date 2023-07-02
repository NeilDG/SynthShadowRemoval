import os.path
import random

import torch
import cv2
import numpy as np
from torch.utils import data
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional
import torch.nn.functional as F
import global_config
import kornia
from pathlib import Path

from config.network_config import ConfigHolder

def normalize(light_angle):
    std = light_angle / 360.0
    min = -1.0
    max = 1.0
    scaled = std * (max - min) + min

    return scaled

class PairedImageDataset(data.Dataset):
    def __init__(self, a_list, b_list, transform_config):
        self.a_list = a_list
        self.b_list = b_list
        self.transform_config = transform_config

        config_holder = ConfigHolder.getInstance()
        self.augment_mode = config_holder.get_network_attribute("augment_key", "none")
        self.use_tanh = config_holder.get_network_attribute("use_tanh", False)

        if (self.transform_config == 1):
            patch_size = config_holder.get_network_attribute("patch_size", 32)
        else:
            patch_size = 256

        self.patch_size = (patch_size, patch_size)
        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor()
        ])

        self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

    def __getitem__(self, idx):
        file_name = self.b_list[idx % len(self.b_list)].split("\\")[-1].split(".")[0]

        a_img = cv2.imread(self.a_list[idx])
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        a_img = self.initial_op(a_img)

        b_img = cv2.imread(self.b_list[(idx % len(self.b_list))])
        b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
        b_img = self.initial_op(b_img)

        if(self.use_tanh):
            a_img = self.norm_op(a_img)
            b_img = self.norm_op(b_img)

        return file_name, a_img, b_img

    def __len__(self):
        return len(self.a_list)

class SingleImageDataset(data.Dataset):
    def __init__(self, a_list, transform_config):
        self.a_list = a_list
        self.transform_config = transform_config

        config_holder = ConfigHolder.getInstance()
        self.augment_mode = config_holder.get_network_attribute("augment_key", "none")
        self.use_tanh = config_holder.get_network_attribute("use_tanh", True)

        if (self.transform_config == 1):
            patch_size = config_holder.get_network_attribute("patch_size", 32)
        else:
            patch_size = 256

        self.patch_size = (patch_size, patch_size)
        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor()
        ])

        self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

    def __getitem__(self, idx):
        file_name = self.a_list[idx].split("/")[-1].split(".")[0]

        a_img = cv2.imread(self.a_list[idx])
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        a_img = self.initial_op(a_img)

        if(self.use_tanh):
            a_img = self.norm_op(a_img)

        return file_name, a_img

    def __len__(self):
        return len(self.a_list)