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


class ColorTransferDataset(data.Dataset):
    def __init__(self, image_list_a, path_b, path_segment, transform_config):
        self.image_list_a = image_list_a
        self.path_b = path_b
        self.path_segment = path_segment
        self.transform_config = transform_config

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        if(transform_config == 1):
            self.final_transform_op = transforms.Compose([
                # transforms.RandomCrop(constants.PATCH_IMAGE_SIZE),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.mask_op = transforms.Compose([
                transforms.ToTensor()
            ])

        else:
            self.final_transform_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.mask_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.path_b + "/" + file_name
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        img_id =  self.path_segment + "/" + file_name
        #print("Img Id: ", img_id)
        img_segment = cv2.cvtColor(cv2.imread(img_id), cv2.COLOR_BGR2RGB)
        img_segment = cv2.resize(img_segment, (256, 256))

        # convert img_b to mask
        img_mask_c = cv2.inRange(img_segment[:, :, 1], 200, 255)  # green segmentation mask = road
        img_mask_c = cv2.cvtColor(img_mask_c, cv2.COLOR_GRAY2RGB)

        img_mask_d = cv2.inRange(img_segment[:, :, 0], 200, 255)  # red segmentation mask = building
        img_mask_d = cv2.cvtColor(img_mask_d, cv2.COLOR_GRAY2RGB)

        img_a = self.initial_op(img_a)
        img_b = self.initial_op(img_b)
        img_mask_c = self.initial_op(img_mask_c)
        img_mask_d = self.initial_op(img_mask_d)

        if(self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(img_a, output_size=constants.PATCH_IMAGE_SIZE)
            i, j, h, w = crop_indices

            img_a = transforms.functional.crop(img_a, i, j, h, w)
            img_b = transforms.functional.crop(img_b, i, j, h, w)
            img_mask_c = transforms.functional.crop(img_mask_c, i, j, h, w)
            img_mask_d = transforms.functional.crop(img_mask_d, i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_b = self.final_transform_op(img_b)
        img_mask_c = self.mask_op(img_mask_c)
        img_mask_d = self.mask_op(img_mask_d)

        return file_name, img_a, img_b, img_mask_c, img_mask_d

    def __len__(self):
        return len(self.image_list_a)

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
        a_img = cv2.imread(self.a_list[idx])
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        a_img = self.initial_op(a_img)

        b_img = cv2.imread(self.b_list[(idx % len(self.b_list))])
        b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
        b_img = self.initial_op(b_img)

        if(self.use_tanh):
            a_img = self.norm_op(a_img)
            b_img = self.norm_op(b_img)

        return a_img, b_img

    def __len__(self):
        return len(self.a_list)