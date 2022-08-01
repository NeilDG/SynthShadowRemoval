import os.path
import torch
import cv2
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional
import constants
import kornia
from pathlib import Path

class CGIDataset(data.Dataset):
    def __init__(self, img_length, rgb_list_ws, transform_config, patch_size):
        self.img_length = img_length
        self.rgb_list_ws = rgb_list_ws
        self.transform_config = transform_config
        self.patch_size = (patch_size, patch_size)

        self.initial_op = transforms.Compose([
            transforms.ToPILImage()])

        if (transform_config == 1):
            self.final_transform_op = transforms.Compose([
                transforms.ToTensor()
            ])

            self.mask_op = transforms.Compose([
                transforms.ToTensor()
            ])

        else:
            self.final_transform_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor()
            ])

            self.mask_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        file_name = self.rgb_list_ws[idx].split("/")[-1].split(".png")[0]
        file_path = Path(self.rgb_list_ws[idx])

        albedo_path = str(file_path.parent) + "/" + file_name + "_albedo.png"
        mask_path = str(file_path.parent) + "/" + file_name + "_mask.png"

        albedo = cv2.imread(albedo_path) #albedo
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

        input_rgb_ws = cv2.imread(self.rgb_list_ws[idx])  # input rgb
        input_rgb_ws = cv2.cvtColor(input_rgb_ws, cv2.COLOR_BGR2RGB)

        # albedo_mask = cv2.imread(mask_path)
        # albedo_mask = cv2.cvtColor(albedo_mask, cv2.COLOR_BGR2GRAY)

        input_rgb_ws = self.initial_op(input_rgb_ws)
        albedo = self.initial_op(albedo)

        if (self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(input_rgb_ws, output_size=self.patch_size)
            i, j, h, w = crop_indices

            input_rgb_ws = transforms.functional.crop(input_rgb_ws, i, j, h, w)
            albedo = transforms.functional.crop(albedo, i, j, h, w)

        input_rgb_ws = self.final_transform_op(input_rgb_ws)
        albedo = self.final_transform_op(albedo)

        return file_name, input_rgb_ws, albedo

    def __len__(self):
        return self.img_length

class Bell2014Dataset(CGIDataset):
    def __init__(self, img_length, r_list, s_list, transform_config, patch_size):
        super().__init__(img_length, None, transform_config, patch_size)
        self.r_list = r_list
        self.s_list = s_list

    def __getitem__(self, idx):
        file_name = self.rgb_list_ws[idx].split("/")[-1].split(".png")[0]

        albedo = cv2.imread(self.r_list[idx]) #albedo
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

        shading = cv2.imread(self.s_list[idx])  # albedo
        shading = cv2.cvtColor(shading, cv2.COLOR_BGR2RGB)

        albedo = self.initial_op(albedo)
        shading = self.initial_op(shading)

        if (self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(albedo, output_size=self.patch_size)
            i, j, h, w = crop_indices

            albedo = transforms.functional.crop(albedo, i, j, h, w)
            shading = transforms.functional.crop(shading, i, j, h, w)

        albedo = self.final_transform_op(albedo)
        shading = self.final_transform_op(shading)
        rgb = albedo * shading

        return file_name, albedo, shading, rgb

    def __len__(self):
        return self.img_length

class IIWDataset(CGIDataset):
    def __init__(self, img_length, rgb_list):
        super().__init__(img_length, rgb_list, 2, (256, 256))

    def __getitem__(self, idx):
        file_name = self.rgb_list_ws[idx].split("/")[-1].split(".jpg")[0]

        try:
            rgb_img = cv2.imread(self.rgb_list_ws[idx])
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = self.initial_op(rgb_img)
            rgb_img = self.final_transform_op(rgb_img)
        except:
            print("Failed to load: ", self.rgb_list_ws[idx])
            rgb_img = None

        return file_name, rgb_img

    def __len__(self):
        return self.img_length