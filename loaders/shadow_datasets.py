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

from config import iid_server_config
from transforms import shadow_map_transforms

class ShadowTrainDataset(data.Dataset):
    def __init__(self, img_length, img_list_a, img_list_b, transform_config, patch_size = 0):
        self.img_length = img_length
        self.img_list_a = img_list_a
        self.img_list_b = img_list_b
        self.transform_config = transform_config
        self.patch_size = (patch_size, patch_size)

        self.shadow_op = shadow_map_transforms.ShadowMapTransforms()
        self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        network_config = sc_instance.interpret_shadow_network_params_from_version()

        if (network_config["augment_mode"] == "augmix" and self.transform_config == 1):
            self.initial_op = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.AugMix(),
                transforms.ToTensor()])
        elif (network_config["augment_mode"] == "trivial_augment_wide" and self.transform_config == 1):
            self.initial_op = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor()])
        else:
            self.initial_op = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor()])

    def __getitem__(self, idx):
        file_name = self.img_list_a[idx].split("/")[-1].split(".png")[0]

        try:
            rgb_ws = cv2.imread(self.img_list_a[idx])
            rgb_ws = cv2.cvtColor(rgb_ws, cv2.COLOR_BGR2RGB)
            state = torch.get_rng_state()
            rgb_ws = self.initial_op(rgb_ws)

            torch.set_rng_state(state)
            rgb_ns = cv2.imread(self.img_list_b[idx])
            rgb_ns = cv2.cvtColor(rgb_ns, cv2.COLOR_BGR2RGB)
            rgb_ns = self.initial_op(rgb_ns)

            if (self.transform_config == 1):
                crop_indices = transforms.RandomCrop.get_params(rgb_ws, output_size=self.patch_size)
                i, j, h, w = crop_indices

                rgb_ws = transforms.functional.crop(rgb_ws, i, j, h, w)
                rgb_ns = transforms.functional.crop(rgb_ns, i, j, h, w)

            rgb_ws, rgb_ns, shadow_map, shadow_matte = self.shadow_op.generate_shadow_map(rgb_ws, rgb_ns, False)

            rgb_ws = self.norm_op(rgb_ws)
            rgb_ns = self.norm_op(rgb_ns)
            shadow_map = self.norm_op(shadow_map)
            shadow_matte = self.norm_op(shadow_matte)

        except Exception as e:
            print("Failed to load: ", self.img_list_a[idx], self.img_list_b[idx])
            print("ERROR: ", e)
            rgb_ws = None
            rgb_ns = None
            shadow_map = None
            shadow_matte = None

        return file_name, rgb_ws, rgb_ns, shadow_map, shadow_matte

    def __len__(self):
        return self.img_length

class ShadowISTDDataset(data.Dataset):
    def __init__(self, img_length, img_list_a, img_list_b, img_list_c, transform_config):
        self.img_length = img_length
        self.img_list_a = img_list_a
        self.img_list_b = img_list_b
        self.img_list_c = img_list_c
        self.transform_config = transform_config

        self.shadow_op = shadow_map_transforms.ShadowMapTransforms()
        self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor()])

    def __getitem__(self, idx):
        file_name = self.img_list_a[idx].split("/")[-1].split(".png")[0]

        try:
            rgb_ws = cv2.imread(self.img_list_a[idx])
            rgb_ws = cv2.cvtColor(rgb_ws, cv2.COLOR_BGR2RGB)
            rgb_ws = self.initial_op(rgb_ws)

            rgb_ns = cv2.imread(self.img_list_b[idx])
            rgb_ns = cv2.cvtColor(rgb_ns, cv2.COLOR_BGR2RGB)
            rgb_ns = self.initial_op(rgb_ns)

            # shadow_mask = cv2.imread(self.img_list_c[idx])

            shadow_map = rgb_ns - rgb_ws
            shadow_matte = kornia.color.rgb_to_grayscale(shadow_map)

            rgb_ws = self.norm_op(rgb_ws)
            rgb_ns = self.norm_op(rgb_ns)
            shadow_matte = self.norm_op(shadow_matte)

        except Exception as e:
            print("Failed to load: ", self.img_list_a[idx], self.img_list_b[idx])
            print("ERROR: ", e)
            rgb_ws = None
            rgb_ns = None
            shadow_mask = None
            shadow_matte = None

        return file_name, rgb_ws, rgb_ns, shadow_matte

    def __len__(self):
        return self.img_length

class ShadowTestDataset(data.Dataset):
    def __init__(self, img_length, img_list_a, img_list_b, patch_size = 0):
        self.img_length = img_length
        self.img_list_a = img_list_a
        self.img_list_b = img_list_b
        self.patch_size = (patch_size, patch_size)

        self.initial_op = transforms.Compose([
            transforms.ToPILImage()])

        self.final_transform_op = transforms.Compose([
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, idx):
        file_name = self.img_list_a[idx].split("/")[-1].split(".png")[0]

        try:
            rgb_ws = cv2.imread(self.img_list_a[idx])
            rgb_ws = cv2.cvtColor(rgb_ws, cv2.COLOR_BGR2RGB)
            rgb_ws = self.initial_op(rgb_ws)
            rgb_ws = self.final_transform_op(rgb_ws)

            rgb_ns = cv2.imread(self.img_list_b[idx])
            rgb_ns = cv2.cvtColor(rgb_ns, cv2.COLOR_BGR2RGB)
            rgb_ns = self.initial_op(rgb_ns)
            rgb_ns = self.final_transform_op(rgb_ns)

        except:
            print("Failed to load: ", self.img_list_a[idx], self.img_list_b[idx])
            rgb_ws = None
            rgb_ns = None

        return file_name, rgb_ws, rgb_ns

    def __len__(self):
        return self.img_length