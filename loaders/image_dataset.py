import os.path
import torch
import cv2
import numpy as np
from torch.utils import data
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional
import torch.nn.functional as F
import constants
import kornia
from pathlib import Path

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

class GenericPairedDataset(data.Dataset):
    def __init__(self, imgx_dir, img_ydir, transform_config, opts):
        self.imgx_dir = imgx_dir
        self.imgy_dir = img_ydir
        self.transform_config = transform_config
        self.patch_size = (opts.patch_size, opts.patch_size)

        if(transform_config == 1):
            # self.transform_op = transforms.Compose([
            #     transforms.ToPILImage(),
            #     transforms.Resize((256, 256)),
            #     transforms.RandomCrop(self.patch_size),
            #     transforms.RandomVerticalFlip(),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # ])

            self.transform_op = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        else:
            self.transform_op = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, idx):
        img_x_path = self.imgx_dir[idx]
        img_x = cv2.imread(img_x_path)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

        img_y_path = self.imgy_dir[idx % len(self.imgy_dir)]
        img_y = cv2.imread(img_y_path)
        img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)

        tensor_x = self.transform_op(img_x)
        tensor_y = self.transform_op(img_y)

        return tensor_x, tensor_y

    def __len__(self):
        return len(self.imgx_dir)

class RelightDataset(data.Dataset):
    def __init__(self, img_length, rgb_dir, albedo_dir, scene_names, opts):
        self.img_length = img_length
        self.rgb_dir = rgb_dir
        self.albedo_dir = albedo_dir
        self.scene_names = scene_names
        self.patch_size = (opts.patch_size, opts.patch_size)

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        file_name = "/synth_" + str(idx) + ".png"

        scene_index = np.random.randint(0, len(self.scene_names))
        rgb_path = self.rgb_dir + self.scene_names[scene_index] + file_name

        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        albedo_img = cv2.imread(self.albedo_dir + file_name)
        albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)

        rgb_tensor = self.initial_op(rgb_img)
        albedo_tensor = self.initial_op(albedo_img)

        scene_tensor = (scene_index / len(self.scene_names)) * 1.0 #normalize to 0.0 - 1.0
        scene_tensor = torch.tensor(scene_tensor)

        return file_name, rgb_tensor, albedo_tensor, scene_tensor

    def __len__(self):
        return self.img_length


class IIDDatasetV2(data.Dataset):
    def __init__(self, img_length, rgb_list_ws, rgb_dir_ns, unlit_dir, albedo_dir, transform_config, patch_size):
        self.img_length = img_length
        self.albedo_dir = albedo_dir
        self.unlit_dir = unlit_dir
        self.rgb_list_ws = rgb_list_ws
        self.rgb_dir_ns = rgb_dir_ns
        self.transform_config = transform_config
        self.patch_size = (patch_size, patch_size)
        self.light_angles = [0, 36, 72, 108, 144]

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        if (transform_config == 1):
            self.final_transform_op = transforms.Compose([
                transforms.ToTensor()
            ])

        else:
            self.final_transform_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        file_name = self.rgb_list_ws[idx].split("/")[-1].split(".png")[0] + ".png"
        scene_name = self.rgb_list_ws[idx].split("/")[-2]
        albedo_path = self.albedo_dir + file_name
        unlit_path = self.unlit_dir + file_name

        albedo = cv2.imread(albedo_path) #albedo
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

        unlit = cv2.imread(unlit_path)
        unlit = cv2.cvtColor(unlit, cv2.COLOR_BGR2RGB)

        input_rgb_ws = cv2.imread(self.rgb_list_ws[idx])  # input rgb
        input_rgb_ws = cv2.cvtColor(input_rgb_ws, cv2.COLOR_BGR2RGB)
        input_rgb_ns = cv2.imread(self.rgb_dir_ns + scene_name + "/" + file_name)
        input_rgb_ns = cv2.cvtColor(input_rgb_ns, cv2.COLOR_BGR2RGB)

        input_rgb_ws = self.initial_op(input_rgb_ws)
        input_rgb_ns = self.initial_op(input_rgb_ns)
        albedo = self.initial_op(albedo)
        unlit = self.initial_op(unlit)

        if (self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(input_rgb_ws, output_size=self.patch_size)
            i, j, h, w = crop_indices

            input_rgb_ws = transforms.functional.crop(input_rgb_ws, i, j, h, w)
            input_rgb_ns = transforms.functional.crop(input_rgb_ns, i, j, h, w)
            albedo = transforms.functional.crop(albedo, i, j, h, w)
            unlit = transforms.functional.crop(unlit, i, j, h, w)

        input_rgb_ws = self.final_transform_op(input_rgb_ws)
        input_rgb_ns = self.final_transform_op(input_rgb_ns)
        albedo = self.final_transform_op(albedo)
        unlit = self.final_transform_op(unlit)

        return file_name, input_rgb_ws, input_rgb_ns, albedo, unlit

    def __len__(self):
        return self.img_length

class UnlitDataset(data.Dataset):
    def __init__(self, img_length, rgb_list, unlit_dir, transform_config, opts):
        self.img_length = img_length
        self.unlit_dir = unlit_dir
        self.rgb_list = rgb_list
        self.transform_config = transform_config
        self.patch_size = (opts.patch_size, opts.patch_size)

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

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
        file_name = self.rgb_list[idx].split("\\")[-1].split(".png")[0] + ".png"
        img_a_path = self.unlit_dir + file_name

        unlit = cv2.imread(img_a_path)  # albedo
        unlit = cv2.cvtColor(unlit, cv2.COLOR_BGR2RGB)

        rgb_img = cv2.imread(self.rgb_list[idx])  # input rgb
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        rgb_img = self.initial_op(rgb_img)
        unlit = self.initial_op(unlit)

        if (self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(rgb_img, output_size=self.patch_size)
            i, j, h, w = crop_indices

            rgb_img = transforms.functional.crop(rgb_img, i, j, h, w)
            unlit = transforms.functional.crop(unlit, i, j, h, w)

        rgb_img = self.final_transform_op(rgb_img)
        unlit = self.final_transform_op(unlit)

        return file_name, rgb_img, unlit

    def __len__(self):
        return self.img_length


class RealWorldDataset(data.Dataset):
    def __init__(self, image_list_a):
        self.image_list_a = image_list_a

        self.initial_op = transforms.Compose([
            transforms.ToPILImage()])


        self.final_transform_op = transforms.Compose([
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_a = self.initial_op(img_a)
        img_a = self.final_transform_op(img_a)

        return file_name, img_a

    def __len__(self):
        return len(self.image_list_a)


class GTATestDataset(data.Dataset):
    def __init__(self, rgb_list, albedo_list, opts):
        self.rgb_list = rgb_list
        self.albedo_list = albedo_list
        self.patch_size = (opts.patch_size, opts.patch_size)
        self.light_angles = [0, 36, 72, 108, 144]

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        self.normalize_op = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normalize_op_grey = transforms.Normalize((0.5), (0.5))

        self.final_transform_op = transforms.Compose([
            # transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor()
        ])

        self.mask_op = transforms.Compose([
            # transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        # file_name = "synth_" + str(idx) + ".png"

        albedo = cv2.imread(self.albedo_list[idx])  # albedo
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

        input_rgb = cv2.imread(self.rgb_list[idx])  # input rgb
        input_rgb = cv2.cvtColor(input_rgb, cv2.COLOR_BGR2RGB)

        input_rgb = self.initial_op(input_rgb)
        albedo = self.initial_op(albedo)

        input_rgb = self.final_transform_op(input_rgb)
        albedo = self.final_transform_op(albedo)

        input_rgb = self.normalize_op(input_rgb)
        albedo = self.normalize_op(albedo)

        return input_rgb, albedo

    def __len__(self):
        return len(self.rgb_list)

class ShadowPriorDataset(data.Dataset):
    def __init__(self, image_list_a, transform_config, opts):
        self.image_list_a = image_list_a
        self.transform_config = transform_config
        self.patch_size = (opts.patch_size, opts.patch_size)

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        self.normalize_op_grey = transforms.Normalize((0.5), (0.5))

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
        img_id = self.image_list_a[idx]
        file_name = os.path.basename(img_id)

        # print(img_id)
        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)  # because matplot uses RGB, openCV is BGR
        img_a = self.initial_op(img_a)

        if (self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(img_a, output_size=self.patch_size)
            i, j, h, w = crop_indices
            img_a = transforms.functional.crop(img_a, i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_a = self.normalize_op_grey(img_a)
        return file_name, img_a

    def __len__(self):
        return len(self.image_list_a)